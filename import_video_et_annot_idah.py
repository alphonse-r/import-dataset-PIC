#!/usr/bin/env python3
"""
Script d'importation automatisée pour la plateforme IDAH (Vidéos & Annotations MOT Challenge).

USAGE :
    1. Assurez-vous d'avoir un fichier '.env' configuré à la racine du projet avec les variables :
       - API_KEY=<votre_cle_api>
       - IDAH_USER_EMAIL=<email_utilisateur>
       - IDAH_PROJECT_ID=<uuid_du_projet>
       - IDAH_DATASET_ID=<uuid_du_dataset>
       - IDAH_BASE_DIR=<chemin_du_dossier_racine> (ex: /mnt/shared/.../sets)
       - IDAH_SETS=<liste_optionnelle> (ex: A1,A2,B1 ; vide = tous les sets trouves)
       - IDAH_DEFAULT_CATEGORY=<categorie_de_repli> (utilisee seulement si l'id
         de categorie est absent ou inconnu de CATEGORY_MAP)
       - IDAH_ENTRY_PREFIX=1|0 (defaut 1 : prefixe le nom d'entree par le set,
         pour eviter les collisions entre sets ayant des clips homonymes)

    2. Lancement du script :
       python3 "import_video_et_annot_idah.py"
       python3 "import_video_et_annot_idah.py" --sets A1,B2          # sous-ensemble
       python3 "import_video_et_annot_idah.py" --base-dir /chemin    # override du .env
       python3 "import_video_et_annot_idah.py" --dry-run             # inventaire seul

PRÉREQUIS :
    - Python 3.8+
    - Paquets requises : requests, python-dotenv, opencv-python
    - Outil système optionnel : ffprobe

STRUCTURE DU DOSSIER SOURCE (<IDAH_BASE_DIR>) :
    IDAH_BASE_DIR/
    ├── A1/
    │   ├── videos/         -> Contient les fichiers vidéo (.mp4, .avi, .mov, .mkv)
    │   └── annotations/    -> Contient les .txt ayant le même nom que le clip
    ├── A2/ ...
    ├── B1/ ...
    ├── B2/ ...
    ├── C1/ ...
    └── C2/ ...

    Les sets sont découverts automatiquement : tout dossier contenant à la fois
    'videos/' et 'annotations/' est traité, quelle que soit sa profondeur
    (donc 'sets/C/C1' fonctionne aussi bien que 'sets/C1'). Si IDAH_BASE_DIR
    contient lui-même 'videos/' et 'annotations/', il est traité comme un set
    unique — l'ancien comportement reste donc valide.

FORMAT ET CONTENU DES ANNOTATIONS ATTENDUS :
    - Fichiers texte (.txt) au format standard MOT Challenge (séparateur virgule) :
      Format : <frame_id>, <track_id>, <x_min>, <y_min>, <width>, <height>, <confidence>, <category>, ...
      Exemple : 1, 36, 450.0, 200.0, 100.0, 80.0, 1, car
    - Conversion automatique : Le script extrait la résolution native de la vidéo (via cv2/ffprobe)
      et convertit chaque rectangle englobant en un polygone 4 points exprimé en coordonnées
      relatives comprises entre [0.0 - 1.0].

RÉSULTAT EN SORTIE :
    1. Connexion automatique via l'API_KEY et récupération d'un jeton JWT valide (une seule fois).
    2. Importation des médias sur le serveur de stockage avec un identifiant 'resource' unique (UUID)
       afin de contourner le soft-delete et éviter tout conflit de fichier existant.
    3. Création des entrées d'annotation ('entries') associées au nom de fichier d'origine.
    4. Enregistrement trame par trame des séquences de suivi d'objets (bounding boxes) sur IDAH.
    5. Récapitulatif par set puis récapitulatif global en fin d'exécution.
"""

import os
import time
import json
import uuid
import argparse
import subprocess
import collections
from pathlib import Path
import requests
from dotenv import load_dotenv

# Charge les variables d'environnement (.env)
load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(os.getenv("IDAH_BASE_DIR", ".")).expanduser()

# Extensions video prises en charge (MOT/VisDrone livrent parfois du .avi).
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
MIME_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}

# Profondeur maximale de recherche des sets sous IDAH_BASE_DIR.
MAX_SCAN_DEPTH = 3

# Correspondance des ids de categorie VisDrone (colonne 8 du format MOT).
# Identique a CATEGORY_MAP dans visdrone_to_video_coco.py.
#
# Format VisDrone-MOT, 10 colonnes :
#   <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
#   <score>,<object_category>,<truncation>,<occlusion>
#
# L'id 0 ("ignored regions") est volontairement ABSENT de cette table : ce n'est
# pas un objet mais une zone que l'annotation demande d'ignorer. Il est filtre
# en amont par IGNORED_CATEGORY_ID, comme dans visdrone_to_video_coco.py.
IGNORED_CATEGORY_ID = 0

CATEGORY_MAP = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others"
}

CTX = {
    "auth_url": "https://idah.ingedata.ai/api/v1/iam/auth/api/login",
    "api_key": os.getenv("API_KEY"),
    "email": os.getenv("IDAH_USER_EMAIL"),

    "base_url": "https://idah.ingedata.ai/api/v1/dataset",
    "media_base_url": "https://idah.ingedata.ai/api/v1/media/medias/files",
    "media_list_url": "https://idah.ingedata.ai/api/v1/media/medias",

    "proj_id": os.getenv("IDAH_PROJECT_ID"),
    "ds_id": os.getenv("IDAH_DATASET_ID"),
    "default_category": os.getenv("IDAH_DEFAULT_CATEGORY", "house")
}

class AuthSession(requests.Session):
    """Session qui se réauthentifie une fois si le JWT a expiré.

    Le token IDAH a une durée de vie plus courte qu'un import complet : sur
    210 vidéos, les dernières requêtes partaient avec un JWT périmé et
    échouaient en 401 (« Unauthorized » sur l'upload du média). Le script ne
    s'authentifiait qu'une fois, au démarrage.
    """

    # (connexion, lecture) en secondes. Sans timeout, requests attend
    # indéfiniment : un import s'est figé toute une nuit sur une socket morte,
    # sans message ni reprise possible.
    TIMEOUT = (10, 300)
    NETWORK_RETRIES = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._relogin_in_progress = False

    @staticmethod
    def _rewind(kwargs):
        """Rembobine les fichiers avant de rejouer une requête.

        Un upload consomme le fichier jusqu'à EOF : sans cela, la requête
        rejouée enverrait un corps vide et créerait un média vide, sans erreur.
        """
        for item in (kwargs.get("files") or {}).values():
            stream = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else item
            if hasattr(stream, "seek"):
                stream.seek(0)

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self.TIMEOUT)

        last_error = None
        for attempt in range(1, self.NETWORK_RETRIES + 1):
            try:
                res = super().request(method, url, **kwargs)
                break
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as err:
                last_error = err
                if attempt == self.NETWORK_RETRIES:
                    raise
                delay = 5 * attempt
                print(f"[!] Réseau ({type(err).__name__}), nouvelle tentative "
                      f"{attempt + 1}/{self.NETWORK_RETRIES} dans {delay}s...")
                time.sleep(delay)
                self._rewind(kwargs)

        # Le garde-fou évite la récursion : login() passe par cette session.
        if (res.status_code != 401
                or self._relogin_in_progress
                or url == CTX.get("auth_url")):
            return res

        print("[+] JWT expiré, réauthentification...")
        self._relogin_in_progress = True
        try:
            login()
        finally:
            self._relogin_in_progress = False

        self._rewind(kwargs)
        return super().request(method, url, **kwargs)

session = AuthSession()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
})

def new_parse_stats() -> dict:
    """Compteurs des lignes ecartees au parsing, remontes en fin d'execution."""
    return {
        "kept": 0,
        "ignored_regions": 0,   # colonne 8 == 0 : zone a ignorer, pas un objet
        "score_zero": 0,        # colonne 7 == 0 : ligne a ne pas prendre en compte
        "invalid_box": 0,       # largeur ou hauteur <= 0
        "malformed": 0,         # ligne illisible ou trop courte
        "unknown_ids": collections.Counter(),   # id hors CATEGORY_MAP -> repli
    }

def merge_parse_stats(target: dict, source: dict):
    for key, value in source.items():
        if key == "unknown_ids":
            target[key].update(value)
        else:
            target[key] += value

# Cache nom d'entree -> id, rempli une seule fois puis mis a jour en memoire.
# Evite de re-interroger l'API pour chaque video (O(n^2) sur 6 sets).
ENTRY_CACHE = {}
ENTRY_CACHE_LOADED = False

# ==============================================================================
# AUTHENTIFICATION API KEY
# ==============================================================================

def login() -> str:
    """Authentification via API_KEY et récupération du JWT Token."""
    print("[+] Connexion à l'API IDAH...")
    auth_payload = {"api_key": CTX["api_key"]}

    auth_response = session.post(CTX["auth_url"], json=auth_payload)

    if auth_response.status_code != 200:
        print(f"[!!] Échec de l'authentification (Code {auth_response.status_code})")
        print(auth_response.text)
        exit(1)

    auth_data = auth_response.json()
    jwt_token = auth_data.get("meta", {}).get("token")

    if not jwt_token:
        print("[!!] Impossible de trouver le JWT dans 'meta.token'")
        exit(1)

    session.cookies.set("auth-token", jwt_token, domain="idah.ingedata.ai")
    session.headers.update({"Authorization": f"Bearer {jwt_token}"})

    print("[+] Connexion réussie !! Token JWT configuré.\n")
    return jwt_token

# ==============================================================================
# DÉCOUVERTE DES SETS
# ==============================================================================

def is_set_dir(path: Path) -> bool:
    return (path / "videos").is_dir() and (path / "annotations").is_dir()

def discover_sets(root: Path) -> list:
    """Retourne la liste triée des dossiers contenant 'videos/' et 'annotations/'.

    Si 'root' est lui-même un set, il est renvoyé seul (compatibilité ascendante).
    """
    if not root.is_dir():
        print(f"[!!] Dossier racine introuvable : {root}")
        exit(1)

    if is_set_dir(root):
        return [root]

    found = []

    def walk(current: Path, depth: int):
        if depth > MAX_SCAN_DEPTH:
            return
        for child in sorted(current.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if is_set_dir(child):
                found.append(child)
            else:
                walk(child, depth + 1)

    walk(root, 1)
    return found

def set_label(set_dir: Path, root: Path) -> str:
    """Nom lisible du set : chemin relatif à la racine (ex: 'C/C1' -> 'C_C1')."""
    try:
        rel = set_dir.relative_to(root)
    except ValueError:
        return set_dir.name
    if str(rel) == ".":
        return set_dir.name
    return "_".join(rel.parts)

def filter_sets(set_dirs: list, root: Path, wanted: list) -> list:
    """Ne conserve que les sets demandés, dans l'ordre demandé."""
    if not wanted:
        return set_dirs

    by_label = {}
    for d in set_dirs:
        by_label[set_label(d, root).lower()] = d
        by_label[d.name.lower()] = d

    selected, missing = [], []
    for name in wanted:
        key = name.strip().lower()
        if not key:
            continue
        target = by_label.get(key)
        if target is None:
            missing.append(name)
        elif target not in selected:
            selected.append(target)

    if missing:
        available = ", ".join(sorted({set_label(d, root) for d in set_dirs})) or "aucun"
        print(f"[!!] Set(s) introuvable(s) : {', '.join(missing)}")
        print(f"     Sets disponibles : {available}")
        exit(1)

    return selected

# ==============================================================================
# UTILS & PARSER
# ==============================================================================

def extract_id(data) -> str:
    if not data: return None
    if isinstance(data, list) and len(data) > 0: return extract_id(data[0])
    if isinstance(data, dict):
        if "id" in data: return str(data["id"])
        if "data" in data: return extract_id(data["data"])
        for key in ["file", "media", "attributes"]:
            if key in data:
                res = extract_id(data[key])
                if res: return res
    return None

def get_video_dimensions(video_path: Path):
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        if w > 0 and h > 0:
            return float(w), float(h)
    except Exception:
        pass

    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
            str(video_path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        w, h = map(float, out.split("x"))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass

    return 1920.0, 1080.0

def find_annotation_file(anno_dir: Path, stem: str) -> Path:
    """Cherche l'annotation correspondant au clip (.txt puis .json)."""
    for ext in (".txt", ".json"):
        candidate = anno_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def parse_annotation_file(file_path: Path, video_path: Path, stats: dict = None) -> list:
    if stats is None:
        stats = new_parse_stats()

    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    if not content:
        return []

    if content.startswith("{") or content.startswith("["):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass

    img_w, img_h = get_video_dimensions(video_path)
    tracks = {}

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            stats["malformed"] += 1
            continue

        try:
            frame_num = int(float(parts[0]))
            track_id = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
        except ValueError:
            stats["malformed"] += 1
            continue

        try:
            # Colonne 7 (score) : 0 = ligne a ne pas prendre en compte.
            # Test sur == 0 exactement, pour ne pas ecarter une confiance
            # fractionnaire (0.9) que int() arrondirait a 0.
            if len(parts) > 6 and parts[6] and float(parts[6]) == 0.0:
                stats["score_zero"] += 1
                continue

            # Boite degeneree : rien a annoter.
            if w <= 0 or h <= 0:
                stats["invalid_box"] += 1
                continue

            category = CTX["default_category"]
            if len(parts) > 7 and parts[7]:
                raw_category = parts[7]
                if raw_category.replace("-", "").isdigit():
                    category_id = int(raw_category)
                    # Colonne 8 == 0 : region ignoree, ce n'est pas un objet.
                    if category_id == IGNORED_CATEGORY_ID:
                        stats["ignored_regions"] += 1
                        continue
                    if category_id in CATEGORY_MAP:
                        category = CATEGORY_MAP[category_id]
                    else:
                        # On ne masque plus silencieusement : le repli est
                        # applique mais l'id inconnu est remonte en fin d'import.
                        stats["unknown_ids"][category_id] += 1
                else:
                    category = raw_category

            x_min = max(0.0, min(1.0, x / img_w))
            y_min = max(0.0, min(1.0, y / img_h))
            x_max = max(0.0, min(1.0, (x + w) / img_w))
            y_max = max(0.0, min(1.0, (y + h) / img_h))

            points = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ]

            if track_id not in tracks:
                tracks[track_id] = {"category": category, "frames": []}

            tracks[track_id]["frames"].append({
                "angle": 0,
                "frame": frame_num,
                "points": points
            })
            stats["kept"] += 1
        except Exception:
            stats["malformed"] += 1
            continue

    annotations = []
    for track_id, t_data in tracks.items():
        sorted_frames = sorted(t_data["frames"], key=lambda item: item["frame"])
        if not sorted_frames:
            continue

        dims = {
            "type": "idah-video:bounding-box",
            "start": sorted_frames[0]["frame"],
            "end": sorted_frames[-1]["frame"],
            "frames": sorted_frames
        }

        annotations.append({
            "dimensions": dims,
            "category": t_data["category"],
            "track_id": str(track_id)
        })

    return annotations

# ==============================================================================
# REQUÊTES IDAH
# ==============================================================================

def select_dataset(name: str) -> str:
    """Bascule CTX['ds_id'] sur le dataset du projet portant ce nom.

    Utilisé par --dataset-per-set : la structure cible est un dataset par set
    (A1, A2, ...), et non un dataset unique. Le cache d'entrées est indexé par
    nom de fichier sans notion de dataset : il DOIT être vidé à chaque bascule,
    sinon les entrées du set précédent seraient prises pour celles du suivant.
    """
    global ENTRY_CACHE_LOADED
    headers = {"Accept": "application/vnd.api+json"}
    url = (f"{CTX['base_url']}/datasets"
           f"?filter[project_id]={CTX['proj_id']}&page[size]=100")
    res = session.get(url, headers=headers)
    if not res.ok:
        print(f"[!!] Lecture des datasets du projet impossible (Code {res.status_code})")
        exit(1)

    matches = [d for d in res.json().get("data", [])
               if d.get("attributes", {}).get("name") == name]
    if not matches:
        print(f"[!!] Aucun dataset nommé '{name}' dans le projet {CTX['proj_id']}.")
        print("     Le créer avec : .venv/bin/python create_idah_datasets.py "
              "--project <projet> --apply")
        exit(1)
    if len(matches) > 1:
        print(f"[!!] {len(matches)} datasets nommés '{name}' dans ce projet : "
              f"{[d['id'] for d in matches]}")
        print("     Impossible de choisir. Renommer les doublons côté plateforme.")
        exit(1)

    CTX["ds_id"] = matches[0]["id"]
    ENTRY_CACHE.clear()
    ENTRY_CACHE_LOADED = False
    print(f"[+] Dataset '{name}' -> {CTX['ds_id']}")
    return CTX["ds_id"]

def load_entry_cache(force: bool = False):
    """Charge (une fois) toutes les entrées du dataset, pagination comprise.

    Avec 6 sets, la liste dépasse largement une page : sans pagination, les
    entrées existantes des derniers sets seraient invisibles et recréées.
    """
    global ENTRY_CACHE_LOADED
    if ENTRY_CACHE_LOADED and not force:
        return

    ENTRY_CACHE.clear()
    # Note : l'API IDAH renvoie 400 si un Content-Type est envoye sur un GET.
    headers = {"Accept": "application/vnd.api+json"}
    page, page_size = 1, 100

    while True:
        url = (f"{CTX['base_url']}/entries"
               f"?filter[dataset_id]={CTX['ds_id']}"
               f"&page[size]={page_size}&page[number]={page}")
        res = session.get(url, headers=headers)
        if not res.ok:
            print(f"[!] Lecture des entrées existantes impossible (Code {res.status_code})")
            break

        body = res.json()
        items = body.get("data", []) if isinstance(body, dict) else body
        if not isinstance(items, list) or not items:
            break

        for item in items:
            if isinstance(item, dict):
                name = item.get("attributes", {}).get("name")
                if name:
                    ENTRY_CACHE[name] = str(item.get("id"))

        if len(items) < page_size:
            break
        page += 1

    ENTRY_CACHE_LOADED = True
    print(f"[+] {len(ENTRY_CACHE)} entrée(s) déjà présente(s) dans le dataset.\n")

def query_entry_id(filename: str) -> str:
    load_entry_cache()
    if filename in ENTRY_CACHE:
        return ENTRY_CACHE[filename]
    # Second essai : une entrée a pu être créée entre-temps (autre import).
    load_entry_cache(force=True)
    return ENTRY_CACHE.get(filename)

def upload_media(video_path: Path, prefix: str) -> tuple:
    ext = video_path.suffix.lower()
    random_key = uuid.uuid4().hex[:16]
    resource_name = f"{prefix}_{random_key}{ext}"

    url = f"{CTX['media_base_url']}/{resource_name}"
    with open(video_path, "rb") as f:
        files = {"file": (resource_name, f, MIME_TYPES.get(ext, "video/mp4"))}
        data = {"project_id": CTX["proj_id"]}
        # Timeout élargi pour l'envoi du corps. Le PREMIER terme compte ici :
        # dans urllib3, la phase d'écriture du corps est régie par le timeout
        # de connexion, pas par celui de lecture. Avec (10, 900), les trois
        # plus gros clips du corpus (9 à 27 Mo) échouaient en "write operation
        # timed out" après exactement 10 s, quel que soit l'état du réseau :
        # ils ne peuvent simplement pas partir en moins de 10 s. Les deux
        # valeurs restent bornées pour ne pas revenir au blocage infini.
        res = session.post(url, files=files, data=data, timeout=(120, 900))

        res.raise_for_status()
        res_json = res.json()

        media_id = extract_id(res_json)
        return media_id, resource_name

def create_entry(media_id: str, resource_name: str, filename: str) -> tuple:
    """Retourne (entry_id, reused: bool)."""
    existing_id = query_entry_id(filename)
    if existing_id:
        print(f" -> Entrée réutilisée (ID: {existing_id})")
        return existing_id, True

    endpoint = f"{CTX['base_url']}/entries"
    payload = {
        "data": {
            "type": "dataset:entries",
            "attributes": {
                "project_id": CTX["proj_id"],
                "resource": resource_name,
                "media_id": media_id,
                "name": filename
            },
            "relationships": {
                "dataset": {
                    "data": {
                        "type": "dataset:datasets",
                        "id": CTX["ds_id"]
                    }
                }
            }
        }
    }
    headers = {"Content-Type": "application/vnd.api+json"}
    res = session.post(endpoint, json=payload, headers=headers)

    if res.status_code == 422 and "already exists" in res.text:
        e_id = query_entry_id(filename)
        if e_id:
            return e_id, True

    res.raise_for_status()
    entry_id = extract_id(res.json())
    ENTRY_CACHE[filename] = entry_id
    return entry_id, False

def upload_annotation(entry_id: str, dimensions: dict, category: str, track_id: str = "1"):
    endpoint = f"{CTX['base_url']}/annotations"
    payload = {
        "data": {
            "type": "dataset:annotations",
            "attributes": {
                "project_id": CTX["proj_id"],
                "dataset_id": CTX["ds_id"],
                "entry_id": entry_id,
                "dimensions": dimensions,
                "annotation": {
                    "category": category,
                    "attributes": {"track_id": str(track_id)}
                },
                "created_by_email": CTX["email"]
            },
            "relationships": {
                "entry": {
                    "data": {
                        "type": "dataset:entries",
                        "id": entry_id
                    }
                }
            }
        }
    }
    headers = {"Content-Type": "application/vnd.api+json"}
    res = session.post(endpoint, json=payload, headers=headers)
    return res.ok, res.status_code, res.text

# ==============================================================================
# PIPELINE
# ==============================================================================

def list_videos(clips_dir: Path) -> list:
    videos = [p for p in clips_dir.iterdir()
              if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos, key=lambda p: p.name)

def process_set(set_dir: Path, label: str, use_prefix: bool) -> dict:
    """Importe un set complet. Retourne les compteurs du set."""
    clips_dir = set_dir / "videos"
    anno_dir = set_dir / "annotations"
    video_files = list_videos(clips_dir)

    stats = {"set": label, "videos": len(video_files), "ok": 0,
             "skipped": 0, "failed": 0, "tracks_ok": 0, "tracks_ko": 0,
             "parse": new_parse_stats()}

    print("=" * 78)
    print(f"SET {label}  ({set_dir})")
    print(f"{len(video_files)} vidéo(s) détectée(s)")
    print("=" * 78)

    for index, video_path in enumerate(video_files, start=1):
        anno_path = find_annotation_file(anno_dir, video_path.stem)

        if anno_path is None:
            print(f"[SKIP] Annotation manquante pour {label}/{video_path.name}")
            stats["skipped"] += 1
            continue

        entry_name = f"{label}_{video_path.name}" if use_prefix else video_path.name
        print(f"\n--- [{label} {index}/{len(video_files)}] {video_path.name} ---")

        try:
            # 1. Upload Média avec UUID unique
            media_id, resource_name = upload_media(video_path, label.lower())
            print(f" -> Média prêt (ID: {media_id}, Resource: {resource_name})")

            # 2. Création Entrée (Resource UUID + Nom lisible prefixe par le set)
            entry_id, reused = create_entry(media_id, resource_name, entry_name)
            if not reused:
                print(f" -> Entrée prête (ID: {entry_id})")

            # 3. Import Annotations MOT
            annotations = parse_annotation_file(anno_path, video_path, stats["parse"])
            for ann in annotations:
                ok, status, text = upload_annotation(
                    entry_id=entry_id,
                    dimensions=ann["dimensions"],
                    category=ann["category"],
                    track_id=ann["track_id"]
                )

                if ok:
                    stats["tracks_ok"] += 1
                    print(f" -> Track #{ann['track_id']} ({ann['category']}) enregistré.")
                else:
                    stats["tracks_ko"] += 1
                    print(f" -> [ERREUR Annotation Track #{ann['track_id']}] Code {status}: {text}")

            stats["ok"] += 1

        except Exception as err:
            stats["failed"] += 1
            print(f"[ÉCHEC] Erreur sur {label}/{video_path.name}: {err}")

    return stats

def print_summary(all_stats: list):
    print("\n" + "=" * 78)
    print("RÉCAPITULATIF GLOBAL")
    print("=" * 78)
    header = f"{'SET':<14}{'VIDÉOS':>8}{'OK':>7}{'SKIP':>7}{'ÉCHEC':>8}{'TRACKS':>9}{'TR.KO':>8}"
    print(header)
    print("-" * 78)

    totals = {"videos": 0, "ok": 0, "skipped": 0, "failed": 0, "tracks_ok": 0, "tracks_ko": 0}
    parse_totals = new_parse_stats()
    for s in all_stats:
        print(f"{s['set']:<14}{s['videos']:>8}{s['ok']:>7}{s['skipped']:>7}"
              f"{s['failed']:>8}{s['tracks_ok']:>9}{s['tracks_ko']:>8}")
        for key in totals:
            totals[key] += s[key]
        merge_parse_stats(parse_totals, s["parse"])

    print("-" * 78)
    print(f"{'TOTAL':<14}{totals['videos']:>8}{totals['ok']:>7}{totals['skipped']:>7}"
          f"{totals['failed']:>8}{totals['tracks_ok']:>9}{totals['tracks_ko']:>8}")
    print("=" * 78)

    print_parse_report(parse_totals, all_stats)

def print_parse_report(parse_totals: dict, all_stats: list):
    """Detaille les lignes ecartees au parsing et les ids de categorie inconnus."""
    labels = [
        ("ignored_regions", "regions ignorees (categorie 0)"),
        ("score_zero", "score 0 (ligne a ne pas prendre en compte)"),
        ("invalid_box", "boites degenerees (largeur ou hauteur <= 0)"),
        ("malformed", "lignes illisibles"),
    ]
    ecartees = sum(parse_totals[key] for key, _ in labels)

    print("\nLIGNES SOURCES")
    print(f"  retenues : {parse_totals['kept']}")
    if ecartees == 0:
        print("  ecartees : 0")
    else:
        print(f"  ecartees : {ecartees}")
        for key, text in labels:
            if parse_totals[key]:
                print(f"    - {text} : {parse_totals[key]}")

    unknown = parse_totals["unknown_ids"]
    if not unknown:
        return

    # Un id hors CATEGORY_MAP part dans IDAH_DEFAULT_CATEGORY : sans cet
    # avertissement, la confusion resterait invisible dans la plateforme.
    fallback = CTX["default_category"]
    print(f"\n[!] IDS DE CATEGORIE INCONNUS -> importes en \"{fallback}\"")
    for cid, count in sorted(unknown.items()):
        print(f"    - id {cid} : {count} ligne(s)")
    concernes = [s["set"] for s in all_stats if s["parse"]["unknown_ids"]]
    print(f"    Set(s) concerne(s) : {', '.join(concernes)}")
    print("    Completer CATEGORY_MAP puis reimporter ces sets.")

def main():
    parser = argparse.ArgumentParser(
        description="Import multi-sets (A1, A2, B1, B2, C1, C2...) de vidéos + annotations vers IDAH."
    )
    parser.add_argument("--base-dir", default=None,
                        help="Dossier racine contenant les sets (override IDAH_BASE_DIR).")
    parser.add_argument("--sets", default=None,
                        help="Liste de sets à importer, séparés par des virgules (défaut : tous).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche l'inventaire des sets sans rien envoyer à IDAH.")
    parser.add_argument("--no-prefix", action="store_true",
                        help="Ne pas préfixer le nom d'entrée par le nom du set.")
    parser.add_argument("--dataset-per-set", action="store_true",
                        help="Envoyer chaque set dans le dataset du projet portant "
                             "son nom (A1 -> dataset 'A1'), au lieu de tout mettre "
                             "dans IDAH_DATASET_ID. Désactive le préfixe d'entrée.")
    args = parser.parse_args()

    root = Path(args.base_dir).expanduser() if args.base_dir else BASE_DIR
    root = root.resolve()

    wanted = (args.sets or os.getenv("IDAH_SETS", "")).split(",")
    wanted = [w for w in (x.strip() for x in wanted) if w]

    use_prefix = not args.no_prefix and os.getenv("IDAH_ENTRY_PREFIX", "1") not in ("0", "false", "False")

    # Le préfixe existe pour éviter que deux clips homonymes de sets différents
    # partagent une entrée. Avec un dataset par set, la collision est
    # impossible : le préfixe n'apporterait qu'un nom plus long.
    if args.dataset_per_set:
        use_prefix = False

    set_dirs = discover_sets(root)
    if not set_dirs:
        print(f"[!!] Aucun set trouvé sous {root}")
        print("     Un set est un dossier contenant à la fois 'videos/' et 'annotations/'.")
        exit(1)

    set_dirs = filter_sets(set_dirs, root, wanted)

    print(f"[+] Racine : {root}")
    print(f"[+] {len(set_dirs)} set(s) à traiter : "
          f"{', '.join(set_label(d, root) for d in set_dirs)}\n")

    if args.dry_run:
        total = 0
        all_stats = []
        categories = collections.Counter()
        for set_dir in set_dirs:
            label = set_label(set_dir, root)
            videos = list_videos(set_dir / "videos")
            missing = []
            parse_stats = new_parse_stats()
            tracks = 0
            for v in videos:
                anno = find_annotation_file(set_dir / "annotations", v.stem)
                if anno is None:
                    missing.append(v.name)
                    continue
                for ann in parse_annotation_file(anno, v, parse_stats):
                    tracks += 1
                    categories[ann["category"]] += 1
            total += len(videos)
            all_stats.append({"set": label, "parse": parse_stats})
            print(f"  {label:<12} {len(videos):>4} vidéo(s)  {tracks:>5} track(s)"
                  + (f"  |  {len(missing)} sans annotation : {', '.join(missing[:5])}"
                     + (" ..." if len(missing) > 5 else "") if missing else ""))

        print(f"\n  Categories : {dict(categories.most_common())}")

        parse_totals = new_parse_stats()
        for s in all_stats:
            merge_parse_stats(parse_totals, s["parse"])
        print_parse_report(parse_totals, all_stats)

        print(f"\n[dry-run] Total : {total} vidéo(s). Aucun envoi effectué.")
        return

    # Une seule authentification pour l'ensemble des sets.
    login()

    all_stats = []
    for set_dir in set_dirs:
        label = set_label(set_dir, root)
        if args.dataset_per_set:
            select_dataset(label)
        all_stats.append(process_set(set_dir, label, use_prefix))

    print_summary(all_stats)

if __name__ == "__main__":
    main()
