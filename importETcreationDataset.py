#!/usr/bin/env python3
"""
Script d'importation automatisée pour IDAH (Vidéos + Annotations VisDrone/MOT).

RÈGLES D'IMPORT :
    - Découvre automatiquement les dossiers de sets (J1_G1, J1_G2, etc.) sous IDAH_BASE_DIR.
    - Crée dynamiquement un Dataset sur IDAH pour chaque sous-dossier s'il n'existe pas encore.
    - Filtre strictement les annotations sur les 4 classes retenues :
        5: van, 6: truck, 8: awning-tricycle, 9: bus.
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

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
MIME_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}

MAX_SCAN_DEPTH = 3
IGNORED_CATEGORY_ID = 0

# Restriction aux 4 classes sélectionnées pour la campagne
ALLOWED_CATEGORIES = {
    5: "van",
    6: "truck",
    8: "awning-tricycle",
    9: "bus",
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
    TIMEOUT = (10, 300)
    NETWORK_RETRIES = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._relogin_in_progress = False

    @staticmethod
    def _rewind(kwargs):
        for item in (kwargs.get("files") or {}).values():
            stream = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else item
            if hasattr(stream, "seek"):
                stream.seek(0)

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self.TIMEOUT)
        for attempt in range(1, self.NETWORK_RETRIES + 1):
            try:
                res = super().request(method, url, **kwargs)
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as err:
                if attempt == self.NETWORK_RETRIES:
                    raise
                delay = 5 * attempt
                print(f"[!] Réseau ({type(err).__name__}), tentative {attempt + 1}/{self.NETWORK_RETRIES} dans {delay}s...")
                time.sleep(delay)
                self._rewind(kwargs)

        if (res.status_code != 401 or self._relogin_in_progress or url == CTX.get("auth_url")):
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
session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})

def new_parse_stats() -> dict:
    return {
        "kept": 0,
        "filtered_classes": 0,
        "score_zero": 0,
        "invalid_box": 0,
        "malformed": 0,
    }

def merge_parse_stats(target: dict, source: dict):
    for key, value in source.items():
        target[key] += value

ENTRY_CACHE = {}
ENTRY_CACHE_LOADED = False

# ==============================================================================
# AUTHENTIFICATION
# ==============================================================================

def login() -> str:
    print("[+] Connexion à l'API IDAH...")
    auth_payload = {"api_key": CTX["api_key"]}
    auth_response = session.post(CTX["auth_url"], json=auth_payload)

    if auth_response.status_code != 200:
        print(f"[!!] Échec de l'authentification (Code {auth_response.status_code}) : {auth_response.text}")
        exit(1)

    jwt_token = auth_response.json().get("meta", {}).get("token")
    if not jwt_token:
        print("[!!] Token JWT introuvable.")
        exit(1)

    session.cookies.set("auth-token", jwt_token, domain="idah.ingedata.ai")
    session.headers.update({"Authorization": f"Bearer {jwt_token}"})
    print("[+] Connexion réussie !\n")
    return jwt_token

# ==============================================================================
# DÉCOUVERTE DES SETS
# ==============================================================================

def is_set_dir(path: Path) -> bool:
    return (path / "videos").is_dir() and (path / "annotations").is_dir()

def discover_sets(root: Path) -> list:
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
    try:
        rel = set_dir.relative_to(root)
    except ValueError:
        return set_dir.name
    return set_dir.name if str(rel) == "." else "_".join(rel.parts)

def filter_sets(set_dirs: list, root: Path, wanted: list) -> list:
    if not wanted:
        return set_dirs

    by_label = {set_label(d, root).lower(): d for d in set_dirs}
    by_label.update({d.name.lower(): d for d in set_dirs})

    selected, missing = [], []
    for name in wanted:
        key = name.strip().lower()
        if not key: continue
        target = by_label.get(key)
        if target is None:
            missing.append(name)
        elif target not in selected:
            selected.append(target)

    if missing:
        print(f"[!!] Set(s) introuvable(s) : {', '.join(missing)}")
        exit(1)

    return selected

# ==============================================================================
# PARSER & UTILS
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
    return 1920.0, 1080.0

def find_annotation_file(anno_dir: Path, stem: str) -> Path:
    for ext in (".txt", ".json"):
        candidate = anno_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def parse_annotation_file(file_path: Path, video_path: Path, stats: dict = None) -> list:
    if stats is None: stats = new_parse_stats()
    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()
    if not content: return []

    img_w, img_h = get_video_dimensions(video_path)
    tracks = {}

    for line in content.splitlines():
        line = line.strip()
        if not line: continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            stats["malformed"] += 1
            continue

        try:
            frame_num = int(float(parts[0]))
            track_id = parts[1]
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        except ValueError:
            stats["malformed"] += 1
            continue

        if len(parts) > 6 and parts[6] and float(parts[6]) == 0.0:
            stats["score_zero"] += 1
            continue

        if w <= 0 or h <= 0:
            stats["invalid_box"] += 1
            continue

        # Filtrage strict sur les 4 classes sélectionnées
        if len(parts) > 7 and parts[7]:
            raw_category = parts[7]
            if raw_category.replace("-", "").isdigit():
                cat_id = int(raw_category)
                if cat_id not in ALLOWED_CATEGORIES:
                    stats["filtered_classes"] += 1
                    continue
                category = ALLOWED_CATEGORIES[cat_id]
            else:
                if raw_category not in ALLOWED_CATEGORIES.values():
                    stats["filtered_classes"] += 1
                    continue
                category = raw_category
        else:
            stats["filtered_classes"] += 1
            continue

        x_min = max(0.0, min(1.0, x / img_w))
        y_min = max(0.0, min(1.0, y / img_h))
        x_max = max(0.0, min(1.0, (x + w) / img_w))
        y_max = max(0.0, min(1.0, (y + h) / img_h))

        points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

        if track_id not in tracks:
            tracks[track_id] = {"category": category, "frames": []}

        tracks[track_id]["frames"].append({"angle": 0, "frame": frame_num, "points": points})
        stats["kept"] += 1

    annotations = []
    for track_id, t_data in tracks.items():
        sorted_frames = sorted(t_data["frames"], key=lambda item: item["frame"])
        if not sorted_frames: continue
        dims = {
            "type": "idah-video:bounding-box",
            "start": sorted_frames[0]["frame"],
            "end": sorted_frames[-1]["frame"],
            "frames": sorted_frames
        }
        annotations.append({"dimensions": dims, "category": t_data["category"], "track_id": str(track_id)})

    return annotations

# ==============================================================================
# REQUÊTES IDAH
# ==============================================================================

def select_dataset(name: str) -> str:
    """Bascule CTX['ds_id'] sur le dataset correspondant au dossier.
    S'il n'existe pas sur IDAH, il est créé automatiquement.
    """
    global ENTRY_CACHE_LOADED
    headers = {"Accept": "application/vnd.api+json"}
    url = f"{CTX['base_url']}/datasets?filter[project_id]={CTX['proj_id']}&page[size]=100"
    res = session.get(url, headers=headers)

    if not res.ok:
        print(f"[!!] Lecture des datasets du projet impossible (Code {res.status_code})")
        exit(1)

    matches = [d for d in res.json().get("data", [])
               if d.get("attributes", {}).get("name") == name]

    if matches:
        CTX["ds_id"] = str(matches[0]["id"])
        print(f"[+] Dataset sélectionné : '{name}' (ID: {CTX['ds_id']})")
    else:
        print(f"[+] Dataset '{name}' introuvable. Création sur IDAH...")
        # Payload direct avec project_id dans attributes (sans bloc relationships)
        payload = {
            "data": {
                "type": "dataset:datasets",
                "attributes": {
                    "name": name,
                    "project_id": CTX["proj_id"]
                }
            }
        }
        create_res = session.post(
            f"{CTX['base_url']}/datasets", 
            json=payload, 
            headers={"Content-Type": "application/vnd.api+json"}
        )

        if not create_res.ok:
            print(f"[!!] Échec de création du dataset '{name}' (Code {create_res.status_code}) : {create_res.text}")
            exit(1)

        new_ds_id = extract_id(create_res.json())
        CTX["ds_id"] = new_ds_id
        print(f"[✓] Dataset '{name}' créé avec succès ! (ID: {CTX['ds_id']})")

    ENTRY_CACHE.clear()
    ENTRY_CACHE_LOADED = False
    return CTX["ds_id"]

def load_entry_cache(force: bool = False):
    global ENTRY_CACHE_LOADED
    if ENTRY_CACHE_LOADED and not force: return
    ENTRY_CACHE.clear()
    headers = {"Accept": "application/vnd.api+json"}
    page = 1
    while True:
        url = f"{CTX['base_url']}/entries?filter[dataset_id]={CTX['ds_id']}&page[size]=100&page[number]={page}"
        res = session.get(url, headers=headers)
        if not res.ok: break
        items = res.json().get("data", [])
        if not items: break
        for item in items:
            name = item.get("attributes", {}).get("name")
            if name: ENTRY_CACHE[name] = str(item.get("id"))
        if len(items) < 100: break
        page += 1
    ENTRY_CACHE_LOADED = True

def query_entry_id(filename: str) -> str:
    load_entry_cache()
    if filename in ENTRY_CACHE:
        return ENTRY_CACHE[filename]
    load_entry_cache(force=True)
    return ENTRY_CACHE.get(filename)

def upload_media(video_path: Path, prefix: str) -> tuple:
    ext = video_path.suffix.lower()
    resource_name = f"{prefix}_{uuid.uuid4().hex[:16]}{ext}"
    url = f"{CTX['media_base_url']}/{resource_name}"
    with open(video_path, "rb") as f:
        files = {"file": (resource_name, f, MIME_TYPES.get(ext, "video/mp4"))}
        res = session.post(url, files=files, data={"project_id": CTX["proj_id"]}, timeout=(120, 900))
        res.raise_for_status()
        return extract_id(res.json()), resource_name

def create_entry(media_id: str, resource_name: str, filename: str) -> tuple:
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

def process_set(set_dir: Path, label: str) -> dict:
    clips_dir = set_dir / "videos"
    anno_dir = set_dir / "annotations"
    video_files = sorted([p for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS], key=lambda p: p.name)

    stats = {"set": label, "videos": len(video_files), "ok": 0, "skipped": 0, "failed": 0, "tracks_ok": 0, "tracks_ko": 0, "parse": new_parse_stats()}

    print("=" * 78)
    print(f"TRAITEMENT DU SET : {label}  ({set_dir})")
    print(f"{len(video_files)} vidéo(s) à traiter")
    print("=" * 78)

    for index, video_path in enumerate(video_files, start=1):
        anno_path = find_annotation_file(anno_dir, video_path.stem)
        if anno_path is None:
            print(f"[SKIP] Annotation manquante pour {video_path.name}")
            stats["skipped"] += 1
            continue

        entry_name = video_path.name
        print(f"\n--- [{label} {index}/{len(video_files)}] {video_path.name} ---")

        try:
            media_id, resource_name = upload_media(video_path, label.lower())
            print(f" -> Média téléversé (ID: {media_id})")

            entry_id, reused = create_entry(media_id, resource_name, entry_name)
            if not reused:
                print(f" -> Entrée créée (ID: {entry_id})")

            annotations = parse_annotation_file(anno_path, video_path, stats["parse"])
            for ann in annotations:
                ok, status, text = upload_annotation(entry_id, ann["dimensions"], ann["category"], ann["track_id"])
                if ok:
                    stats["tracks_ok"] += 1
                    print(f" -> Track #{ann['track_id']} ({ann['category']}) ajouté.")
                else:
                    stats["tracks_ko"] += 1
                    print(f" -> [ERREUR Track #{ann['track_id']}] Code {status}: {text}")

            stats["ok"] += 1

        except Exception as err:
            stats["failed"] += 1
            print(f"[ÉCHEC] Erreur sur {video_path.name}: {err}")

    return stats

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Import Vidéos + Annotations vers IDAH (Création automatique des datasets).")
    parser.add_argument("--base-dir", default=None, help="Dossier racine contenant vos sous-dossiers/datasets (ex: J1_G1, J1_G2).")
    parser.add_argument("--sets", default=None, help="Sous-ensemble de dossiers à importer (séparés par des virgules).")
    args = parser.parse_args()

    root = Path(args.base_dir).expanduser().resolve() if args.base_dir else BASE_DIR.resolve()
    wanted = [w.strip() for w in (args.sets or os.getenv("IDAH_SETS", "")).split(",") if w.strip()]

    set_dirs = filter_sets(discover_sets(root), root, wanted)

    print(f"[+] Racine : {root}")
    print(f"[+] {len(set_dirs)} dossier(s) détecté(s) : {', '.join(set_label(d, root) for d in set_dirs)}\n")

    login()

    all_stats = []
    for set_dir in set_dirs:
        label = set_label(set_dir, root)
        # Sélectionne ou crée automatiquement le dataset du nom du sous-dossier (ex: J1_G1)
        select_dataset(label)
        all_stats.append(process_set(set_dir, label))

    print("\n[+] Importation globale terminée avec succès !")

if __name__ == "__main__":
    main()