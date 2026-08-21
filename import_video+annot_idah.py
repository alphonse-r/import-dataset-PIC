#!/usr/bin/env python3
"""
Script d'importation automatisée pour la plateforme IDAH (Vidéos & Annotations MOT Challenge).

USAGE :
    1. Assurez-vous d'avoir un fichier '.env' configuré à la racine du projet avec les variables :
       - API_KEY=<votre_cle_api>
       - IDAH_USER_EMAIL=<email_utilisateur>
       - IDAH_PROJECT_ID=<uuid_du_projet>
       - IDAH_DATASET_ID=<uuid_du_dataset>
       - IDAH_BASE_DIR=<chemin_du_dossier_source> (ex: /mnt/shared/.../sets/C/C1)
       - IDAH_DEFAULT_CATEGORY=<categorie_par_defaut> (ex: house, car)

    2. Lancement du script :
       python3 all_idah.py

PRÉREQUIS :
    - Python 3.8+
    - Paquets requises : requests, python-dotenv, opencv-python
    - Outil système optionnel : ffprobe

STRUCTURE DU DOSSIER SOURCE (<IDAH_BASE_DIR>) :
    IDAH_BASE_DIR/
    ├── clips/          -> Contient les fichiers vidéo (.mp4)
    └── annotations/    -> Contient les fichiers d'annotations (.txt) ayant le même nom que le clip

FORMAT ET CONTENU DES ANNOTATIONS ATTENDUS :
    - Fichiers texte (.txt) au format standard MOT Challenge (séparateur virgule) :
      Format : <frame_id>, <track_id>, <x_min>, <y_min>, <width>, <height>, <confidence>, <category>, ...
      Exemple : 1, 36, 450.0, 200.0, 100.0, 80.0, 1, car
    - Conversion automatique : Le script extrait la résolution native de la vidéo (via cv2/ffprobe)
      et convertit chaque rectangle englobant en un polygone 4 points exprimé en coordonnées
      relatives comprises entre [0.0 - 1.0].

RÉSULTAT EN SORTIE :
    1. Connexion automatique via l'API_KEY et récupération d'un jeton JWT valide.
    2. Importation des médias sur le serveur de stockage avec un identifiant 'resource' unique (UUID)
       afin de contourner le soft-delete et éviter tout conflit de fichier existant.
    3. Création des entrées d'annotation ('entries') associées au nom de fichier d'origine.
    4. Enregistrement trame par trame des séquences de suivi d'objets (bounding boxes) sur IDAH.
"""

import os
import json
import uuid
import subprocess
from pathlib import Path
import requests
from dotenv import load_dotenv

# Charge les variables d'environnement (.env)
load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(os.getenv("IDAH_BASE_DIR", "."))

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

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
})

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

def parse_annotation_file(file_path: Path, video_path: Path) -> list:
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
            continue

        try:
            frame_num = int(float(parts[0]))
            track_id = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            category = CTX["default_category"]
            if len(parts) > 7 and parts[7] and not parts[7].replace("-", "").isdigit():
                category = parts[7]

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
        except Exception:
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

def query_entry_id(filename: str) -> str:
    headers = {"Content-Type": "application/vnd.api+json"}
    url = f"{CTX['base_url']}/entries?filter[dataset_id]={CTX['ds_id']}&page[size]=100"
    res = session.get(url, headers=headers)
    if res.ok:
        body = res.json()
        items = body.get("data", []) if isinstance(body, dict) else body
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    attrs = item.get("attributes", {})
                    if attrs.get("name") == filename:
                        return str(item.get("id"))
    return None

def upload_media(video_path: Path) -> tuple:
    ext = video_path.suffix.lower()
    random_key = uuid.uuid4().hex[:16]
    resource_name = f"c1_{random_key}{ext}"

    url = f"{CTX['media_base_url']}/{resource_name}"
    with open(video_path, "rb") as f:
        files = {"file": (resource_name, f, "video/mp4")}
        data = {"project_id": CTX["proj_id"]}
        res = session.post(url, files=files, data=data)
        
        res.raise_for_status()
        res_json = res.json()
        
        media_id = extract_id(res_json)
        return media_id, resource_name

def create_entry(media_id: str, resource_name: str, filename: str) -> str:
    existing_id = query_entry_id(filename)
    if existing_id:
        print(f" -> Entrée réutilisée (ID: {existing_id})")
        return existing_id

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
        if e_id: return e_id

    res.raise_for_status()
    return extract_id(res.json())

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

def process_set_c1():
    login()

    clips_dir = BASE_DIR / "clips"
    anno_dir = BASE_DIR / "annotations"

    video_files = sorted(clips_dir.glob("*.mp4"))
    print(f"Lancement de l'import C1 : {len(video_files)} vidéos détectées...")

    for video_path in video_files:
        stem = video_path.stem
        anno_path = anno_dir / f"{stem}.txt"

        if not anno_path.exists():
            print(f"[SKIP] Annotation manquante pour {video_path.name}")
            continue

        print(f"\n--- Traitement : {video_path.name} ---")

        try:
            # 1. Upload Média avec UUID unique
            media_id, resource_name = upload_media(video_path)
            print(f" -> Média prêt (ID: {media_id}, Resource: {resource_name})")

            # 2. Création Entrée (Resource UUID + Nom lisible)
            entry_id = create_entry(media_id, resource_name, video_path.name)
            print(f" -> Entrée prête (ID: {entry_id})")

            # 3. Import Annotations MOT
            annotations = parse_annotation_file(anno_path, video_path)
            for ann in annotations:
                ok, status, text = upload_annotation(
                    entry_id=entry_id,
                    dimensions=ann["dimensions"],
                    category=ann["category"],
                    track_id=ann["track_id"]
                )

                if ok:
                    print(f" -> Track #{ann['track_id']} ({ann['category']}) enregistré.")
                else:
                    print(f" -> [ERREUR Annotation Track #{ann['track_id']}] Code {status}: {text}")

        except Exception as err:
            print(f"[ÉCHEC] Erreur sur {video_path.name}: {err}")

if __name__ == "__main__":
    process_set_c1()