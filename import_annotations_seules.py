#!/usr/bin/env python3
"""
Script d'importation d'annotations SEULES pour IDAH (Filtrées).

Rattache les annotations (van, truck, awning-tricycle, bus) aux 'entries' existantes sur IDAH.
"""

import os
import argparse
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(os.getenv("IDAH_BASE_DIR", ".")).expanduser()

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
    "proj_id": os.getenv("IDAH_PROJECT_ID"),
    "ds_id": os.getenv("IDAH_DATASET_ID"),
}

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})

ENTRY_CACHE = {}

def login():
    auth_payload = {"api_key": CTX["api_key"]}
    auth_response = session.post(CTX["auth_url"], json=auth_payload)
    if auth_response.status_code != 200:
        print(f"[!!] Échec de l'authentification (Code {auth_response.status_code})")
        exit(1)

    jwt_token = auth_response.json().get("meta", {}).get("token")
    if not jwt_token:
        exit(1)

    session.cookies.set("auth-token", jwt_token, domain="idah.ingedata.ai")
    session.headers.update({"Authorization": f"Bearer {jwt_token}"})

def load_entry_cache():
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
            if name:
                ENTRY_CACHE[name] = str(item.get("id"))
        if len(items) < 100: break
        page += 1
    print(f"[+] {len(ENTRY_CACHE)} entrée(s) chargée(s) depuis IDAH.")

def get_video_dimensions(video_path: Path):
    if video_path.exists():
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            if w > 0 and h > 0: return float(w), float(h)
        except Exception:
            pass
    return 1920.0, 1080.0

def parse_annotation_file(file_path: Path, video_path: Path) -> list:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()
    if not content: return []

    img_w, img_h = get_video_dimensions(video_path)
    tracks = {}

    for line in content.splitlines():
        line = line.strip()
        if not line: continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6: continue

        try:
            frame_num = int(float(parts[0]))
            track_id = parts[1]
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        except ValueError:
            continue

        if len(parts) > 6 and parts[6] and float(parts[6]) == 0.0: continue
        if w <= 0 or h <= 0: continue

        # Filtrage strict sur les 4 classes autorisées
        if len(parts) > 7 and parts[7]:
            raw_cat = parts[7]
            if raw_cat.replace("-", "").isdigit():
                cat_id = int(raw_cat)
                if cat_id not in ALLOWED_CATEGORIES:
                    continue
                category = ALLOWED_CATEGORIES[cat_id]
            else:
                if raw_cat not in ALLOWED_CATEGORIES.values():
                    continue
                category = raw_cat
        else:
            continue

        x_min, y_min = max(0.0, min(1.0, x / img_w)), max(0.0, min(1.0, y / img_h))
        x_max, y_max = max(0.0, min(1.0, (x + w) / img_w)), max(0.0, min(1.0, (y + h) / img_h))
        points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

        if track_id not in tracks:
            tracks[track_id] = {"category": category, "frames": []}

        tracks[track_id]["frames"].append({"angle": 0, "frame": frame_num, "points": points})

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
                "annotation": {"category": category, "attributes": {"track_id": str(track_id)}},
                "created_by_email": CTX["email"]
            },
            "relationships": {"entry": {"data": {"type": "dataset:entries", "id": entry_id}}}
        }
    }
    headers = {"Content-Type": "application/vnd.api+json"}
    res = session.post(endpoint, json=payload, headers=headers)
    return res.ok

def main():
    parser = argparse.ArgumentParser(description="Import d'annotations seules (van, truck, awning-tricycle, bus) vers IDAH.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--no-prefix", action="store_true")
    args = parser.parse_args()

    root = Path(args.base_dir).expanduser().resolve() if args.base_dir else BASE_DIR.resolve()
    use_prefix = not args.no_prefix and os.getenv("IDAH_ENTRY_PREFIX", "1") not in ("0", "false", "False")

    login()
    load_entry_cache()

    for txt_file in root.rglob("*.txt"):
        if "annotations" not in txt_file.parts: continue

        set_label = txt_file.parent.parent.name
        video_name = f"{txt_file.stem}.mp4"
        entry_name = f"{set_label}_{video_name}" if use_prefix else video_name

        entry_id = ENTRY_CACHE.get(entry_name)
        if not entry_id:
            print(f"[SKIP] Aucune entrée trouvée sur IDAH pour {entry_name}")
            continue

        video_path = txt_file.parent.parent / "videos" / video_name
        annotations = parse_annotation_file(txt_file, video_path)

        count = 0
        for ann in annotations:
            if upload_annotation(entry_id, ann["dimensions"], ann["category"], ann["track_id"]):
                count += 1

        print(f"[✓] {count} track(s) annoté(s) envoyé(s) pour : {entry_name}")

if __name__ == "__main__":
    main()