#!/usr/bin/env python3
"""
Script d'importation automatisée complète sur IDAH : Vidéos + Annotations (Filtrées).

Classes autorisées :
  - 5 : van
  - 6 : truck
  - 8 : awning-tricycle
  - 9 : bus
"""

import os
import time
import json
import uuid
import argparse
import collections
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

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
    "proj_id": os.getenv("IDAH_PROJECT_ID"),
    "ds_id": os.getenv("IDAH_DATASET_ID"),
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
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == self.NETWORK_RETRIES:
                    raise
                time.sleep(5 * attempt)
                self._rewind(kwargs)

        if (res.status_code != 401 or self._relogin_in_progress or url == CTX.get("auth_url")):
            return res

        self._relogin_in_progress = True
        try:
            login()
        finally:
            self._relogin_in_progress = False

        self._rewind(kwargs)
        return super().request(method, url, **kwargs)

session = AuthSession()
session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})

ENTRY_CACHE = {}
ENTRY_CACHE_LOADED = False

def new_parse_stats() -> dict:
    return {
        "kept": 0,
        "filtered_classes": 0,
        "score_zero": 0,
        "invalid_box": 0,
        "malformed": 0,
    }

def login() -> str:
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
    return jwt_token

def is_set_dir(path: Path) -> bool:
    return (path / "videos").is_dir() and (path / "annotations").is_dir()

def discover_sets(root: Path) -> list:
    if not root.is_dir():
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
    selected = []
    for name in wanted:
        key = name.strip().lower()
        if key in by_label and by_label[key] not in selected:
            selected.append(by_label[key])
    return selected

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
        if w > 0 and h > 0: return float(w), float(h)
    except Exception:
        pass
    return 1920.0, 1080.0

def find_annotation_file(anno_dir: Path, stem: str) -> Path:
    for ext in (".txt", ".json"):
        candidate = anno_dir / f"{stem}{ext}"
        if candidate.exists(): return candidate
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

        # Filtrage strict sur la 8e colonne (classes VisDrone autorisées)
        if len(parts) > 7 and parts[7]:
            raw_cat = parts[7]
            if raw_cat.replace("-", "").isdigit():
                cat_id = int(raw_cat)
                if cat_id not in ALLOWED_CATEGORIES:
                    stats["filtered_classes"] += 1
                    continue
                category = ALLOWED_CATEGORIES[cat_id]
            else:
                if raw_cat not in ALLOWED_CATEGORIES.values():
                    stats["filtered_classes"] += 1
                    continue
                category = raw_cat
        else:
            stats["filtered_classes"] += 1
            continue

        x_min, y_min = max(0.0, min(1.0, x / img_w)), max(0.0, min(1.0, y / img_h))
        x_max, y_max = max(0.0, min(1.0, (x + w) / img_w)), max(0.0, min(1.0, (y + h) / img_h))

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
    load_entry_cache()
    if filename in ENTRY_CACHE:
        return ENTRY_CACHE[filename], True

    endpoint = f"{CTX['base_url']}/entries"
    payload = {
        "data": {
            "type": "dataset:entries",
            "attributes": {"project_id": CTX["proj_id"], "resource": resource_name, "media_id": media_id, "name": filename},
            "relationships": {"dataset": {"data": {"type": "dataset:datasets", "id": CTX["ds_id"]}}}
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
                "annotation": {"category": category, "attributes": {"track_id": str(track_id)}},
                "created_by_email": CTX["email"]
            },
            "relationships": {"entry": {"data": {"type": "dataset:entries", "id": entry_id}}}
        }
    }
    headers = {"Content-Type": "application/vnd.api+json"}
    res = session.post(endpoint, json=payload, headers=headers)
    return res.ok, res.status_code, res.text

def process_set(set_dir: Path, label: str, use_prefix: bool) -> dict:
    clips_dir = set_dir / "videos"
    anno_dir = set_dir / "annotations"
    video_files = sorted([p for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS], key=lambda x: x.name)

    stats = {"set": label, "videos": len(video_files), "ok": 0, "skipped": 0, "failed": 0, "tracks_ok": 0, "tracks_ko": 0, "parse": new_parse_stats()}

    for index, video_path in enumerate(video_files, start=1):
        anno_path = find_annotation_file(anno_dir, video_path.stem)
        if anno_path is None:
            stats["skipped"] += 1
            continue

        entry_name = f"{label}_{video_path.name}" if use_prefix else video_path.name
        try:
            media_id, resource_name = upload_media(video_path, label.lower())
            entry_id, _ = create_entry(media_id, resource_name, entry_name)

            annotations = parse_annotation_file(anno_path, video_path, stats["parse"])
            for ann in annotations:
                ok, _, _ = upload_annotation(entry_id, ann["dimensions"], ann["category"], ann["track_id"])
                if ok: stats["tracks_ok"] += 1
                else: stats["tracks_ko"] += 1
            stats["ok"] += 1
            print(f"[✓] [{label}] Vidéo + Annotations importées : {entry_name}")
        except Exception as e:
            stats["failed"] += 1
            print(f"[X] [{label}] Échec sur {entry_name} : {e}")

    return stats

def main():
    # parser = argparse.ArgumentParser(description="Import Vidéos + Annotations (van, truck, awning-tricycle, bus) vers IDAH.")
    # parser.add_argument("--base-dir", default=None)
    # parser.add_argument("--sets", default=None)
    # parser.add_argument("--no-prefix", action="store_true")
    # args = parser.parse_args()

    # root = Path(args.base_dir).expanduser().resolve() if args.base_dir else BASE_DIR.resolve()
    # wanted = [w.strip() for w in (args.sets or os.getenv("IDAH_SETS", "")).split(",") if w.strip()]
    # use_prefix = not args.no_prefix and os.getenv("IDAH_ENTRY_PREFIX", "1") not in ("0", "false", "False")

    # set_dirs = filter_sets(discover_sets(root), root, wanted)
    # login()

    # all_stats = [process_set(s, set_label(s, root), use_prefix) for s in set_dirs]
    # print("[+] Processus d'importation vidéo + annotations filtrées terminé.")

    def main():
        print("[DEBUG] Démarrage du script...")
        
        parser = argparse.ArgumentParser(description="Import Vidéos + Annotations (van, truck, awning-tricycle, bus) vers IDAH.")
        parser.add_argument("--base-dir", default=None)
        parser.add_argument("--sets", default=None)
        parser.add_argument("--no-prefix", action="store_true")
        args = parser.parse_args()

        root = Path(args.base_dir).expanduser().resolve() if args.base_dir else BASE_DIR.resolve()
        print(f"[DEBUG] Dossier racine analysé : {root}")

        wanted = [w.strip() for w in (args.sets or os.getenv("IDAH_SETS", "")).split(",") if w.strip()]
        use_prefix = not args.no_prefix and os.getenv("IDAH_ENTRY_PREFIX", "1") not in ("0", "false", "False")

        sets_trouves = discover_sets(root)
        print(f"[DEBUG] Sets trouvés sous la racine : {sets_trouves}")

        set_dirs = filter_sets(sets_trouves, root, wanted)
        print(f"[DEBUG] Sets retenus après filtrage : {set_dirs}")

        if not set_dirs:
            print("[X] AUCUN SET TROUVÉ ! Vérifiez que vos sous-dossiers contiennent bien les dossiers 'videos' et 'annotations'.")
            return

        print("[DEBUG] Tentative de connexion à IDAH...")
        login()

        all_stats = [process_set(s, set_label(s, root), use_prefix) for s in set_dirs]
        print("[+] Processus d'importation vidéo + annotations filtrées terminé.")

if __name__ == "__main__":
    main()