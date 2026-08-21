"""
UTILISATION
-----------
    python3 visdrone_to_video_coco.py \
        --root /chemin/vers/VisDrone_dataset \
        --out ./output_visdrone \
        --fps 20

DESCRIPTION
-----------
    Convertit automatiquement toutes les séquences VisDrone d'un dossier
    (images JPG/PNG + annotations TXT) en fichiers vidéos MP4 et génère
    les fichiers JSON au format Video-COCO correspondants.

STRUCTURE DU DOSSIER D'ENTRÉE ATTENDUE (--root)
-----------------------------------------------
    <dossier_racine>/
    ├── annotations/
    │   ├── uav0000370_00001_v.txt
    │   ├── uav0000370_00002_v.txt
    │   └── ...
    └── sequences/
        ├── uav0000370_00001_v/
        │   ├── 0000001.jpg
        │   ├── 0000002.jpg
        │   └── ...
        └── uav0000370_00002_v/
            ├── 0000001.jpg
            ├── 0000002.jpg
            └── ...
"""

import os
import csv
import json
import argparse
import subprocess
from pathlib import Path
from PIL import Image

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

def build_mp4(seq_dir: Path, out_mp4: Path, fps: int):
    """Assemble la suite d'images JPG/PNG en un fichier MP4 avec ffmpeg."""
    frames = sorted(list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png")))
    if not frames:
        return 0, 0, 0

    with Image.open(frames[0]) as img:
        width, height = img.size

    pattern = str(seq_dir / f"*{frames[0].suffix}")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_mp4)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return len(frames), width, height

def parse_visdrone_txt(ann_path: Path):
    """Lit le fichier d'annotation VisDrone .txt et filtre les boîtes invalides."""
    rows = []
    if not ann_path.exists():
        return rows

    with open(ann_path, "r", newline="") as f:
        reader = csv.reader(f)
        for line in reader:
            line = [x.strip() for x in line if x.strip()]
            if len(line) < 8:
                continue

            frame_idx = int(line[0])
            track_id = int(line[1])
            left = float(line[2])
            top = float(line[3])
            w = float(line[4])
            h = float(line[5])
            score = int(line[6])
            category = int(line[7])

            # Exclusion des régions ignorées (catégorie 0), score 0 et boîtes invalides
            if category == 0 or score == 0 or w <= 0 or h <= 0:
                continue

            rows.append({
                "frame_index": frame_idx,
                "track_id": track_id,
                "bbox": [round(left, 2), round(top, 2), round(w, 2), round(h, 2)],
                "category_id": category
            })
    return rows

def convert_sequence(seq_dir: Path, ann_path: Path, out_dir: Path, fps: int):
    """Traite une séquence unique : génère la vidéo MP4 et le JSON Video-COCO."""
    seq_name = seq_dir.name
    mp4_path = out_dir / "clips" / f"{seq_name}.mp4"
    json_path = out_dir / "annotations" / f"{seq_name}_video_coco.json"

    num_frames, width, height = build_mp4(seq_dir, mp4_path, fps)
    if num_frames == 0:
        return

    frames = sorted(list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png")))
    rows = parse_visdrone_txt(ann_path)

    # 1. Structure du tableau 'images'
    coco_images = [
        {
            "id": i + 1,
            "file_name": frames[i].name,
            "width": width,
            "height": height,
            "frame_index": i + 1,
            "sequence": seq_name,
            "video_id": 1,
            "frame_id": i
        }
        for i in range(len(frames))
    ]

    # 2. Structure du tableau 'annotations' (avec video_id requis par Picsellia)
    coco_annotations = []
    for ann_id, r in enumerate(rows, start=1):
        f_idx = r["frame_index"] - 1
        coco_annotations.append({
            "id": ann_id,
            "image_id": r["frame_index"],
            "video_id": 1,
            "category_id": r["category_id"],
            "bbox": r["bbox"],
            "area": round(r["bbox"][2] * r["bbox"][3], 2),
            "iscrowd": 0,
            "track_id": r["track_id"],
            "attributes": {
                "track_id": r["track_id"],
                "frame_id": f_idx,
                "keyframe": True
            }
        })

    # 3. Structure globale du JSON Video-COCO
    used_categories = sorted({r["category_id"] for r in rows})
    coco_categories = [
        {"id": cid, "name": CATEGORY_MAP.get(cid, f"unknown-{cid}")}
        for cid in used_categories
    ]

    coco_data = {
        "info": {"description": f"VisDrone sequence {seq_name} converted to COCO"},
        "videos": [{"id": 1, "name": f"{seq_name}.mp4", "file_name": f"{seq_name}.mp4"}],
        "images": coco_images,
        "categories": coco_categories,
        "annotations": coco_annotations
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"[OK] {seq_name} : {num_frames} frames, {len(coco_annotations)} annotations -> {json_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Conversion des séquences VisDrone en MP4 et JSON Video-COCO")
    parser.add_argument("--root", required=True, help="Chemin vers le dossier racine VisDrone (contenant 'sequences/' et 'annotations/')")
    parser.add_argument("--out", required=True, help="Dossier de destination")
    parser.add_argument("--fps", type=int, default=20, help="Images par seconde pour l'encodage MP4 (défaut: 20)")
    args = parser.parse_args()

    root = Path(args.root)
    seq_root = root / "sequences"
    ann_root = root / "annotations"
    out_root = Path(args.out)

    if not seq_root.exists():
        raise FileNotFoundError(f"Dossier introuvable : {seq_root}")

    seq_folders = sorted([p for p in seq_root.iterdir() if p.is_dir()])
    print(f"Traitement de {len(seq_folders)} séquences dans {seq_root}...\n")

    for seq_dir in seq_folders:
        ann_path = ann_root / f"{seq_dir.name}.txt"
        convert_sequence(seq_dir, ann_path, out_root, args.fps)

if __name__ == "__main__":
    main()