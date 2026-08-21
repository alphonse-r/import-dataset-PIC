"""
UTILISATION
-----------
    python3 import_annot_to_picsellia.py \
        --annotations_dir ./output_visdrone/annotations

DESCRIPTION
-----------
    Importe directement les fichiers JSON Video-COCO générés par
    visdrone_to_video_coco.py vers la version de dataset Picsellia.

PRÉREQUIS
---------
    1. Fichier .env dans le même dossier contenant :
       PICSELLIA_API_TOKEN="votre_token_api"
       PICSELLIA_DATASET_VERSION_ID="votre_id_de_version"

    2. Les vidéos MP4 associées doivent déjà être présentes sur Picsellia.
"""

import os
import glob
import argparse
from dotenv import load_dotenv
from picsellia import Client

def import_annotations(annotations_dir: str):
    load_dotenv()

    api_token = os.getenv("PICSELLIA_API_TOKEN")
    dataset_version_id = os.getenv("PICSELLIA_DATASET_VERSION_ID")

    if not api_token or not dataset_version_id:
        raise ValueError("Variables PICSELLIA_API_TOKEN ou PICSELLIA_DATASET_VERSION_ID manquantes dans le fichier .env")

    client = Client(api_token=api_token)
    dataset_version = client.get_dataset_version_by_id(dataset_version_id)

    json_files = sorted(glob.glob(os.path.join(annotations_dir, "*.json")))
    if not json_files:
        print(f"[AVERTISSEMENT] Aucun fichier JSON trouvé dans : {annotations_dir}")
        return

    print(f"Trouvé {len(json_files)} fichiers JSON Video-COCO. Début de l'importation directe...\n")

    for json_path in json_files:
        filename = os.path.basename(json_path)
        print(f"Importation de {filename}...")
        try:
            res = dataset_version.import_annotations_coco_video_file(
                file_path=json_path,
                fail_on_asset_not_found=True
            )
            print(f"[OK] {filename} -> {res}")
        except Exception as e:
            print(f"[ERREUR] Échec de l'importation pour {filename}: {e}")

    print("\nImportation terminée. Rafraîchissez l'outil Video Studio dans Picsellia.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Importation directe des fichiers JSON Video-COCO vers Picsellia")
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="./output_visdrone_test/annotations",
        help="Chemin vers le dossier contenant les fichiers JSON Video-COCO"
    )

    args = parser.parse_args()
    import_annotations(args.annotations_dir)