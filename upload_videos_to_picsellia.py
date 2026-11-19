#!/usr/bin/env python3
"""
UTILISATION
-----------
    python3 upload_videos_to_picsellia.py --clips_dir ./output_visdrone/clips           # simulation
    python3 upload_videos_to_picsellia.py --clips_dir ./output_visdrone/clips --apply   # envoi

DESCRIPTION
-----------
    Met les vidéos MP4 produites par visdrone_to_video_coco.py dans la version
    de dataset Picsellia, pour que import_annot_to_picsellia.py puisse ensuite
    y accrocher les annotations.

    Picsellia range les fichiers sur deux étages : le Datalake stocke les
    fichiers (des "data"), et une version de dataset en référence une
    sélection (des "assets"). Les annotations s'accrochent aux assets, et
    import_annot_to_picsellia.py les retrouve par NOM DE FICHIER. Le script
    s'occupe donc des deux étages :
        1. il téléverse dans le Datalake les vidéos qui n'y sont pas encore ;
        2. il les ajoute comme assets à la version de dataset.

    On peut le relancer sans risque : une vidéo déjà présente dans la version
    est ignorée, et une vidéo déjà présente dans le Datalake est réutilisée au
    lieu d'être téléversée une seconde fois.

    Sans --apply, rien n'est envoyé : le script affiche seulement ce qu'il
    ferait.

PRÉREQUIS
---------
    Fichier .env contenant :
        PICSELLIA_API_TOKEN=...
        PICSELLIA_DATASET_VERSION_ID=...
"""

import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from picsellia import Client
from picsellia.exceptions import NoDataError, NothingDoneError
from picsellia.sdk.data import Data
from picsellia.services.error_manager import ErrorManager

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

# Les recherches par nom de fichier se font par paquets, pour garder des
# requêtes de taille raisonnable quand il y a des centaines de vidéos.
CHUNK_SIZE = 50


def find_by_filename(lister, filenames):
    """Retourne {nom de fichier: [objets trouvés]}.

    `lister` est dataset_version.list_assets ou datalake.list_data. Les deux
    lèvent NoDataError quand rien ne correspond, au lieu de renvoyer une liste
    vide : on traite ce cas comme « rien trouvé ».
    """
    found = defaultdict(list)
    for start in range(0, len(filenames), CHUNK_SIZE):
        try:
            result = lister(filenames=filenames[start:start + CHUNK_SIZE])
        except NoDataError:
            continue
        for obj in result.items:
            found[obj.filename].append(obj)
    return found


def list_videos(clips_dir: Path) -> dict:
    if not clips_dir.is_dir():
        return {}
    return {p.name: p for p in sorted(clips_dir.iterdir())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS}


def main():
    parser = argparse.ArgumentParser(
        description="Téléverse les vidéos dans la version de dataset Picsellia."
    )
    parser.add_argument("--clips_dir", default="./output_visdrone/clips",
                        help="Dossier des vidéos (défaut : ./output_visdrone/clips)")
    parser.add_argument("--datalake", default=None,
                        help="Nom du Datalake à utiliser (défaut : celui de l'organisation)")
    parser.add_argument("--apply", action="store_true",
                        help="Envoie réellement (sans ce drapeau : simulation)")
    args = parser.parse_args()

    load_dotenv()
    api_token = os.getenv("PICSELLIA_API_TOKEN")
    dataset_version_id = os.getenv("PICSELLIA_DATASET_VERSION_ID")
    if not api_token or not dataset_version_id:
        sys.exit("[!!] PICSELLIA_API_TOKEN ou PICSELLIA_DATASET_VERSION_ID manquant dans le .env")

    clips_dir = Path(args.clips_dir).expanduser()
    videos = list_videos(clips_dir)
    if not videos:
        sys.exit(f"[!!] Aucune vidéo trouvée dans {clips_dir}")
    names = list(videos)

    client = Client(api_token=api_token)
    dataset_version = client.get_dataset_version_by_id(dataset_version_id)
    datalake = client.get_datalake(name=args.datalake)

    print(f"[+] {len(names)} vidéo(s) dans {clips_dir}")
    print(f"[+] Version de dataset : {dataset_version.name} / {dataset_version.version}")
    print(f"[+] Datalake : {datalake.name}\n")

    # 1. Ce qui est déjà dans la version : rien à faire.
    in_version = find_by_filename(dataset_version.list_assets, names)
    todo = [n for n in names if n not in in_version]

    # 2. Ce qui est déjà dans le Datalake : on l'ajoute à la version sans le
    #    téléverser une seconde fois.
    in_datalake = find_by_filename(datalake.list_data, todo)
    to_upload = [n for n in todo if n not in in_datalake]

    print(f"  déjà dans la version   : {len(in_version)}")
    print(f"  déjà dans le Datalake  : {len(in_datalake)}  (à ajouter à la version)")
    print(f"  à téléverser           : {len(to_upload)}")

    # Picsellia retrouve la vidéo d'un JSON par son nom : avec deux assets du
    # même nom, import_annot_to_picsellia.py échoue sur ce fichier.
    duplicated = sorted(n for n, assets in in_version.items() if len(assets) > 1)
    if duplicated:
        print(f"\n[!] {len(duplicated)} vidéo(s) présente(s) plusieurs fois dans la version :")
        for n in duplicated[:10]:
            print(f"    - {n} ({len(in_version[n])} fois)")
        print("    L'import de leurs annotations échouera. Supprimer les doublons dans Picsellia.")

    if not todo:
        print("\n[OK] Toutes les vidéos sont déjà dans la version. Rien à faire.")
        return

    if not args.apply:
        print("\n[simulation] Ajouter --apply pour envoyer. Aucune modification effectuée.")
        return

    # Un même nom peut exister plusieurs fois dans le Datalake : on n'en prend
    # qu'un, pour ne pas créer deux assets du même nom dans la version.
    data_to_add = [in_datalake[n][0] for n in todo if n in in_datalake]

    upload_failed = []
    if to_upload:
        print(f"\n[+] Téléversement de {len(to_upload)} vidéo(s) dans le Datalake...")
        errors = ErrorManager()
        try:
            uploaded = datalake.upload_data([str(videos[n]) for n in to_upload],
                                            error_manager=errors)
            data_to_add += [uploaded] if isinstance(uploaded, Data) else list(uploaded.items)
        except NothingDoneError:
            pass  # tout a échoué : le détail est dans `errors`
        upload_failed = sorted(os.path.basename(str(getattr(e, "path", e)))
                               for e in errors.errors)

    if data_to_add:
        print(f"\n[+] Ajout de {len(data_to_add)} vidéo(s) à la version de dataset...")
        dataset_version.add_data(data_to_add)

    # Contrôle final : on relit la version plutôt que de se fier aux retours.
    final = find_by_filename(dataset_version.list_assets, names)
    missing = [n for n in names if n not in final]

    print(f"\n[+] {len(names) - len(missing)}/{len(names)} vidéo(s) présentes dans la version.")
    if upload_failed:
        print(f"[!] Échec du téléversement pour {len(upload_failed)} vidéo(s) : "
              f"{', '.join(upload_failed[:10])}")
    if missing:
        print(f"[!!] Absentes de la version : {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))
        print("     Relancer le script : il ne renverra que celles-ci.")
        sys.exit(1)

    print("[OK] Tu peux maintenant lancer import_annot_to_picsellia.py.")


if __name__ == "__main__":
    main()
