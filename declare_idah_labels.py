#!/usr/bin/env python3
"""
Declare la liste des classes (labels) sur le dataset IDAH.

POURQUOI :
    Un dataset cree sans labels a "labels": [] cote plateforme. L'API accepte
    alors n'importe quelle chaine dans annotation.category sans validation :
    l'import reussit, la donnee est correcte, mais l'outil d'annotation n'a
    aucune liste de classes a laquelle rattacher un nom ou une couleur. Toutes
    les boites se ressemblent a l'ecran. Declarer les labels corrige cela.

SCHEMA ATTENDU PAR L'API (etabli par sondage, aucune doc exposee) :
    L'attribut "labels" est un tableau de CHAINES, stocke dans une colonne
    Postgres text[]. Rien d'autre n'est accepte :
      - une chaine simple      -> 422 "must be an array"
      - une liste d'objets     -> 200 MAIS ecrit ["false", "false"] en base
        ({"name": ..., "color": ...} est evalue comme un booleen Ruby)
      - {"id": ..., ...}       -> 500 PG::InvalidTextRepresentation
    Il n'y a donc pas de couleur ni d'id par label : juste des noms.
    La liste d'objets est le piege : elle repond 200 et corrompt en silence.

USAGE :
    .venv/bin/python declare_idah_labels.py            # affiche le diff, ne change rien
    .venv/bin/python declare_idah_labels.py --apply    # ecrit les labels
    .venv/bin/python declare_idah_labels.py --apply --labels a,b,c   # liste explicite

Par defaut la liste declaree est CATEGORY_MAP de import_video_et_annot_idah.py,
dans l'ordre des ids VisDrone 1..11. L'id 0 ("ignored regions") en est exclu :
ce n'est pas un objet, et le script d'import le filtre desormais.
"""

import sys
import json
import argparse
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("import_video_et_annot_idah.py")


def load_importer():
    spec = importlib.util.spec_from_file_location("idah", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Declare les labels du dataset IDAH.")
    parser.add_argument("--apply", action="store_true",
                        help="Ecrit reellement les labels (sans ce drapeau : simulation).")
    parser.add_argument("--labels", default=None,
                        help="Liste explicite separee par des virgules (defaut : CATEGORY_MAP).")
    args = parser.parse_args()

    m = load_importer()
    m.login()
    base, ds = m.CTX["base_url"], m.CTX["ds_id"]
    h_get = {"Accept": "application/vnd.api+json"}
    h_patch = {"Content-Type": "application/vnd.api+json"}

    if args.labels:
        wanted = [x.strip() for x in args.labels.split(",") if x.strip()]
    else:
        wanted = [m.CATEGORY_MAP[i] for i in sorted(m.CATEGORY_MAP)]

    res = m.session.get(f"{base}/datasets/{ds}", headers=h_get)
    res.raise_for_status()
    attrs = res.json()["data"]["attributes"]
    current = attrs.get("labels") or []

    print(f"\nDataset : {attrs.get('name')}  ({ds})")
    print(f"  entrees        : {attrs.get('entries_total_count')}")
    print(f"  labels actuels : {json.dumps(current, ensure_ascii=False)}")
    print(f"  labels voulus  : {json.dumps(wanted, ensure_ascii=False)}")

    if current == wanted:
        print("\nRien a faire : les labels sont deja ceux attendus.")
        return

    if not args.apply:
        print("\n[simulation] Ajouter --apply pour ecrire. Aucune modification effectuee.")
        return

    payload = {"data": {"type": "dataset:datasets", "id": ds,
                        "attributes": {"labels": wanted}}}
    res = m.session.patch(f"{base}/datasets/{ds}", json=payload, headers=h_patch)
    print(f"\nPATCH -> {res.status_code}")
    if not res.ok:
        print(res.text[:800])
        sys.exit(1)

    # Relecture : le 200 ne suffit pas, une forme invalide passerait en silence.
    after = m.session.get(f"{base}/datasets/{ds}", headers=h_get).json()["data"]["attributes"]
    written = after.get("labels") or []
    print(f"labels en base : {json.dumps(written, ensure_ascii=False)}")

    if written != wanted:
        print("\n[!!] Ce qui est en base ne correspond pas a ce qui a ete envoye.")
        sys.exit(1)
    print("\nOK : labels declares.")


if __name__ == "__main__":
    main()
