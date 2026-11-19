#!/usr/bin/env python3
"""
Declare la configuration d'affichage des labels (couleurs) du dataset IDAH.

LE PROBLEME :
    Dans l'outil d'annotation, un meme label apparait en plusieurs couleurs.
    Ce n'est ni le GT ni l'import : aucune couleur n'existe dans la donnee.
    import_video_et_annot_idah.py n'envoie que category + attributes.track_id, et
    l'objet annotation en base n'a aucun champ couleur. Faute de mapping
    classe -> couleur, le front colore PAR ANNOTATION -- donc par track, chez
    nous -- et six tracks "bicycle" donnent six couleurs.

    L'attribut "labels" (declare_idah_labels.py) ne peut pas corriger cela :
    c'est une colonne Postgres text[], donc des noms nus, sans couleur. Le
    mapping vit dans "labeling_configuration". A noter : tous les datasets
    configures a la main via l'interface ont "labels": [] -- c'est bien
    labeling_configuration qui porte la liste de classes utilisee a l'ecran.

SCHEMA (releve sur des datasets configures via l'interface, aucune doc) :
    La colonne est un jsonb LIBRE : elle accepte n'importe quelle forme, y
    compris une cle inventee, sans erreur. Un PATCH qui repond 200 ne prouve
    donc rien du tout ; seule la forme ci-dessous est lue par le front.

    {
      "<modality>:<shape>": {            # ex: "idah-video:bounding-box"
        "order": 0,
        "values": [
          {
            "id":         "car",        # DOIT egaler annotation.category
            "label":      "Car",        # nom affiche
            "color":      "#00BBD6",    # couleur de la boite
            "text_color": "#000000",    # couleur du texte par-dessus
            "order":      0,
            "parent":     null,         # hierarchie : id = "vehicle/car"
            "children":   [],
            "expanded":   true
          }
        ],
        "properties": []                # champs de saisie par annotation
      }
    }

    "id" est la cle de jointure avec la donnee : c'est lui qui doit
    correspondre EXACTEMENT aux valeurs de annotation.category, donc aux noms
    de CATEGORY_MAP. "label" n'est qu'un libelle d'affichage.

PALETTE :
    Les couleurs sont celles du selecteur de l'interface IDAH, avec leur
    text_color d'origine. Elles sont fixees ici en dur, une par classe
    VisDrone : c'est ce qui garantit qu'un label garde la meme couleur d'un
    dataset a l'autre. Toute modification se fait ICI et nulle part ailleurs.

USAGE :
    .venv/bin/python declare_idah_label_config.py             # simulation
    .venv/bin/python declare_idah_label_config.py --apply     # ecrit
    .venv/bin/python declare_idah_label_config.py --apply --dataset <uuid>[,<uuid>]
    .venv/bin/python declare_idah_label_config.py --apply --force   # ecrase une config existante
    .venv/bin/python declare_idah_label_config.py --apply --reset   # remet {}
"""

import sys
import json
import argparse
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("import_video_et_annot_idah.py")

# Forme d'annotation produite par l'import : des boites englobantes.
SHAPE = "bounding-box"

# Une couleur par classe VisDrone, prise dans la palette de l'interface IDAH.
# L'ordre suit les ids VisDrone 1..11 de CATEGORY_MAP.
PALETTE = {
    "pedestrian":      ("#F6402B", "#FFFFFF"),  # Red
    "people":          ("#EB1461", "#FFFFFF"),  # Pink
    "bicycle":         ("#CCDD1D", "#000000"),  # Lime
    "car":             ("#3C4DB7", "#FFFFFF"),  # Blue
    "van":             ("#00BBD6", "#000000"),  # Cyan
    "truck":           ("#6A34BE", "#FFFFFF"),  # Deep Purple
    "tricycle":        ("#FFC100", "#000000"),  # Amber
    "awning-tricycle": ("#FF9700", "#000000"),  # Orange
    "bus":             ("#46AF4A", "#FFFFFF"),  # Green
    "motor":           ("#009687", "#000000"),  # Teal
    "others":          ("#828282", "#FFFFFF"),  # Gray
}


def load_importer():
    spec = importlib.util.spec_from_file_location("idah", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_config(category_map, modality):
    """Construit labeling_configuration pour une modalite donnee."""
    values = []
    for order, cat_id in enumerate(sorted(category_map)):
        name = category_map[cat_id]
        if name not in PALETTE:
            print(f"[!!] Pas de couleur pour '{name}' : completer PALETTE.")
            sys.exit(1)
        color, text_color = PALETTE[name]
        values.append({
            "id": name,                 # doit egaler annotation.category
            "label": name.capitalize(),
            "color": color,
            "text_color": text_color,
            "order": order,
            "parent": None,
            "children": [],
            "expanded": True,
        })
    return {f"{modality}:{SHAPE}": {"order": 0, "values": values, "properties": []}}


def describe(config):
    for key, block in config.items():
        print(f"  {key}")
        for v in block.get("values", []):
            print(f"    {v['id']:<18} {v['color']}  (texte {v['text_color']})")


def main():
    parser = argparse.ArgumentParser(
        description="Declare les couleurs par label du dataset IDAH.")
    parser.add_argument("--apply", action="store_true",
                        help="Ecrit reellement (sans ce drapeau : simulation).")
    parser.add_argument("--dataset", default=None,
                        help="Uuid(s) separes par des virgules (defaut : IDAH_DATASET_ID).")
    parser.add_argument("--force", action="store_true",
                        help="Ecrase une labeling_configuration existante differente.")
    parser.add_argument("--reset", action="store_true",
                        help="Remet labeling_configuration a {}.")
    args = parser.parse_args()

    m = load_importer()
    m.login()
    base = m.CTX["base_url"]
    h_get = {"Accept": "application/vnd.api+json"}
    h_patch = {"Content-Type": "application/vnd.api+json"}

    targets = ([x.strip() for x in args.dataset.split(",") if x.strip()]
               if args.dataset else [m.CTX["ds_id"]])

    failures = 0
    for ds in targets:
        res = m.session.get(f"{base}/datasets/{ds}", headers=h_get)
        if not res.ok:
            print(f"\n[!!] Dataset {ds} illisible (code {res.status_code}).")
            failures += 1
            continue
        attrs = res.json()["data"]["attributes"]
        modality = attrs.get("modality")
        current = attrs.get("labeling_configuration") or {}

        print(f"\nDataset : {attrs.get('name')}  ({ds})")
        print(f"  modalite  : {modality}")
        print(f"  entrees   : {attrs.get('entries_total_count')}")

        wanted = {} if args.reset else build_config(m.CATEGORY_MAP, modality)

        if current:
            print(f"  config actuelle : {len(current)} bloc(s) -> {', '.join(current)}")
        else:
            print("  config actuelle : {} (aucune couleur : le front colore par track)")

        if current == wanted:
            print("  Rien a faire : la configuration est deja celle attendue.")
            continue

        # Une config posee a la main via l'interface ne doit pas etre ecrasee
        # par accident : elle peut contenir des classes ou des proprietes que
        # ce script ne connait pas.
        if current and not args.force and not args.reset:
            print("  [!] Une configuration existe deja et differe de celle voulue.")
            print("      Relancer avec --force pour l'ecraser. Contenu actuel :")
            print("      " + json.dumps(current, ensure_ascii=False)[:300])
            failures += 1
            continue

        if not args.reset:
            print("  configuration voulue :")
            describe(wanted)

        if not args.apply:
            print("  [simulation] Ajouter --apply pour ecrire.")
            continue

        payload = {"data": {"type": "dataset:datasets", "id": ds,
                            "attributes": {"labeling_configuration": wanted}}}
        res = m.session.patch(f"{base}/datasets/{ds}", json=payload, headers=h_patch)
        print(f"  PATCH -> {res.status_code}")
        if not res.ok:
            print("  " + " ".join(res.text.split())[:300])
            failures += 1
            continue

        # Relecture obligatoire : la colonne est un jsonb libre, un 200 ne
        # garantit pas que ce qui a ete envoye est ce qui est en base.
        after = m.session.get(f"{base}/datasets/{ds}", headers=h_get)
        written = after.json()["data"]["attributes"].get("labeling_configuration") or {}
        if written != wanted:
            print("  [!!] Ce qui est en base ne correspond pas a ce qui a ete envoye :")
            print("       " + json.dumps(written, ensure_ascii=False)[:300])
            failures += 1
            continue
        print("  OK : relecture conforme.")

    if failures:
        sys.exit(1)
    if args.apply:
        print("\nRecharger l'onglet de l'outil d'annotation pour voir les couleurs.")


if __name__ == "__main__":
    main()
