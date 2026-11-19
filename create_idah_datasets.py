#!/usr/bin/env python3
"""
Cree la structure IDAH : un projet, et un dataset par set du disque.

POURQUOI :
    L'arborescence source a un dossier par set (A1, A2, B1, B2, C1, C2) et
    la structure cible sur IDAH doit la refleter : un DATASET par set, et non
    un dataset unique contenant les 210 videos. Les noms de datasets sont
    donc lus sur le disque, pas ecrits en dur ici.

    Le projet dedie regle au passage la question du nommage : l'API autorise
    les doublons de noms de datasets, et l'organisation contient deja des
    datasets A1, A2, B1... sans rapport. Isoles dans leur propre projet, nos
    "A1" ne peuvent etre confondus avec eux.

CE QUE L'API ACCEPTE (releve dans le bundle du front, aucune doc exposee) :

    POST /dataset/projects
      attributes : name, description, organization_id
      Reserve au scope 'as_org_owner' cote interface.

    POST /dataset/datasets
      attributes    : name, modality, labeling_configuration, workflow_configuration
      relationships : project -> {"type": "dataset:projects", "id": <uuid>}

    labeling_configuration se pose DES LA CREATION : les couleurs par label
    sont donc bonnes d'emblee, sans repasser derriere avec
    declare_idah_label_config.py. C'est ce meme module qui fournit la config,
    pour que les couleurs restent identiques dans tous les datasets.

USAGE :
    .venv/bin/python create_idah_datasets.py --project TASK_FORCE_GT
    .venv/bin/python create_idah_datasets.py --project TASK_FORCE_GT --apply
    .venv/bin/python create_idah_datasets.py --project TASK_FORCE_GT --apply --create-project
    .venv/bin/python create_idah_datasets.py --project <uuid> --apply --sets A1,A2

--project accepte un uuid ou un nom. Avec un nom absent de l'organisation,
--create-project cree le projet ; sans ce drapeau le script s'arrete.

Le script est idempotent : un dataset dont le nom existe deja dans le projet
est laisse tel quel, jamais recree ni modifie. Il affiche en fin d'execution
la correspondance set -> uuid, a reporter dans le .env.
"""

import sys
import json
import argparse
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("import_video_et_annot_idah.py")
CONFIG_SCRIPT = Path(__file__).with_name("declare_idah_label_config.py")

MODALITY = "idah-video"
ORGANIZATION_ID = 11


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_project(m, base, ref):
    """Retourne (uuid, nom) du projet designe par un uuid ou un nom, sinon None."""
    h = {"Accept": "application/vnd.api+json"}
    res = m.session.get(f"{base}/projects?page[size]=100", headers=h)
    res.raise_for_status()
    for d in res.json().get("data", []):
        if d["id"] == ref or d["attributes"].get("name") == ref:
            return d["id"], d["attributes"].get("name")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Cree un projet IDAH et un dataset par set du disque.")
    parser.add_argument("--project", required=True,
                        help="Uuid ou nom du projet cible.")
    parser.add_argument("--create-project", action="store_true",
                        help="Cree le projet si son nom n'existe pas encore.")
    parser.add_argument("--base-dir", default=None,
                        help="Racine contenant les sets (defaut : IDAH_BASE_DIR).")
    parser.add_argument("--sets", default=None,
                        help="Sous-ensemble de sets, separes par des virgules.")
    parser.add_argument("--apply", action="store_true",
                        help="Cree reellement (sans ce drapeau : simulation).")
    args = parser.parse_args()

    m = load_module(SCRIPT, "idah")
    cfg = load_module(CONFIG_SCRIPT, "idah_cfg")

    root = Path(args.base_dir).expanduser() if args.base_dir else m.BASE_DIR
    root = root.resolve()

    set_dirs = m.discover_sets(root)
    if not set_dirs:
        print(f"[!!] Aucun set trouve sous {root}")
        sys.exit(1)
    wanted = [x.strip() for x in (args.sets or "").split(",") if x.strip()]
    if wanted:
        set_dirs = m.filter_sets(set_dirs, root, wanted)
    labels = [m.set_label(d, root) for d in set_dirs]

    m.login()
    base = m.CTX["base_url"]
    h_get = {"Accept": "application/vnd.api+json"}
    h_post = {"Content-Type": "application/vnd.api+json"}

    print(f"Racine  : {root}")
    print(f"Sets    : {', '.join(labels)}")

    found = find_project(m, base, args.project)
    if found:
        project_id, project_name = found
        print(f"Projet  : {project_name}  ({project_id})")
    elif not args.create_project:
        print(f"\n[!!] Projet introuvable : {args.project!r}")
        print("     Ajouter --create-project pour le creer, ou passer un uuid existant.")
        sys.exit(1)
    elif not args.apply:
        project_id, project_name = None, args.project
        print(f"Projet  : {args.project}  (serait cree)")
    else:
        payload = {"data": {"type": "dataset:projects", "attributes": {
            "name": args.project, "organization_id": ORGANIZATION_ID}}}
        res = m.session.post(f"{base}/projects", json=payload, headers=h_post)
        if not res.ok:
            print(f"\n[!!] Creation du projet refusee ({res.status_code}) : "
                  + " ".join(res.text.split())[:300])
            sys.exit(1)
        project_id = res.json()["data"]["id"]
        project_name = args.project
        print(f"Projet  : {project_name}  ({project_id})  [cree]")

    # Datasets deja presents dans le projet, pour rester idempotent.
    existants = {}
    if project_id:
        res = m.session.get(
            f"{base}/datasets?filter[project_id]={project_id}&page[size]=100",
            headers=h_get)
        res.raise_for_status()
        for d in res.json().get("data", []):
            existants[d["attributes"]["name"]] = d["id"]

    config = cfg.build_config(m.CATEGORY_MAP, MODALITY)
    print(f"Config  : {len(config[f'{MODALITY}:{cfg.SHAPE}']['values'])} classes, "
          f"cle {MODALITY}:{cfg.SHAPE}")
    print()

    mapping, created, skipped, failed = {}, 0, 0, 0
    for label in labels:
        if label in existants:
            mapping[label] = existants[label]
            skipped += 1
            print(f"  {label:<6} existe deja  -> {existants[label]}")
            continue

        if not args.apply:
            print(f"  {label:<6} serait cree  (modalite {MODALITY}, couleurs incluses)")
            continue

        payload = {"data": {
            "type": "dataset:datasets",
            "attributes": {"name": label, "modality": MODALITY,
                           "labeling_configuration": config,
                           "workflow_configuration": {}},
            "relationships": {"project": {"data": {
                "type": "dataset:projects", "id": project_id}}}}}
        res = m.session.post(f"{base}/datasets", json=payload, headers=h_post)
        if not res.ok:
            failed += 1
            print(f"  [ECHEC {res.status_code}] {label} : "
                  + " ".join(res.text.split())[:200])
            continue

        ds_id = res.json()["data"]["id"]
        # Relecture : labeling_configuration est un jsonb libre, un 201 ne
        # garantit pas que la config posee est celle relue.
        after = m.session.get(f"{base}/datasets/{ds_id}", headers=h_get)
        written = after.json()["data"]["attributes"].get("labeling_configuration")
        if written != config:
            failed += 1
            print(f"  [!!] {label} cree ({ds_id}) mais la config relue differe :")
            print("       " + json.dumps(written, ensure_ascii=False)[:200])
            continue

        mapping[label] = ds_id
        created += 1
        print(f"  {label:<6} cree         -> {ds_id}  (config relue conforme)")

    if not args.apply:
        print("\n[simulation] Ajouter --apply pour creer. Aucune modification effectuee.")
        return

    print(f"\nCrees : {created}   Deja presents : {skipped}   Echecs : {failed}")
    if mapping:
        print("\nCorrespondance set -> dataset :")
        for label, ds_id in mapping.items():
            print(f"  {label:<6} {ds_id}")
        print(f"\nA reporter dans le .env :\n  IDAH_PROJECT_ID={project_id}")
        print("\nL'import resout les datasets par leur nom :")
        print('  .venv/bin/python "import_video_et_annot_idah.py" --dataset-per-set')
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
