#!/usr/bin/env python3
"""
Supprime des entrees d'un dataset IDAH, selectionnees par expression reguliere.

POURQUOI :
    Les entrees sont retrouvees PAR NOM par import_video+annot_idah.py. Un
    import fait sans prefixe de set, puis rejoue avec prefixe, ne reutilise
    donc pas les anciennes entrees : il en cree de nouvelles et le dataset
    garde les deux jeux. Ce script sert a retirer le jeu obsolete.

ATTENTION : la suppression est definitive cote plateforme et emporte les
annotations rattachees. Le script est en simulation par defaut et n'ecrit
rien sans --apply.

USAGE :
    .venv/bin/python delete_idah_entries.py --match '^uav'            # simulation
    .venv/bin/python delete_idah_entries.py --match '^uav' --apply    # supprime
    .venv/bin/python delete_idah_entries.py --match '^uav' --dataset <uuid>
    .venv/bin/python delete_idah_entries.py --match '^uav' --apply --reset-status

Une entree ouverte dans l'outil d'annotation passe en statut in_progress et
l'API refuse alors sa suppression (403 "Unable to delete in progress or
completed entry"). --reset-status la remet a 'pending' puis reessaie.

--match est OBLIGATOIRE : il n'y a pas de mode "tout supprimer" implicite.
Le script refuse une expression qui selectionne la totalite des entrees, sauf
--force, pour rendre impossible un vidage accidentel du dataset.
"""

import re
import sys
import argparse
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("import_video+annot_idah.py")


def load_importer():
    spec = importlib.util.spec_from_file_location("idah", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Supprime des entrees d'un dataset IDAH.")
    parser.add_argument("--match", required=True,
                        help="Expression reguliere appliquee au NOM de l'entree.")
    parser.add_argument("--dataset", default=None,
                        help="Uuid du dataset (defaut : IDAH_DATASET_ID du .env).")
    parser.add_argument("--apply", action="store_true",
                        help="Supprime reellement (sans ce drapeau : simulation).")
    parser.add_argument("--force", action="store_true",
                        help="Autorise une selection portant sur toutes les entrees.")
    parser.add_argument("--reset-status", action="store_true",
                        help="Remet a 'pending' une entree refusee car in_progress "
                             "ou completed, puis reessaie la suppression.")
    args = parser.parse_args()

    try:
        pattern = re.compile(args.match)
    except re.error as err:
        print(f"[!!] Expression reguliere invalide : {err}")
        sys.exit(1)

    m = load_importer()
    if args.dataset:
        m.CTX["ds_id"] = args.dataset
    m.login()
    m.load_entry_cache()

    base = m.CTX["base_url"]
    total = len(m.ENTRY_CACHE)
    selected = sorted(n for n in m.ENTRY_CACHE if pattern.search(n))
    kept = total - len(selected)

    print(f"\nDataset  : {m.CTX['ds_id']}")
    print(f"  entrees          : {total}")
    print(f"  selectionnees    : {len(selected)}  (motif {args.match!r})")
    print(f"  conservees       : {kept}")

    if not selected:
        print("\nAucune entree ne correspond. Rien a faire.")
        return

    print("\nEntrees selectionnees :")
    for name in selected:
        print(f"  {name}   ({m.ENTRY_CACHE[name]})")

    if kept == 0 and not args.force:
        print("\n[!] Ce motif selectionne TOUTES les entrees du dataset.")
        print("    Ajouter --force si c'est bien l'intention.")
        sys.exit(1)

    if not args.apply:
        print(f"\n[simulation] {len(selected)} entree(s) seraient supprimees, "
              "annotations comprises.")
        print("Ajouter --apply pour executer. Aucune modification effectuee.")
        return

    print()
    ok = ko = 0
    h_patch = {"Content-Type": "application/vnd.api+json"}
    for name in selected:
        entry_id = m.ENTRY_CACHE[name]
        res = m.session.delete(f"{base}/entries/{entry_id}")

        # L'API refuse (403) de supprimer une entree ouverte dans l'outil
        # d'annotation : "Unable to delete in progress or completed entry".
        # La remettre a 'pending' leve le blocage.
        if res.status_code == 403 and args.reset_status:
            payload = {"data": {"type": "dataset:entries", "id": entry_id,
                                "attributes": {"status": "pending"}}}
            reset = m.session.patch(f"{base}/entries/{entry_id}",
                                    json=payload, headers=h_patch)
            if reset.ok:
                print(f"  statut remis a pending : {name}")
                res = m.session.delete(f"{base}/entries/{entry_id}")

        if res.ok or res.status_code == 404:
            ok += 1
            print(f"  supprime : {name}")
        else:
            ko += 1
            print(f"  [ECHEC {res.status_code}] {name} : "
                  + " ".join(res.text.split())[:160])

    # Relecture : on recharge la liste depuis l'API plutot que de se fier aux
    # codes retour, la pagination du cache pouvant masquer un reste.
    m.ENTRY_CACHE.clear()
    m.ENTRY_CACHE_LOADED = False
    m.load_entry_cache()
    restant = sorted(n for n in m.ENTRY_CACHE if pattern.search(n))

    print(f"\nSupprimees : {ok}   Echecs : {ko}")
    print(f"Entrees restantes dans le dataset : {len(m.ENTRY_CACHE)}")
    if restant:
        print(f"[!!] {len(restant)} entree(s) correspondent encore au motif : "
              f"{restant[:5]}")
        sys.exit(1)
    print("OK : plus aucune entree ne correspond au motif.")


if __name__ == "__main__":
    main()
