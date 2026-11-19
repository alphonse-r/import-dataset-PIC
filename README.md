# Scripts du Task Force

| Script | Rôle |
|---|---|
| `visdrone_to_video_coco.py` | Convertit les `folder_name/*.jpg` en vidéos `folder_name.mp4` |
| `import_annot_to_picsellia.py` | Importe les annotations sur Picsellia (SDK) |
| `import_video+annot_idah.py` | Importe les vidéos + annotations sur IDAH |
| `create_idah_datasets.py` | Crée le projet IDAH et un dataset par set |
| `declare_idah_label_config.py` | Déclare les couleurs par label d'un dataset |
| `delete_idah_entries.py` | Supprime des entrées d'un dataset |

Détails de la chaîne IDAH (schémas d'API, pièges) : `GUIDE_IMPORT_IDAH.txt`.

---

# Import IDAH

## 1. Ce qu'il faut modifier

Un seul chemin à changer dans le `.env` : **`IDAH_BASE_DIR`**, la racine qui
contient les sets. Chemin **absolu** (pas de `~`, Python ne l'expanse pas).

```ini
IDAH_BASE_DIR=/Users/macbook-029h/dataset/by_subset
```

Elle doit contenir un dossier par set, chacun avec `videos/` et `annotations/` :

```
by_subset/
├── A1/videos/*.mp4  +  A1/annotations/*.txt
├── A2/  B1/  B2/  C1/  C2/
```

Les autres variables du `.env` ne changent qu'au changement de projet IDAH :

```ini
API_KEY=<clé API>
IDAH_USER_EMAIL=<email>
IDAH_PROJECT_ID=<uuid du projet>        # affiché par l'étape 3
IDAH_DATASET_ID=<uuid>                  # ignoré avec --dataset-per-set
IDAH_DEFAULT_CATEGORY="vehicle"
```

## 2. Installation (une seule fois)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import cv2, requests, dotenv; print('OK', cv2.__version__)"
```

→ `OK 5.0.0`

Le `3.12` est obligatoire : `python3` pointe sur 3.14, sans wheel `opencv-python`.

## 3. Vérifier les données (n'écrit rien)

```bash
.venv/bin/python "import_video+annot_idah.py" --dry-run
```

```
A1  35 vidéo(s)  238 track(s)      B2  35 vidéo(s)  238 track(s)
A2  35 vidéo(s)  242 track(s)      C1  35 vidéo(s)  227 track(s)
B1  35 vidéo(s)  201 track(s)      C2  35 vidéo(s)  244 track(s)

Categories : truck 412, bicycle 335, van 281, tricycle 214, bus 100, awning-tricycle 48
LIGNES SOURCES  retenues : 7889   ecartees : 0
```

**Ne pas continuer si** un set manque, si un bloc `IDS DE CATEGORIE INCONNUS`
apparaît, ou si `vehicle` figure dans les catégories.

## 4. Créer le projet et les datasets

```bash
.venv/bin/python create_idah_datasets.py --project TASK_FORCE_GT --apply --create-project
```

```
A1     cree  -> 01a05dd9-d4ab-...  (config relue conforme)
...
Crees : 6   Deja presents : 0   Echecs : 0
A reporter dans le .env :
  IDAH_PROJECT_ID=01a05dd8-7d73-...
```

Reporter cet `IDAH_PROJECT_ID` dans le `.env`. Relançable sans risque : un
dataset déjà présent est laissé tel quel.

## 5. Importer

```bash
.venv/bin/python "import_video+annot_idah.py" --dataset-per-set 2>&1 | tee import_idah.log
```

Chaque set part dans le dataset portant son nom. Compter ~1 h pour 210 vidéos.

```
SET             VIDÉOS     OK   SKIP   ÉCHEC   TRACKS   TR.KO
A1                  35     35      0       0      238       0
...
TOTAL              210    210      0       0     1390       0
```

`SKIP` = vidéo sans annotation, `ÉCHEC` = erreur d'import, `TR.KO` = track
refusée. Tout doit être à 0. Une erreur sur une vidéo n'arrête pas le reste :
relancer les sets concernés avec `--sets B2,C1`.

## 6. Vérifier le résultat

```bash
.venv/bin/python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('idah', 'import_video+annot_idah.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.login()
b = m.CTX['base_url']; h = {'Accept': 'application/vnd.api+json'}
r = m.session.get(f\"{b}/datasets?filter[project_id]={m.CTX['proj_id']}&page[size]=100\", headers=h)
for d in sorted(r.json()['data'], key=lambda x: x['attributes']['name']):
    m.CTX['ds_id'] = d['id']; m.ENTRY_CACHE.clear(); m.ENTRY_CACHE_LOADED = False
    m.load_entry_cache()
    n = sum(len(m.session.get(f'{b}/annotations?filter[entry_id]={e}&page[size]=500').json().get('data', []))
            for e in m.ENTRY_CACHE.values())
    print(f\"  {d['attributes']['name']:<4} {len(m.ENTRY_CACHE):>4} entries  {n:>5} tracks\")
    "
```

Les comptes doivent être identiques au tableau de l'étape 5. Le corps Python est
collé à la marge gauche : ne pas l'indenter en copiant.

Enfin, ouvrir une vidéo dans l'outil d'annotation : deux tracks d'une même
classe doivent avoir la même couleur.

---

# Relancer un set déjà importé

Les annotations ne sont pas dédupliquées : réimporter sans vider crée des
doublons.

```bash
.venv/bin/python delete_idah_entries.py --dataset <uuid> --match '.' --apply --force --reset-status
.venv/bin/python "import_video+annot_idah.py" --dataset-per-set --sets B2
```
