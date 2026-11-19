# import-dataset-PIC

Des scripts Python pour envoyer des vidéos annotées (format VisDrone / MOT) sur
des plateformes d'annotation. **PIC** veut dire **P**icsellia, **I**DAH et
**C**VAT, les trois plateformes visées par le projet.

Ils ont été écrits pour la Task Force, afin de mettre en ligne la vérité
terrain (GT) : des vidéos de drone où chaque objet suivi a sa boîte englobante,
image par image.

Aujourd'hui, le repo couvre IDAH, où se fait l'essentiel du travail, et
Picsellia, dont la partie est plus ancienne et plus simple. Il n'y a encore
rien pour CVAT.

## Ce qu'il y a dans le repo

**Pour IDAH**

- `import_video_et_annot_idah.py` : le script principal. Il parcourt tes
  dossiers, envoie chaque vidéo, crée l'entrée correspondante sur IDAH et y
  attache toutes les tracks.
- `create_idah_datasets.py` : prépare le terrain en créant le projet et un
  dataset par set (A1, A2…), avec les couleurs des labels déjà réglées.
- `declare_idah_label_config.py` : fixe une couleur par classe sur un dataset
  existant, pour que tous les « bicycle » aient la même couleur à l'écran.
- `declare_idah_labels.py` : déclare la liste des classes d'un dataset. Il
  sert peu aujourd'hui, car c'est la configuration du script précédent que
  l'interface lit.
- `delete_idah_entries.py` : supprime des entrées d'un dataset, par exemple
  pour vider un set avant de le réimporter.

**Variante « 4 classes »** (van, truck, awning-tricycle, bus), utilisée pour
une campagne dont les sets s'appellent `J1_G1`, `J1_G2`…

- `importETcreationDataset.py` : même principe que l'import principal, mais il
  ne garde que ces 4 classes et crée à la volée les datasets qui manquent.
- `import_annotations_seules.py` : renvoie uniquement les annotations, sur des
  entrées qui existent déjà.
- `import_video_et_annotations-draft.py` : brouillon, il ne fait rien en
  l'état.

**Pour Picsellia**

- `visdrone_to_video_coco.py` : transforme des séquences d'images VisDrone en
  vidéos MP4 et en fichiers JSON Video-COCO.
- `import_annot_to_picsellia.py` : envoie ces JSON sur une version de dataset
  Picsellia.

Enfin, `GUIDE_IMPORT_IDAH.txt` est le carnet de bord du projet. Il rassemble
tout ce qu'on a appris de l'API IDAH, qui n'a aucune documentation, ainsi que
les bugs rencontrés et leur correction. C'est le fichier à ouvrir quand
quelque chose ne se passe pas comme prévu.

## Installation

Il faut Python 3.12, pas 3.14 : quand le projet a été monté, `opencv-python`
n'avait pas encore de version pour 3.14. OpenCV n'est pas un détail. C'est lui
qui lit la résolution des vidéos. Sans lui, le script suppose du 1920×1080 :
toutes les boîtes sont alors décalées, sans le moindre message d'erreur.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import cv2, requests, dotenv; print('OK', cv2.__version__)"
```

Toutes les commandes de ce README supposent que le venv est activé.

Pour la partie Picsellia, il faut aussi `ffmpeg` (`brew install ffmpeg` sur
Mac).

### Le fichier `.env`

Copie le modèle, puis remplis-le :

```bash
cp .env-example .env
```

| Variable | À quoi elle sert |
|---|---|
| `API_KEY` | Clé API IDAH |
| `IDAH_USER_EMAIL` | Email associé aux annotations créées |
| `IDAH_PROJECT_ID` | UUID du projet IDAH (affiché par `create_idah_datasets.py`) |
| `IDAH_DATASET_ID` | Dataset cible quand tout part dans un seul dataset. Sert aussi de valeur par défaut aux scripts annexes |
| `IDAH_BASE_DIR` | Dossier racine qui contient tes sets |
| `IDAH_DEFAULT_CATEGORY` | Classe de secours pour les ids inconnus (voir [Bon à savoir](#bon-à-savoir)) |
| `IDAH_SETS` | *Optionnel.* Sets à importer, par exemple `A1,B2`. Vide = tous |
| `IDAH_ENTRY_PREFIX` | *Optionnel.* Mettre `0` pour ne pas préfixer le nom des entrées par le set |
| `PICSELLIA_API_TOKEN` | Token API Picsellia |
| `PICSELLIA_DATASET_VERSION_ID` | Version de dataset Picsellia visée |

Le `.env` contient tes clés. Il est ignoré par git : ne le commite jamais.

## Importer un dataset sur IDAH

### Organisation des données

Le script attend un dossier racine qui contient un sous-dossier par set.
Chaque set a deux dossiers, `videos/` et `annotations/`, et chaque vidéo a un
fichier d'annotation du même nom :

```
by_subset/                 ← IDAH_BASE_DIR pointe ici
├── A1/
│   ├── videos/        clip_001.mp4, clip_002.mp4…
│   └── annotations/   clip_001.txt, clip_002.txt…
├── A2/
└── …
```

Les sets sont détectés tout seuls, jusqu'à trois niveaux de profondeur : tout
dossier qui contient `videos/` et `annotations/` est un set.

Les annotations sont au format VisDrone-MOT, avec une ligne par objet et par
image :

```
<frame>,<track_id>,<x>,<y>,<largeur>,<hauteur>,<score>,<classe>,<troncature>,<occlusion>
```

La classe est un id numérique (1 = pedestrian, 4 = car, 6 = truck,
9 = bus…), traduit en nom par la table `CATEGORY_MAP`. Le script écarte trois
types de lignes : l'id 0 (« ignored regions »), les lignes de score 0 et les
boîtes de taille nulle.

### 1. Vérifier avant d'envoyer

```bash
python import_video_et_annot_idah.py --dry-run
```

Rien ne part sur IDAH : le script lit les fichiers et fait l'inventaire. Tu
dois obtenir quelque chose comme ceci :

```
  A1             35 vidéo(s)    238 track(s)
  A2             35 vidéo(s)    242 track(s)
  …
  Categories : {'truck': 412, 'bicycle': 335, 'van': 281, 'tricycle': 214, 'bus': 100, 'awning-tricycle': 48}

LIGNES SOURCES
  retenues : 7889
  ecartees : 0
```

Arrête-toi là dans trois cas : un set manque, un bloc
`IDS DE CATEGORIE INCONNUS` apparaît, ou ta catégorie de secours figure dans
la liste des catégories.

### 2. Créer le projet et les datasets

```bash
python create_idah_datasets.py --project TASK_FORCE_GT --apply --create-project
```

Le script crée un projet dédié, puis un dataset par set, avec les couleurs de
labels déjà configurées. À la fin, il affiche l'`IDAH_PROJECT_ID` à reporter
dans le `.env`.

Sans `--apply`, il montre seulement ce qu'il ferait. Tu peux le relancer sans
risque : il ne touche pas aux datasets qui existent déjà.

### 3. Lancer l'import

```bash
PYTHONUNBUFFERED=1 python import_video_et_annot_idah.py --dataset-per-set 2>&1 | tee import_idah.log
```

Avec `--dataset-per-set`, chaque set part dans le dataset qui porte son nom.
Sans cette option, tout va dans `IDAH_DATASET_ID`, et le nom de chaque entrée
est préfixé par son set (`A1_clip_001.mp4`) pour éviter les collisions.

`PYTHONUNBUFFERED=1` sert à suivre le log en direct. Sans lui, Python garde sa
sortie en mémoire et le fichier reste vide pendant de longues minutes.

Ce qui prend du temps, c'est l'envoi des vidéos. La durée dépend donc surtout
de ta connexion montante : on a vu 210 vidéos passer en 18 minutes, et
d'autres fois en plus d'une heure. Le script se reconnecte tout seul quand son
token expire, et il réessaie en cas de coupure réseau.

À la fin, il affiche un récapitulatif :

```
SET             VIDÉOS     OK   SKIP   ÉCHEC   TRACKS   TR.KO
A1                  35     35      0       0      238       0
…
TOTAL              210    210      0       0     1390       0
```

Les colonnes à surveiller :

- `SKIP` : vidéos sans fichier d'annotation ;
- `ÉCHEC` : vidéos dont l'import a planté ;
- `TR.KO` : tracks refusées par l'API.

Toutes doivent rester à zéro. Une vidéo en erreur n'arrête pas les autres. Tu
relances ensuite uniquement les sets concernés avec `--sets B2,C1`, mais lis
d'abord la section [Relancer un set déjà
importé](#relancer-un-set-déjà-importé).

### 4. Contrôler le résultat

Ce petit script recompte, dataset par dataset, ce qui se trouve réellement sur
IDAH. Ses chiffres doivent correspondre au récapitulatif de l'import.

<details>
<summary>Script de vérification</summary>

```bash
python - <<'EOF'
import import_video_et_annot_idah as m

m.login()
base = m.CTX["base_url"]
headers = {"Accept": "application/vnd.api+json"}
url = f"{base}/datasets?filter[project_id]={m.CTX['proj_id']}&page[size]=100"
datasets = m.session.get(url, headers=headers).json()["data"]

for d in sorted(datasets, key=lambda x: x["attributes"]["name"]):
    m.CTX["ds_id"] = d["id"]
    m.load_entry_cache(force=True)
    tracks = sum(
        len(m.session.get(f"{base}/annotations?filter[entry_id]={e}&page[size]=500").json().get("data", []))
        for e in m.ENTRY_CACHE.values()
    )
    print(f"{d['attributes']['name']:<6} {len(m.ENTRY_CACHE):>4} entrées {tracks:>6} tracks")
EOF
```

</details>

Dernier contrôle, à l'œil : ouvre une vidéo dans l'outil d'annotation. Deux
tracks de la même classe doivent avoir la même couleur. Sinon, lance
`python declare_idah_label_config.py --apply --dataset <uuid>`.

### Relancer un set déjà importé

Les annotations ne sont **pas dédupliquées**. Si tu réimportes un set
par-dessus lui-même, toutes ses tracks sont créées une deuxième fois. Il faut
donc d'abord le vider :

```bash
python delete_idah_entries.py --dataset <uuid-du-dataset> --match '.' --apply --force --reset-status
python import_video_et_annot_idah.py --dataset-per-set --sets B2
```

Les deux options de la première commande :

- `--force` est obligatoire pour tout supprimer. C'est un garde-fou contre un
  vidage accidentel.
- `--reset-status` débloque les entrées déjà ouvertes dans l'outil
  d'annotation, que l'API refuse sinon de supprimer.

## Bon à savoir

- **La catégorie de secours n'est pas une vraie classe.** Quand un id n'est
  pas dans `CATEGORY_MAP`, la ligne est quand même importée, sous
  `IDAH_DEFAULT_CATEGORY`, et le script le signale à la fin. Choisis une
  valeur qui saute aux yeux (`house` dans le modèle) pour repérer le problème
  tout de suite.
- **`CATEGORY_MAP` existe en deux exemplaires**, dans
  `import_video_et_annot_idah.py` et dans `visdrone_to_video_coco.py`. Si tu
  en modifies un, reporte la modification dans l'autre.
- **Une track garde la classe de sa première image.** Il arrive que VisDrone
  change la classe d'un objet en cours de route (c'est du bruit dans les
  données) : c'est alors la première classe qui est retenue.
- **Un code 200 de l'API IDAH ne prouve rien.** Certains champs acceptent
  n'importe quoi sans broncher. C'est pour ça que les scripts relisent
  toujours ce qu'ils viennent d'écrire.

## Importer sur Picsellia

Il y a trois étapes, dont une à faire à la main.

**1. Convertir les séquences VisDrone.** Le dossier d'entrée doit contenir
`sequences/` (un dossier d'images par séquence) et `annotations/` (un `.txt`
par séquence) :

```bash
python visdrone_to_video_coco.py --root /chemin/vers/VisDrone --out ./output_visdrone --fps 20
```

Tu obtiens `output_visdrone/clips/*.mp4` et
`output_visdrone/annotations/*_video_coco.json`.

**2. Envoyer les vidéos sur Picsellia**, dans la version de dataset visée. Le
script de l'étape suivante ne le fait pas, et il refuse d'importer une
annotation dont la vidéo est absente.

**3. Importer les annotations :**

```bash
python import_annot_to_picsellia.py --annotations_dir ./output_visdrone/annotations
```

Précise bien `--annotations_dir` : par défaut, le script cherche dans
`./output_visdrone_test/annotations`.

Il ne reste plus qu'à rafraîchir le Video Studio de Picsellia pour voir le
résultat.
