# Projet Deep Learning — Segmentation binaire avec U-Net

Implémentation **from scratch** d'un réseau **U-Net** avec **PyTorch** pour
segmenter les images du dataset **COCO**. On sépare uniquement l'**avant-plan**
(les objets) de l'**arrière-plan** : c'est de la segmentation *binaire*, il n'y a
pas de classe sémantique (la segmentation sémantique est un bonus possible).

Contraintes du sujet respectées :
- image d'entrée RGB, masque de sortie de **même taille** que l'entrée ;
- évaluation sur des données **non vues** pendant l'entraînement (split validation) ;
- **PyTorch** uniquement, **pas de Lightning**, **pas de modèle pré-entraîné** ni
  d'implémentation existante de U-Net ;
- les choix sont expliqués dans le notebook.

## Structure

```
projet-Deep/
├── notebook.ipynb     # LIVRABLE : notebook auto-contenu (à exécuter puis exporter en HTML)
├── train.py           # entrainement en ligne de commande (pour le PC d'entrainement / Kaggle / Colab)
├── smoke_test.py      # test rapide du pipeline sur des données synthetiques (sans COCO)
├── requirements.txt
└── src/
    ├── model.py       # U-Net (DoubleConv, encodeur/decodeur, skip connections)
    ├── dataset.py     # CocoBinaryDataset (masque binaire) + SyntheticShapes (fallback)
    ├── metrics.py     # loss BCE + Dice, metriques Dice / IoU / accuracy
    ├── engine.py      # boucles train / eval + early stopping
    └── utils.py       # seed, device, visualisation
```

Le code des modules `src/` est aussi recopié dans `notebook.ipynb` pour que le
notebook soit **auto-contenu** (exigence du sujet).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Données COCO

Télécharger COCO 2017 (images + annotations) et les ranger ainsi :

```
data/coco/
├── train2017/                 # images
├── val2017/                   # images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Liens : <http://images.cocodataset.org/zips/train2017.zip>,
`val2017.zip`, `annotations_trainval2017.zip`. Sur **Kaggle** le dataset COCO 2017
est déjà disponible (il suffit d'adapter les chemins).

## Entraînement

Notebook (livrable) : ouvrir `notebook.ipynb`, mettre `use_coco = True` avec les
bons chemins dans la cellule *Configuration*, puis *Run all*. Enfin exporter en
HTML (`File > Save and Export Notebook As > HTML`).

Ou en ligne de commande :

```bash
python train.py \
  --train-images data/coco/train2017 \
  --train-ann    data/coco/annotations/instances_train2017.json \
  --val-images   data/coco/val2017 \
  --val-ann      data/coco/annotations/instances_val2017.json \
  --img-size 128 --batch-size 16 --epochs 30 --base 64
```

## Test rapide (sans COCO)

Pour vérifier que tout fonctionne sans rien télécharger (données synthétiques) :

```bash
python smoke_test.py
```

Le notebook se lance aussi sans COCO : s'il ne trouve pas les chemins COCO, il
bascule automatiquement sur le dataset synthétique.

## Livrable

Le rendu Moodle est l'**export HTML** du `notebook.ipynb` **entièrement exécuté**
(à faire sur le PC d'entraînement avec COCO).
