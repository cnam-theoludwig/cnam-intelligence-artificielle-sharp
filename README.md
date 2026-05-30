# cnam-intelligence-artificielle-sharp

## À propos

Projet réalisé dans le cadre de la formation [Ingénieur en Informatique et Systèmes d'Information (SI), CNAM](https://www.itii-alsace.fr/formations/informatique-et-systemes-dinformation-le-cnam/), pour le module Intelligence Artificielle (IA).

[SUJET](./SUJET.md)

### Membres du groupe

- [Matys FREYERMUTH](https://github.com/Mfxof)
- [Théo LUDWIG](https://github.com/theoludwig)
- [Shamim SEDGHI](https://github.com/shamim130)

## Prérequis

- [Python](https://www.python.org/) v3.12.13
- [uv](https://docs.astral.sh)
- [Docker](https://www.docker.com/)

## Installation

```sh
# Cloner le dépôt
git clone git@github.com:cnam-theoludwig/cnam-intelligence-artificielle-sharp.git

# Se déplacer dans le répertoire
cd cnam-intelligence-artificielle-sharp

# Installer les dépendances
uv sync --frozen

# Copier le fichier d'environnement et remplir les variables
cp .env.example .env

# Télécharger le modèle entraîné
wget https://github.com/cnam-theoludwig/cnam-intelligence-artificielle-sharp/releases/download/v1.0.0/sharp.pt
```

## Utilisation

```sh
# Lancer l'entraînement
mkdir runs
uv run -m src.train | tee runs/train_log.txt

# Lancer l'inférence webcam
uv run -m src.predict

# Lancer le serveur web
uv run -m src.api

# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type-check
uv run ty check .

# Tests (with coverage)
uv run pytest --cov
```

## Serveur d'API

Le serveur expose le modèle YOLO via une API HTTP. Le modèle (`sharp.pt`) est chargé une seule fois au démarrage.

- `HOST` et `PORT` sont configurables via le fichier `.env` (par défaut `0.0.0.0:8080`).
- [FastAPI](https://fastapi.tiangolo.com/) définit les routes et la sérialisation, [Uvicorn](https://www.uvicorn.org/) est le serveur qui écoute sur le réseau.
- Le serveur sert aussi l'app web statique (`web/`) à la racine `/`.

### App Web

L'app (`web/`, HTML/CSS/JavaScript natif) est servi à la racine. Une fois le serveur lancé, ouvrir [http://127.0.0.1:8080/](http://127.0.0.1:8080/).

Le navigateur capture la webcam (`getUserMedia`), envoie chaque frame en JPEG à `POST /predict` (~30 fps), puis superpose les bounding boxes et leurs classes sur le flux. La somme des doigts détectés est affichée à côté.

### `POST /predict`

Reçoit une image (multipart, champ `file`), exécute l'inférence et renvoie les détections ainsi que la somme des doigts visibles.

```sh
curl -F "file=@image.jpg" http://127.0.0.1:8080/predict
```

Réponse :

```json
{
    "detections": [
        {
            "x_minimum": 330.25677490234375,
            "y_minimum": 94.41045379638672,
            "x_maximum": 543.9456787109375,
            "y_maximum": 397.6856689453125,
            "class_name": "2_fingers",
            "confidence": 0.7250035405158997
        }
    ],
    "total_fingers": 2
}
```
