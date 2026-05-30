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
uv run src/train.py | tee runs/train_log.txt

# Lancer l'inférence
uv run src/predict.py

# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type-check
uv run ty check .

# Tests (with coverage)
uv run pytest --cov
```
