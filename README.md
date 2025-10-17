# Crop Disease Detection

Detect crop diseases from leaf images using a ResNet50-based CNN with Grad-CAM visualizations.

## Project Overview
- Dataset: PlantVillage (38 classes) from Kaggle
- Model: ResNet50 transfer learning, data augmentation, class weighting
- Demo: Gradio UI for image upload, prediction, and Grad-CAM heatmap

## Requirements
- Windows 10/11, Python 3.10+ (tested on 3.13)
- A virtual environment at `.venv` (created by you or PyCharm)
- GPU optional; CPU works but training is slower

## Setup (cmd.exe)
```cmd
cd /d "C:\Users\sm01\PycharmProjects\Crop Disease Detection"

:: create venv if you don't have one yet
python -m venv .venv

:: activate venv
call .venv\Scripts\activate.bat

:: install dependencies
python -m pip install --upgrade --upgrade-strategy eager -r requirements.txt
```

## Dataset (optional if you already have it)
Download from Kaggle and unzip. On Windows, use PowerShell's Expand-Archive.

```cmd
:: ensure Kaggle is installed
python -m pip install -U kaggle

:: download zip into data/
python -m kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data

:: unzip with PowerShell
powershell -NoProfile -Command "Expand-Archive -Path 'data\plantvillage-dataset.zip' -DestinationPath 'data\plantvillage' -Force"
```

The expected images directory used by the code is:
```
data/plantvillage/plantvillage dataset/color
```

## Run the demo
```cmd
call .venv\Scripts\activate.bat
set GRADIO_SERVER_NAME=127.0.0.1
set GRADIO_SERVER_PORT=7861
python 03_demo.py
```
Then open http://127.0.0.1:7861 in your browser.

## Notes
- If you see UndefinedMetricWarning in sklearn during evaluation, it means some classes had zero predicted samples; consider training longer or balancing data.
- For unzip errors on Windows, use Expand-Archive as shown (no built-in `unzip` in cmd).

## Git hygiene
A `.gitignore` is included to exclude `.venv/`, `data/`, `models/`, caches, and secrets (e.g., `kaggle.json`). If you already committed large files, run:
```cmd
git rm -r --cached .
git add .
git commit -m "Apply .gitignore and remove tracked artifacts"
```

## First push to GitHub
```cmd
:: inside repo root
git init
git add .
git commit -m "Initial commit: crop disease detection"
:: create a repo on GitHub, then:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

## Files
- `01_eda_crop_disease.ipynb` – Exploratory data analysis
- `02_train_model.ipynb` – Training notebook (saves model to `models/`)
- `03_demo.py` – Gradio demo (loads `models/resnet50_crop_disease.h5`)
- `requirements.txt` – Reproducible dependencies
- `.gitignore` – Excludes venv, datasets, models, caches, secrets
