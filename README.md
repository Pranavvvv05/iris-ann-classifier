---
title: Iris ANN Classifier
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
---

# Iris Field Guide — ANN Classifier

A 3-layer neural network (16 → 8 → 3 units, trained with Keras) that
identifies an Iris flower's species — *setosa*, *versicolor*, or
*virginica* — from its sepal and petal measurements.
**Live Demo:**iris-ann-classifier.com( https://iris-ann-classifier-p2da.onrender.com)

Enter the four measurements in the UI and the model returns the predicted
species along with its confidence across all three classes.

## Stack

- **Model:** Keras Sequential ANN, trained on the classic 150-row Iris
  dataset (Fisher, 1936)
- **Backend:** Flask (`app.py`) serving a `/predict` endpoint
- **Frontend:** Vanilla HTML/CSS/JS (`templates/`, `static/`)
- **Deployment:** Docker, trains the model at image build time
  (`train_model.py`)

## Running locally

```bash
pip install -r requirements.txt
python train_model.py   # produces model.h5, scaler.pkl, classes.json
python app.py            # serves at http://localhost:7860
```
