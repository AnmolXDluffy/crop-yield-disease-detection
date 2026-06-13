# Crop Disease, Yield Prediction and Recommendation System

A machine learning project with three separate modules:

- Disease detection from plant leaf images
- Crop yield prediction from tabular climate/agriculture data
- Crop recommendation from soil nutrients and weather inputs

The frontend is built with Streamlit and loads trained local models from the `models/` folder.

## Project Structure

```text
.
├── app.py
├── run_frontend.bat
├── requirements.txt
├── models/
│   ├── plant_disease_model.pth
│   ├── disease_classes.json
│   ├── yield_model.pkl
│   ├── crop_recommendation_model.pkl
│   └── model_metrics.json
├── notebooks/
│   ├── disease/
│   │   ├── plant-diseases-detection-pytorch.ipynb
│   │   └── test_saved_model.ipynb
│   ├── yield/
│   │   ├── yield_prediction.ipynb
│   │   └── test_yield_model.ipynb
│   └── recommendation/
│       └── recommendation_system.ipynb
├── yield_data/
│   ├── yield_df.csv
│   ├── yield.csv
│   ├── rainfall.csv
│   ├── temp.csv
│   └── pesticides.csv
└── recommend_data/
    └── Crop_recommendation.csv
```

## Models

| Module | Model file | Purpose |
|---|---|---|
| Disease Detection | `models/plant_disease_model.pth` | Predicts disease class from leaf image |
| Yield Prediction | `models/yield_model.pkl` | Predicts crop yield from tabular inputs |
| Crop Recommendation | `models/crop_recommendation_model.pkl` | Recommends crop from soil/weather inputs |

## Run the Frontend

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python -m streamlit run app.py
```

On Windows, you can also double-click:

```text
run_frontend.bat
```

Then open:

```text
http://localhost:8501
```

## Frontend Tabs

### Disease Detection

Upload one or more leaf images. The app predicts the crop/disease class using the trained PyTorch model.

### Yield Prediction

Select area, crop, year, rainfall, pesticide use, and average temperature. The app predicts yield in `hg/ha` and `tonnes/ha`.

### Crop Recommendation

Enter N, P, K, temperature, humidity, pH, and rainfall. The app recommends a suitable crop with confidence scores.

## Dataset Notes

The full plant disease image dataset is not included in the GitHub repository because it is large. It is ignored through `.gitignore` as:

```text
archive/
```

The trained disease model and `models/disease_classes.json` are included, so the frontend can still run disease prediction without the full training dataset.

If you want to retrain the disease model, place the PlantVillage-style dataset back into:

```text
archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/
```

## Important Limitations

- Disease detection, yield prediction, and crop recommendation are intentionally separate modules.
- Yield prediction supports the crops available in the yield dataset, not every crop in the disease dataset.
- Predictions are educational/project outputs and should not replace expert agricultural advice.

## GitHub Notes

The model files are kept under GitHub's 100 MB single-file limit. If future models become larger, use Git LFS or upload large models through GitHub Releases.
=======
# crop-yield-disease-detection
An AI-powered system for crop yield prediction and plant disease detection using machine learning and agricultural data analysis.

