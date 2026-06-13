import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DISEASE_MODEL_PATH = MODELS_DIR / "plant_disease_model.pth"
YIELD_MODEL_PATH = MODELS_DIR / "yield_model.pkl"
RECOMMENDATION_MODEL_PATH = MODELS_DIR / "crop_recommendation_model.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
DISEASE_CLASSES_PATH = MODELS_DIR / "disease_classes.json"

DISEASE_TRAIN_DIR = BASE_DIR / "archive" / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "train"
YIELD_DATA_PATH = BASE_DIR / "yield_data" / "yield_df.csv"


st.set_page_config(
    page_title="Crop Intelligence System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    .stApp {
        background: linear-gradient(135deg, #fff7ed 0%, #ecfdf5 34%, #eff6ff 68%, #fdf2f8 100%);
        color: #172033;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #172033;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #14532d 0%, #0f766e 48%, #2563eb 100%);
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        color: #ffffff;
        font-size: 2.45rem;
        letter-spacing: 0;
        line-height: 1.08;
    }
    .hero p {
        margin: 0;
        color: rgba(255,255,255,0.9);
        font-size: 1.06rem;
        max-width: 820px;
    }
    .module-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 1rem 0 1.3rem 0;
    }
    .module-card {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
    }
    .module-card strong {
        display: block;
        font-size: 1.02rem;
        margin-bottom: 0.25rem;
        color: #111827;
    }
    .module-card small {
        color: #475569;
        font-size: 0.88rem;
    }
    .result-panel {
        padding: 1.05rem;
        border-radius: 14px;
        border: 1px solid rgba(15, 23, 42, 0.10);
        background: #ffffff;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.09);
        margin-bottom: 1rem;
    }
    div.stButton > button:first-child {
        border-radius: 999px;
        border: 0;
        background: linear-gradient(135deg, #16a34a, #0891b2);
        color: white;
        font-weight: 700;
        min-height: 2.85rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
        box-shadow: 0 10px 24px rgba(8, 145, 178, 0.28);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #15803d, #0e7490);
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.62);
        border-radius: 999px;
        padding: 0.35rem;
        width: fit-content;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.6rem 1.1rem;
        color: #334155;
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background: #111827;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] *,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] p {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"],
    [data-baseweb="input"] > div,
    [data-baseweb="base-input"],
    [data-baseweb="base-input"] > div {
        background-color: rgba(255, 255, 255, 0.96) !important;
        color: #172033 !important;
        border-color: rgba(15, 23, 42, 0.14) !important;
    }
    [data-baseweb="select"] input,
    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input,
    [data-baseweb="input"] textarea,
    [data-baseweb="base-input"] textarea {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        caret-color: #172033 !important;
    }
    [data-baseweb="select"] svg {
        fill: #334155 !important;
    }
    [data-testid="stFileUploader"] button {
        border-radius: 999px !important;
        border: 0 !important;
        background: linear-gradient(135deg, #16a34a, #0891b2) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 24px rgba(8, 145, 178, 0.24) !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, #15803d, #0e7490) !important;
        color: #ffffff !important;
    }
    [data-testid="stFileUploader"] section {
        background: #ffffff;
        border: 1px dashed #0891b2;
        border-radius: 14px;
        color: #172033;
    }
    [data-testid="stFileUploader"] small {
        color: #475569;
    }
    .stAlert {
        background: rgba(255,255,255,0.8);
        border-radius: 12px;
    }
    @media (max-width: 900px) {
        .module-strip {
            grid-template-columns: 1fr;
        }
        .hero h1 {
            font-size: 1.85rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class CNNNeuralNet(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)


@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_data(show_spinner=False)
def load_disease_classes():
    if DISEASE_CLASSES_PATH.exists():
        return json.loads(DISEASE_CLASSES_PATH.read_text(encoding="utf-8"))
    return sorted(path.name for path in DISEASE_TRAIN_DIR.iterdir() if path.is_dir())


@st.cache_resource(show_spinner=False)
def load_disease_model():
    device = get_device()
    classes = load_disease_classes()
    model = CNNNeuralNet(3, len(classes))
    state_dict = torch.load(DISEASE_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_yield_model():
    return joblib.load(YIELD_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_recommendation_model():
    return joblib.load(RECOMMENDATION_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_yield_reference_data():
    df = pd.read_csv(YIELD_DATA_PATH).drop(columns=["Unnamed: 0"], errors="ignore").dropna()
    return df


def clean_label(label):
    return label.replace("___", " - ").replace("_", " ").replace("(maize)", "maize")


def disease_predict(image):
    device = get_device()
    model = load_disease_model()
    classes = load_disease_classes()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
        top_conf, top_idx = torch.topk(probabilities, k=min(5, len(classes)))
    top_predictions = [
        {"class": classes[idx.item()], "confidence": conf.item()}
        for conf, idx in zip(top_conf, top_idx)
    ]
    return classes[pred_idx.item()], confidence.item(), top_predictions


def yield_predict(area, crop, year, rainfall, pesticides, avg_temp):
    model = load_yield_model()
    row = pd.DataFrame([{
        "Area": area,
        "Item": crop,
        "Year": int(year),
        "average_rain_fall_mm_per_year": float(rainfall),
        "pesticides_tonnes": float(pesticides),
        "avg_temp": float(avg_temp),
    }])
    prediction = float(model.predict(row)[0])
    return row, prediction


def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    model = load_recommendation_model()
    row = pd.DataFrame([{
        "N": float(N),
        "P": float(P),
        "K": float(K),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "ph": float(ph),
        "rainfall": float(rainfall),
    }])
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    classes = model.named_steps["model"].classes_
    top = (
        pd.DataFrame({"Crop": classes, "Confidence": probabilities})
        .sort_values("Confidence", ascending=False)
        .head(5)
    )
    return row, prediction, top


st.markdown(
    """
    <div class="hero">
        <h1>Crop Intelligence System</h1>
        <p>Detect leaf diseases, estimate crop yield, and recommend crops from soil and climate inputs in one clean workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="module-strip">
        <div class="module-card"><strong>Disease Detection</strong><small>Upload leaf images and get clear disease predictions.</small></div>
        <div class="module-card"><strong>Yield Prediction</strong><small>Estimate yield from crop, region, weather, and pesticide inputs.</small></div>
        <div class="module-card"><strong>Crop Recommendation</strong><small>Find suitable crops using soil nutrients and climate conditions.</small></div>
    </div>
    """,
    unsafe_allow_html=True,
)

disease_tab, yield_tab, recommendation_tab = st.tabs([
    "Disease Detection",
    "Yield Prediction",
    "Crop Recommendation",
])

with disease_tab:
    st.subheader("Disease Detection")
    st.write("Upload one or more leaf images and get the predicted crop disease.")

    uploads = st.file_uploader(
        "Upload leaf image files",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="disease_uploads",
    )

    if uploads:
        for upload in uploads:
            image = Image.open(upload).convert("RGB")
            pred_class, confidence, top_predictions = disease_predict(image)
            left, right = st.columns([1, 1.4])
            with left:
                st.image(image, caption=upload.name, use_container_width=True)
            with right:
                st.markdown('<div class="result-panel">', unsafe_allow_html=True)
                st.markdown(f"### {clean_label(pred_class)}")
                st.write(f"Confidence: **{confidence * 100:.2f}%**")
                top_df = pd.DataFrame([
                    {
                        "Prediction": clean_label(item["class"]),
                        "Confidence": f"{item['confidence'] * 100:.2f}%",
                    }
                    for item in top_predictions
                ])
                st.dataframe(top_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Upload one or more leaf images to begin.")

with yield_tab:
    st.subheader("Yield Prediction")
    st.write("Enter field and climate details to estimate expected crop yield.")

    yield_df = load_yield_reference_data()
    areas = sorted(yield_df["Area"].unique())
    crops = sorted(yield_df["Item"].unique())

    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.selectbox("Area / Country", areas, index=areas.index("India") if "India" in areas else 0)
        year = st.number_input("Year", min_value=1961, max_value=2035, value=2024, step=1)
    with c2:
        crop = st.selectbox("Crop", crops, index=crops.index("Maize") if "Maize" in crops else 0)
        rainfall = st.number_input(
            "Average rainfall (mm/year)",
            min_value=0.0,
            value=float(yield_df["average_rain_fall_mm_per_year"].median()),
            step=10.0,
        )
    with c3:
        pesticides = st.number_input(
            "Pesticides (tonnes)",
            min_value=0.0,
            value=float(yield_df["pesticides_tonnes"].median()),
            step=100.0,
        )
        avg_temp = st.number_input(
            "Average temperature (C)",
            min_value=-20.0,
            max_value=60.0,
            value=float(yield_df["avg_temp"].median()),
            step=0.1,
        )

    if st.button("Predict Yield"):
        input_row, predicted_yield = yield_predict(area, crop, year, rainfall, pesticides, avg_temp)
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown(f"### Predicted yield: {predicted_yield:,.2f} hg/ha")
        st.write(f"Equivalent: **{predicted_yield / 10000:.4f} tonnes/ha**")
        st.dataframe(input_row, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Supported crops in this yield model"):
        st.write(", ".join(crops))

with recommendation_tab:
    st.subheader("Crop Recommendation")
    st.write("Enter soil and weather values to get a suitable crop recommendation.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        N = st.number_input("Nitrogen N", min_value=0.0, value=90.0, step=1.0)
        P = st.number_input("Phosphorus P", min_value=0.0, value=42.0, step=1.0)
    with c2:
        K = st.number_input("Potassium K", min_value=0.0, value=43.0, step=1.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    with c3:
        temperature = st.number_input("Temperature (C)", min_value=-20.0, max_value=60.0, value=20.88, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.5)
    with c4:
        rec_rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=202.94, step=1.0)

    if st.button("Recommend Crop"):
        row, prediction, top = recommend_crop(N, P, K, temperature, humidity, ph, rec_rainfall)
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown(f"### Recommended crop: {prediction.title()}")
        st.dataframe(row, use_container_width=True, hide_index=True)
        top_display = top.copy()
        top_display["Confidence"] = top_display["Confidence"].map(lambda value: f"{value * 100:.2f}%")
        st.dataframe(top_display, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
