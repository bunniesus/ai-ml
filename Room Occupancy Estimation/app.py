import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Room Occupancy Estimator",
    page_icon="🏢",
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #111827, #1f2937);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
    .hero p {margin: 0; color: #d1d5db;}
    .result-card {
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #374151;
        background: #111827;
        text-align: center;
    }
    .result-number {font-size: 3rem; font-weight: 800; margin: .2rem 0;}
    .occupant-icons {
        font-size: 2.6rem;
        letter-spacing: .25rem;
        margin: .5rem 0 .2rem 0;
    }
    .occupancy-about {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid #374151;
        background: #111827;
        margin: .7rem 0 1.2rem 0;
    }
    .occupancy-about h4 {margin: 0 0 .4rem 0;}
    .occupancy-about p {margin: .25rem 0; color: #d1d5db;}
    /* Keep sensor controls compact while leaving date/time full width. */
    div[data-testid="stNumberInput"] {
        max-width: 240px;
    }
    div[data-testid="stNumberInput"] input {
        padding-top: .45rem;
        padding-bottom: .45rem;
    }
    .muted {color: #6b7280;}
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        padding: 12px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🏢 Room Occupancy Estimator</h1>
    <p>Machine Learning based estimation of the number of people in a room using environmental sensor readings.</p>
</div>
""", unsafe_allow_html=True)

FEATURE_COLS = [
    "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
    "S1_Light", "S2_Light", "S3_Light", "S4_Light",
    "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
    "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR",
    "Hour", "Minute", "DayOfWeek",
    "Avg_Temp", "Avg_Light", "Avg_Sound", "PIR_Total"
]

DISPLAY_NAMES = {
    0: "Empty",
    1: "1 Occupant",
    2: "2 Occupants",
    3: "3 Occupants",
}

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("Occupancy_Estimation.csv")

@st.cache_resource
def train_models(df):
    df = df.drop_duplicates().reset_index(drop=True).copy()

    if "Date" in df.columns and "Time" in df.columns:
        dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
        df["Hour"] = dt.dt.hour
        df["Minute"] = dt.dt.minute
        df["DayOfWeek"] = dt.dt.dayofweek
    else:
        raise ValueError("Dataset must contain Date and Time columns.")

    df["Avg_Temp"] = df[["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp"]].mean(axis=1)
    df["Avg_Light"] = df[["S1_Light", "S2_Light", "S3_Light", "S4_Light"]].mean(axis=1)
    df["Avg_Sound"] = df[["S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]].mean(axis=1)
    df["PIR_Total"] = df["S6_PIR"] + df["S7_PIR"]

    missing = [c for c in FEATURE_COLS + ["Room_Occupancy_Count"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    X = df[FEATURE_COLS]
    y = df["Room_Occupancy_Count"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, pred, average="weighted", zero_division=0),
        }

    return df, results

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Project Setup")
    uploaded = st.file_uploader(
        "Upload Occupancy_Estimation.csv",
        type=["csv"],
        help="Leave empty if Occupancy_Estimation.csv is in the same folder as this app."
    )
    st.caption("The app follows the same feature engineering and 80/20 stratified split used in the ML notebook.")

try:
    df, results = train_models(load_data(uploaded))
except Exception as e:
    st.error("Dataset could not be loaded.")
    st.code(str(e))
    st.info("Place `Occupancy_Estimation.csv` in the same folder as `app.py`, or upload it from the sidebar.")
    st.stop()

best_name = max(results, key=lambda x: results[x]["Accuracy"])
best_model = results[best_name]["model"]

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔮 Predict Occupancy", "📊 Model Performance", "ℹ️ About Project"])

with tab1:
    st.subheader("About Room Occupancy")
    st.markdown("""
    <div class="occupancy-about">
        <h4>🏢 What does this model estimate?</h4>
        <p>The system estimates how many people are present in a room using temperature, light, sound, CO₂ and PIR sensor readings.</p>
        <p><b>Current model scope:</b> 0–3 occupants, because the training dataset contains only these four occupancy classes.</p>
        <p>So the visualization below represents the model's actual prediction rather than inventing classes the model was never trained on.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Enter Room Sensor Readings")
    st.caption("The four averaged features and PIR total are calculated automatically from the sensor inputs.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        s1_temp = st.number_input("S1 Temperature (°C)", value=24.94, step=0.1)
        s1_light = st.number_input("S1 Light", value=121, step=1)
        s1_sound = st.number_input("S1 Sound", value=0.08, step=0.01)
        s5_co2 = st.number_input("S5 CO₂", value=390, step=1)
    with c2:
        s2_temp = st.number_input("S2 Temperature (°C)", value=24.75, step=0.1)
        s2_light = st.number_input("S2 Light", value=34, step=1)
        s2_sound = st.number_input("S2 Sound", value=0.08, step=0.01)
        co2_slope = st.number_input("S5 CO₂ Slope", value=-0.02, step=0.01)
    with c3:
        s3_temp = st.number_input("S3 Temperature (°C)", value=24.56, step=0.1)
        s3_light = st.number_input("S3 Light", value=53, step=1)
        s3_sound = st.number_input("S3 Sound", value=0.05, step=0.01)
        s6_pir = st.number_input("S6 PIR", value=1, step=1, min_value=0)
    with c4:
        s4_temp = st.number_input("S4 Temperature (°C)", value=25.38, step=0.1)
        s4_light = st.number_input("S4 Light", value=40, step=1)
        s4_sound = st.number_input("S4 Sound", value=0.05, step=0.01)
        s7_pir = st.number_input("S7 PIR", value=1, step=1, min_value=0)

    dt = st.date_input("Date", value=datetime(2017, 12, 22).date())
    tm = st.time_input("Time", value=datetime(2017, 12, 22, 10, 50).time())

    dt_obj = datetime.combine(dt, tm)
    hour = dt_obj.hour
    minute = dt_obj.minute
    day_of_week = dt_obj.weekday()

    avg_temp = np.mean([s1_temp, s2_temp, s3_temp, s4_temp])
    avg_light = np.mean([s1_light, s2_light, s3_light, s4_light])
    avg_sound = np.mean([s1_sound, s2_sound, s3_sound, s4_sound])
    pir_total = s6_pir + s7_pir

    with st.expander("Calculated Features"):
        st.write({
            "Hour": hour,
            "Minute": minute,
            "DayOfWeek": day_of_week,
            "Avg_Temp": round(avg_temp, 3),
            "Avg_Light": round(avg_light, 3),
            "Avg_Sound": round(avg_sound, 3),
            "PIR_Total": pir_total
        })

    if st.button("🔍 Estimate Room Occupancy", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "S1_Temp": s1_temp, "S2_Temp": s2_temp, "S3_Temp": s3_temp, "S4_Temp": s4_temp,
            "S1_Light": s1_light, "S2_Light": s2_light, "S3_Light": s3_light, "S4_Light": s4_light,
            "S1_Sound": s1_sound, "S2_Sound": s2_sound, "S3_Sound": s3_sound, "S4_Sound": s4_sound,
            "S5_CO2": s5_co2, "S5_CO2_Slope": co2_slope,
            "S6_PIR": s6_pir, "S7_PIR": s7_pir,
            "Hour": hour, "Minute": minute, "DayOfWeek": day_of_week,
            "Avg_Temp": avg_temp, "Avg_Light": avg_light, "Avg_Sound": avg_sound,
            "PIR_Total": pir_total
        }])[FEATURE_COLS]

        prediction = int(best_model.predict(row)[0])

        if hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(row)[0]
            classes = best_model.classes_
            confidence = float(np.max(probabilities))
        else:
            confidence = None

        st.markdown("---")
        left, right = st.columns([1, 1.5])

        with left:
            with st.container(border=True):
                st.markdown("### Estimated Occupancy")
                st.markdown(f'<div class="result-number">{prediction}</div>', unsafe_allow_html=True)
                label = DISPLAY_NAMES.get(prediction, str(prediction))
                st.markdown(f"**{label}**")

                if prediction == 0:
                    st.markdown('<div class="occupant-icons">🚪</div>', unsafe_allow_html=True)
                    st.caption("No occupants detected")
                else:
                    icons = " ".join(["🧍"] * prediction)
                    st.markdown(f'<div class="occupant-icons">{icons}</div>', unsafe_allow_html=True)
                    st.caption(f"{prediction} occupant{'s' if prediction != 1 else ''} represented")

                if confidence is not None:
                    st.progress(confidence)
                    st.caption(f"Model confidence: {confidence:.1%}")

        with right:
            st.info(
                f"Prediction generated using **{best_name}**, the best-performing model "
                f"from the project evaluation."
            )
            st.write("**Derived sensor features**")
            st.write(
                f"Average temperature: `{avg_temp:.2f} °C`  ·  "
                f"Average light: `{avg_light:.2f}`  ·  "
                f"Average sound: `{avg_sound:.3f}`  ·  "
                f"PIR total: `{pir_total}`"
            )

with tab2:
    st.subheader("Model Comparison")
    comparison = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": values["Accuracy"],
            "Precision": values["Precision"],
            "Recall": values["Recall"],
            "F1-Score": values["F1-Score"]
        }
        for name, values in results.items()
    ]).set_index("Model")

    st.dataframe(
        comparison.style.format("{:.4f}"),
        use_container_width=True
    )

    st.bar_chart(comparison)

    st.success(
        f"Best model: **{best_name}** with accuracy **{results[best_name]['Accuracy']:.2%}**."
    )

    if best_name == "Random Forest":
        st.subheader("Top Random Forest Features")
        rf = results["Random Forest"]["model"]
        importance = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=False).head(10).set_index("Feature")
        st.bar_chart(importance)

    st.caption(
        "The evaluation uses the same 80/20 stratified train-test split and model settings "
        "as the provided ML case study."
    )

with tab3:
    st.subheader("About the Mini Project")
    st.write(
        "This web interface demonstrates the Room Occupancy Estimation ML case study. "
        "Sensor readings are transformed into the same 23-feature input used during model training."
    )

    a, b, c = st.columns(3)
    a.metric("Dataset Records", f"{len(df):,}")
    b.metric("Input Features", "23")
    c.metric("Best Accuracy", f"{results[best_name]['Accuracy']:.2%}")

    st.markdown("### ML Pipeline")
    st.markdown("""
    **Sensor readings → Feature Engineering → Train/Test Split → ML Model → Occupancy Prediction**

    - Temperature, light, sound, CO₂ and PIR sensor values are used.
    - Date and time produce `Hour`, `Minute` and `DayOfWeek`.
    - Average temperature, light and sound are calculated.
    - `PIR_Total` is calculated from the two PIR sensors.
    - Three classifiers are evaluated: Logistic Regression, KNN and Random Forest.
    """)

    st.markdown("### Occupancy Classes")
    st.write("The target variable `Room_Occupancy_Count` contains four classes: 0, 1, 2 and 3 occupants.")
