import streamlit as st
import pandas as pd
import json
import joblib
import os
import datetime
from pathlib import Path

# --- Configuration ---
st.set_page_config(page_title="KYC Risk Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Constants & Paths ---
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# --- CSS Injection ---
css_path = BASE_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# --- Dictionary Definitions ---
FIELD_DESCRIPTIONS = {
    "age": "Customer age in years",
    "job": "Type of job",
    "marital": "Marital status",
    "education": "Highest level of education",
    "default": "Has credit in default?",
    "housing": "Has housing loan?",
    "loan": "Has personal loan?",
    "contact": "Contact communication type",
    "month": "Last contact month of year",
    "day_of_week": "Last contact day of the week",
    "duration": "Last contact duration (seconds)",
    "campaign": "Number of contacts performed during this campaign",
    "previous": "Number of contacts performed before this campaign",
    "poutcome": "Outcome of the previous marketing campaign",
    "previously_contacted": "Was the customer previously contacted? (1=yes, 0=no)",
    "pdays_clean": "Days since last contact (26 = never contacted)",
    "emp.var.rate": "Employment variation rate (quarterly)",
    "cons.price.idx": "Consumer price index (monthly)",
    "cons.conf.idx": "Consumer confidence index (monthly)",
    "euribor3m": "Euribor 3 month rate (daily)",
    "nr.employed": "Number of employees (quarterly)"
}

FIELD_GROUPS = {
    "Demographics": ["age", "job", "marital", "education"],
    "Financial standing": ["default", "housing", "loan"],
    "Contact & campaign history": ["contact", "month", "day_of_week", "duration", "campaign", "previous", "poutcome", "previously_contacted", "pdays_clean"],
    "Economic context": ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
}

# --- Data Loading (Cached) ---
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = ARTIFACT_DIR / "kyc_risk_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data(show_spinner=False)
def load_json_artifact(filename):
    filepath = ARTIFACT_DIR / filename
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return None

@st.cache_data(show_spinner=False)
def load_csv_artifact(filename):
    filepath = ARTIFACT_DIR / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    return None

def get_file_metadata(filename):
    filepath = ARTIFACT_DIR / filename
    if filepath.exists():
        size = filepath.stat().st_size
        mtime = filepath.stat().st_mtime
        return {
            "size_bytes": size,
            "modified_time": datetime.datetime.fromtimestamp(mtime).isoformat()
        }
    return None

def render_gauge(risk, probs_dict):
    p_low = probs_dict.get("Low", 0.0)
    p_med = probs_dict.get("Medium", 0.0)
    p_high = probs_dict.get("High", 0.0)
    
    if risk == "Low":
        pos = 16.66
    elif risk == "Medium":
        pos = 50.0
    elif risk == "High":
        pos = 83.33
    else:
        pos = 50.0

    low_cls = "highlight" if risk == "Low" else ""
    med_cls = "highlight" if risk == "Medium" else ""
    high_cls = "highlight" if risk == "High" else ""

    html = f"""
    <div class="risk-gauge-container">
        <div class="risk-gauge-track">
            <div class="risk-zone-low"></div>
            <div class="risk-zone-medium"></div>
            <div class="risk-zone-high"></div>
            <div class="risk-marker" style="left: {pos}%;"></div>
        </div>
        <div class="risk-labels">
            <div style="width: 33.33%; text-align: center;">LOW<br><span class="{low_cls}">{p_low*100:.1f}%</span></div>
            <div style="width: 33.33%; text-align: center;">MEDIUM<br><span class="{med_cls}">{p_med*100:.1f}%</span></div>
            <div style="width: 33.33%; text-align: center;">HIGH<br><span class="{high_cls}">{p_high*100:.1f}%</span></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Load artifacts
model = load_model()
metrics = load_json_artifact("metrics.json") or {}
feature_schema = load_json_artifact("feature_schema.json") or {}
label_map_data = load_json_artifact("label_map.json") or {}
classification_report = load_json_artifact("classification_report.json") or {}
sample_predictions = load_csv_artifact("sample_predictions.csv")
confusion_matrix = load_csv_artifact("confusion_matrix.csv")
feature_importances = load_csv_artifact("feature_importances.csv")

inverse_risk_map = {}
risk_map = label_map_data.get("risk_map", {})
if risk_map:
    inverse_risk_map = {v: k for k, v in risk_map.items()}

class_names = []
if model is not None and hasattr(model, "classes_"):
    class_names = [inverse_risk_map.get(c, str(c)) for c in model.classes_]
elif "label_names" in label_map_data:
    class_names = label_map_data["label_names"]

# --- Main App Header ---
st.markdown("""
<div class="top-header-bar">
    <h1>KYC RISK PREDICTOR</h1>
    <div class="live-indicator"><span class="live-dot"></span> Model live</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.markdown("<div style='font-family: \"Source Serif 4\", serif; font-size: 1.2rem; font-weight: 600; color: #0E1626; margin-bottom: 10px;'>NAVIGATION</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["Overview", "Risk Prediction", "Batch Prediction", "Model Performance", "Feature Analysis", "About the Model", "System Status"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("<span style='font-family: \"IBM Plex Sans\", sans-serif; font-size: 0.75rem; color: #4A5568;'>This model predicts a synthetic research-labeled risk tier derived from the UCI Bank Marketing dataset. It is not a regulatory KYC/AML determination.</span>", unsafe_allow_html=True)

# --- Pages ---

if page == "Overview":
    st.markdown("<div class='fieldset-container'><div class='fieldset-title'>Dashboard Overview</div>", unsafe_allow_html=True)
    st.write(
        "This application predicts synthetic Low, Medium, and High risk tiers based on the labeling policy "
        "used in the project dataset. It is an educational and analytical prototype."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if metrics:
            acc = metrics.get('accuracy', 0)
            f1 = f"{metrics.get('macro_f1', 0):.2%}"
            b_acc = f"{metrics.get('balanced_accuracy', 0):.2%}"
            html_model_card = f"""
            <div class="fieldset-container" style="padding: 1.5rem; border-radius: 4px;">
                <div class="hero-metric">{acc:.1%}</div>
                <div class="hero-label">MODEL ACCURACY</div>
                <table class="ledger-table" style="margin-bottom: 0;">
                    <tbody>
                        <tr><td>Macro F1</td><td>{f1}</td></tr>
                        <tr><td>Balanced Acc</td><td>{b_acc}</td></tr>
                        <tr><td>Test Samples</td><td>{metrics.get('test_samples', 'N/A')}</td></tr>
                    </tbody>
                </table>
            </div>
            """
            st.markdown(html_model_card, unsafe_allow_html=True)
            
    with col2:
        if metrics and "class_distribution_test" in metrics:
            st.markdown("<div class='fieldset-container'><div class='fieldset-title'>Test Set Distribution</div>", unsafe_allow_html=True)
            dist = metrics["class_distribution_test"]
            dist_mapped = {}
            for k, v in dist.items():
                if k.isdigit():
                    name = inverse_risk_map.get(int(k), k)
                    dist_mapped[name] = v
                else:
                    dist_mapped[k] = v
            df_dist = pd.DataFrame(list(dist_mapped.items()), columns=["Class", "Proportion"])
            st.bar_chart(df_dist.set_index("Class"))
            st.markdown("</div>", unsafe_allow_html=True)
            
    if sample_predictions is not None:
        st.markdown("<div class='fieldset-container'><div class='fieldset-title'>Sample Test-Set Predictions</div>", unsafe_allow_html=True)
        st.dataframe(sample_predictions.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Risk Prediction":
    st.markdown("<div class='fieldset-title' style='border: none; margin-bottom: 0;'>Single Customer Assessment</div>", unsafe_allow_html=True)
    
    if not model or not feature_schema:
        st.error("Model or feature schema not loaded. Cannot perform predictions.")
        st.stop()
        
    with st.form("single_prediction_form", border=False):
        input_data = {}
        numeric_feats = feature_schema.get("numeric_features", [])
        cat_feats = feature_schema.get("categorical_features", [])
        all_feats = numeric_feats + cat_feats
        
        # Render groups
        for group_name, cols_in_group in FIELD_GROUPS.items():
            valid_cols = [c for c in cols_in_group if c in numeric_feats or c in cat_feats]
            if not valid_cols:
                continue
                
            st.markdown(f"<div class='fieldset-container'><div class='fieldset-title'>{group_name}</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            
            for i, feat in enumerate(valid_cols):
                target_col = c1 if i % 2 == 0 else c2
                with target_col:
                    if feat in numeric_feats:
                        ranges = feature_schema["numeric_ranges"][feat]
                        min_val = float(ranges["min"])
                        max_val = float(ranges["max"])
                        default_val = min_val + (max_val - min_val)/2
                        input_data[feat] = st.number_input(feat, min_value=min_val, max_value=max_val, value=default_val, key=f"input_{feat}")
                        
                        caption_text = f"{FIELD_DESCRIPTIONS.get(feat, feat)}<br/>Range: {min_val:g} to {max_val:g}"
                        st.markdown(f"<span class='field-caption'>{caption_text}</span>", unsafe_allow_html=True)
                    else:
                        options = feature_schema["categorical_values"][feat]
                        input_data[feat] = st.selectbox(feat, options, key=f"input_{feat}")
                        
                        desc = FIELD_DESCRIPTIONS.get(feat, feat)
                        caption_text = f"{desc}<br/>{len(options)} options" if len(options) > 6 else f"{desc}"
                        st.markdown(f"<span class='field-caption'>{caption_text}</span>", unsafe_allow_html=True)
                        
            st.markdown("</div>", unsafe_allow_html=True)

        leftovers = [f for f in all_feats if not any(f in g for g in FIELD_GROUPS.values())]
        if leftovers:
            st.markdown(f"<div class='fieldset-container'><div class='fieldset-title'>Other Variables</div>", unsafe_allow_html=True)
            for feat in leftovers:
                if feat in numeric_feats:
                    min_val = float(feature_schema["numeric_ranges"][feat]["min"])
                    max_val = float(feature_schema["numeric_ranges"][feat]["max"])
                    default_val = min_val + (max_val - min_val)/2
                    input_data[feat] = st.number_input(feat, min_value=min_val, max_value=max_val, value=default_val, key=f"input_{feat}")
                    st.markdown(f"<span class='field-caption'>Range: {min_val:g} to {max_val:g}</span>", unsafe_allow_html=True)
                else:
                    options = feature_schema["categorical_values"][feat]
                    input_data[feat] = st.selectbox(feat, options, key=f"input_{feat}")
                    st.markdown(f"<span class='field-caption'>{len(options)} options</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        submit_btn = st.form_submit_button("Assess Risk")
        
        if submit_btn:
            input_df = pd.DataFrame([input_data])[all_feats]
            try:
                preds = model.predict(input_df)
                probs = model.predict_proba(input_df)
                
                pred_idx = preds[0]
                risk = inverse_risk_map.get(pred_idx, str(pred_idx))
                prob_array = probs[0]
                
                prob_dict = {}
                for j, class_val in enumerate(model.classes_):
                    class_name = inverse_risk_map.get(class_val, str(class_val))
                    prob_dict[class_name] = prob_array[j]
                    
                st.markdown(f"<div class='fieldset-container'><div class='fieldset-title'>Result: <span class='risk-badge-{risk.lower()}'>{risk.upper()} RISK</span></div>", unsafe_allow_html=True)
                render_gauge(risk, prob_dict)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "Batch Prediction":
    st.markdown("<div class='fieldset-container'>", unsafe_allow_html=True)
    st.markdown("<div class='fieldset-title'>Batch Upload</div>", unsafe_allow_html=True)
    st.markdown("<span style='font-family: \"IBM Plex Sans\", sans-serif; font-size: 0.9rem; color: #0E1626;'>Upload a CSV file containing multiple customer records.</span>", unsafe_allow_html=True)
    
    if not model or not feature_schema:
        st.error("Model or feature schema not loaded. Cannot perform predictions.")
        st.stop()
        
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        numeric_feats = feature_schema.get("numeric_features", [])
        cat_feats = feature_schema.get("categorical_features", [])
        expected_cols = numeric_feats + cat_feats
        
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            df_valid = df[expected_cols].copy()
            if st.button("Run Batch Prediction"):
                with st.spinner("Predicting..."):
                    preds = model.predict(df_valid)
                    probs = model.predict_proba(df_valid)
                    
                    df_results = df_valid.copy()
                    pred_labels = [inverse_risk_map.get(p, str(p)) for p in preds]
                    df_results["predicted_risk_level"] = pred_labels
                    
                    st.success(f"Successfully processed {len(df_results)} records.")
                    st.dataframe(df_results.head(100), use_container_width=True)
                    
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results", data=csv, file_name="batch_predictions.csv", mime="text/csv")
                    
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Performance":
    st.markdown("<div class='fieldset-container'><div class='fieldset-title'>Model Performance</div>", unsafe_allow_html=True)
    if metrics:
        acc = f"{metrics.get('accuracy', 0):.2%}"
        f1 = f"{metrics.get('macro_f1', 0):.2%}"
        b_acc = f"{metrics.get('balanced_accuracy', 0):.2%}"
        st.markdown(f"""
        <table class="ledger-table">
            <tbody>
                <tr><td>Accuracy</td><td>{acc}</td></tr>
                <tr><td>Macro F1</td><td>{f1}</td></tr>
                <tr><td>Balanced Accuracy</td><td>{b_acc}</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if confusion_matrix is not None:
        st.markdown("<div class='fieldset-container'><div class='fieldset-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        st.dataframe(confusion_matrix, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Feature Analysis":
    st.markdown("<div class='fieldset-container'><div class='fieldset-title'>Feature Analysis</div>", unsafe_allow_html=True)
    if feature_importances is not None:
        st.dataframe(feature_importances, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About the Model":
    st.markdown("<div class='fieldset-container'><div class='fieldset-title'>About the Model</div>", unsafe_allow_html=True)
    st.markdown("""
    - **Model type:** scikit-learn Pipeline (Preprocessing + Random Forest Classifier)
    - **Target classes:** Low, Medium, High
    - **Training approach:** Supervised learning on synthetic risk labels
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "System Status":
    st.markdown("<div class='fieldset-container'><div class='fieldset-title'>System Status</div>", unsafe_allow_html=True)
    st.write(f"**Artifact Directory:** `{ARTIFACT_DIR}`")
    if ARTIFACT_DIR.exists():
        st.success("Artifact directory is accessible.")
    if model is not None:
        st.success("Model loaded successfully.")
    st.markdown("</div>", unsafe_allow_html=True)
