# KYC Risk Classification Dashboard (Standalone)

This repository contains a professional, standalone Streamlit application for analyzing and predicting synthetic KYC risk tiers.

## Architecture

This application loads the trained `kyc_risk_model.joblib` artifact directly into memory and performs all predictions locally within the dashboard. There is no external API or database requirement.

```text
Streamlit Dashboard  --loads-->  artifacts/kyc_risk_model.joblib
                                        |
                                        v
                                 app.py (predictions & UI)
```

## Setup & Local Development

1. Navigate to the dashboard directory:
   ```bash
   cd kyc-standalone-dashboard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

## Pages

1. **Overview:** Executive summary, disclaimers, and model performance KPIs.
2. **Risk Prediction:** A dynamic form populated from the schema for single-customer predictions.
3. **Batch Prediction:** Upload a CSV for bulk predictions with robust data validation.
4. **Model Performance:** Detailed metrics, heatmaps, and classification reports from held-out test data.
5. **Feature Analysis:** A breakdown of the top 15 feature importances used by the model.
6. **About the Model:** Comprehensive model card and limitations.
7. **System Status:** Local environment checks for required artifacts.

## Deployment

Since this is a standalone application, you can deploy it instantly on Streamlit Community Cloud:
1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Set the Main file path to `kyc-standalone-dashboard/app.py`.
4. Deploy!

No secrets or environment variables are required.
