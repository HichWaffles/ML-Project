import os
import sys
from pathlib import Path
import json
import traceback

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

# Add project root to sys path to import src modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import RAW_DATA_PATH, MODELS_DIR, MODEL_PATHS, load_model
from src.preprocessing import prepare_features, transform_test
from src.utils.logger import logger

app = Flask(__name__)

# --- App State/Globals ---
fitted_artifacts = None
model = None
app_schema = None
seg_artifacts = None
df_sample_pool = None

def init_app():
    global fitted_artifacts, model, app_schema, seg_artifacts, df_sample_pool
    logger.info("Initializing Web App resources...")
    
    # 1. Load artifacts and model
    try:
        fitted_artifacts = joblib.load(MODELS_DIR / "fitted_artifacts.joblib")
        # Load XGBoost as the primary real-time inference model
        model = load_model(MODEL_PATHS["xgboost"])
        
        # Load Segmentation Models
        seg_artifacts_path = MODELS_DIR / "segmentation_artifacts.joblib"
        if seg_artifacts_path.exists():
            seg_artifacts = joblib.load(seg_artifacts_path)
            logger.info("Successfully loaded Segmentation Artifacts.")
            
        logger.info("Successfully loaded XGBoost model and fitted artifacts.")
    except Exception as e:
        logger.error(f"Failed to load model/artifacts: {e}")
        
    # 1B. Load clean sample pool for UI UX
    try:
        df_full = pd.read_csv(RAW_DATA_PATH)
        df_clean = df_full.dropna()
        sentinels = {'', 'nan', 'null', 'none', 'nat', 'missing', 'unknown', 'unspecified', 'inconnu'}
        
        for pool_col in df_clean.select_dtypes(include=['object']):
            df_clean = df_clean[~df_clean[pool_col].astype(str).str.lower().str.strip().isin(sentinels)]
            
        if 'SupportTicketsCount' in df_clean.columns:
            df_clean = df_clean[~df_clean['SupportTicketsCount'].isin([-1, 999])]
        if 'SatisfactionScore' in df_clean.columns:
            df_clean = df_clean[~df_clean['SatisfactionScore'].isin([-1, 0, 99])]
            
        df_sample_pool = df_clean
        logger.info(f"Successfully loaded {len(df_sample_pool)} clean sample rows.")
    except Exception as e:
        logger.error(f"Failed to build sample pool: {e}")

    # 2. Automatically generate the schema for the UI based on actual raw training data
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH, nrows=500) # Read enough rows to capture categorical unique values
        schema = {}
        
        # We don't want the user inputting IDs or targets
        cols_to_skip = {'CustomerID', 'Churn', 'Recency', 'ChurnRiskCategory'}
        
        for col in df_raw.columns:
            if col in cols_to_skip: 
                continue
                
            dtype = str(df_raw[col].dtype)
            
            # Special manual overrides for logic
            if col == "LastLoginIP":
                schema[col] = {"type": "text", "default": "8.8.8.8"}
                continue
            if "Date" in col:
                schema[col] = {"type": "date", "default": "2023-01-01"}
                continue
                
            if dtype == "object":
                raw_uniques = df_raw[col].dropna().unique()
                unique_vals = []
                for val in raw_uniques:
                    s_val = str(val).strip()
                    if s_val.lower() not in {'', 'nan', 'null', 'none', 'nat', 'missing', 'unknown', 'unspecified', 'inconnu'}:
                        unique_vals.append(s_val)
                        
                # If there are too many uniques, make it text. Otherwise select.
                if len(unique_vals) > 20:
                    schema[col] = {"type": "text", "default": unique_vals[0] if unique_vals else ""}
                else:
                    schema[col] = {"type": "select", "values": unique_vals, "default": unique_vals[0] if unique_vals else ""}
            elif "int" in dtype or "float" in dtype:
                # Filter out known numeric sentinels before calculating default median
                s_subset = df_raw[col].copy()
                if col == "SupportTicketsCount":
                    s_subset = s_subset.replace([-1, 999], np.nan)
                elif col == "SatisfactionScore":
                    s_subset = s_subset.replace([-1, 0, 99], np.nan)
                    
                median_val = s_subset.median()
                schema[col] = {"type": "number", "default": float(median_val) if pd.notna(median_val) else 0.0}
            elif dtype == "bool":
                schema[col] = {"type": "select", "values": ["True", "False"], "default": "False"}
                
        app_schema = schema
        logger.info("Successfully built dynamic UI form schema.")
    except Exception as e:
        logger.error(f"Failed to build schema: {e}")
        app_schema = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/api/schema", methods=["GET"])
def get_schema():
    if not app_schema:
        return jsonify({"error": "Schema not initialized"}), 500
    return jsonify(app_schema)

@app.route("/api/sample", methods=["GET"])
def get_sample():
    if df_sample_pool is None or df_sample_pool.empty:
        return jsonify({"error": "No valid sample pool available."}), 500
        
    sample_row = df_sample_pool.sample(1).iloc[0]
    
    result = {}
    for col, meta in app_schema.items():
        if col in sample_row.index:
            val = sample_row[col]
            if pd.isna(val):
                continue
            if meta["type"] == "number":
                result[col] = float(val)
            elif meta["type"] == "date":
                # Ensure date string keeps HTML5 input format 'YYYY-MM-DD'
                try:
                    result[col] = pd.to_datetime(val).strftime('%Y-%m-%d')
                except Exception:
                    result[col] = str(val).split(" ")[0]
            else:
                result[col] = str(val)
                
    return jsonify({"success": True, "data": result})

@app.route('/api/predict', methods=['POST'])
def run_prediction():
    if fitted_artifacts is None or model is None:
        return jsonify({"error": "Model not loaded on the backend."}), 500
        
    try:
        user_input = request.json
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
            
        # Convert strings back to correct types matching schema expected
        # (The frontend sends everything as structural strings/numbers but pandas takes care of most of it)
        
        # 1. Wrap dict into 1-row DataFrame
        df_input = pd.DataFrame([user_input])
        
        # 2. Structural pre-transform
        processed_df = prepare_features(df_input)
        
        # 3. Predict segments if artifacts exist
        segments_output = {}
        if seg_artifacts is not None:
            import numpy as np
            for theme, art in seg_artifacts.items():
                feats = art["features"]
                df_theme = processed_df[feats].copy()
                if "SupportTicketsCount" in df_theme.columns:
                    df_theme["SupportTicketsCount"] = df_theme["SupportTicketsCount"].replace([-1, 999], np.nan)
                
                # Impute missing values with 0 so predict doesn't fail
                df_theme = df_theme.fillna(0)
                
                scaled = art["scaler"].transform(df_theme)
                cluster_id = art["kmeans"].predict(scaled)[0]
                segments_output[theme] = art["names"].get(cluster_id, "Unknown")
        
        # 4. Model-dependent preprocessing pipeline
        X_test_pca = transform_test(processed_df, fitted_artifacts)
        
        # 5. Predict Churn
        prediction = model.predict(X_test_pca)[0]
        probability = model.predict_proba(X_test_pca)[0, 1]
        
        return jsonify({
            "success": True,
            "prediction": int(prediction),
            "probability": float(probability),
            "segments": segments_output
        })

    except Exception as e:
        logger.error(f"Prediction Error: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # Initialize globals before starting server
    init_app()
    app.run(host='0.0.0.0', port=5001, debug=True)
