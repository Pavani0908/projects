from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Load Models
custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
# Update these paths to your actual model paths
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
lstm_path = os.path.join(model_dir, "lstm_autoencoder.h5")
lgb_path = os.path.join(model_dir, "lightgbm_model.txt")

# Check if models exist, otherwise use placeholders for development
try:
    lstm_autoencoder = load_model(lstm_path, custom_objects=custom_objects)
    lgb_model = lgb.Booster(model_file=lgb_path)
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    # Create dummy models for development
    expected_feature_count = 10  # Default value
    lstm_autoencoder = None
    lgb_model = None
else:
    expected_feature_count = lstm_autoencoder.input_shape[-1]

# Initialize scaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Dummy training data for scaler (replace with real training data)
training_data = np.random.rand(1000, expected_feature_count)
scaler.fit(training_data)

# Attack Categories
attack_classes = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode',
                  'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']

def detect_anomaly(sample):
    if lstm_autoencoder is None:
        # Dummy implementation for development
        mse_loss = np.random.rand(sample.shape[0])
        is_anomaly = mse_loss > 0.5
        return mse_loss, ["Yes" if x else "No" for x in is_anomaly]
    
    reconstructed = lstm_autoencoder.predict(sample)
    mse_loss = np.mean(np.abs(reconstructed - sample), axis=(1, 2))
    is_anomaly = mse_loss > 3.0
    return mse_loss, ["Yes" if x else "No" for x in is_anomaly]

def classify_attack(sample, anomaly_labels):
    if lgb_model is None:
        # Dummy implementation for development
        return [attack_classes[np.random.randint(0, len(attack_classes))] if label == "Yes" else "N/A" 
                for label in anomaly_labels]
    
    preds = lgb_model.predict(sample)
    predicted_classes = np.argmax(preds, axis=1) if preds.ndim > 1 else preds
    return [attack_classes[int(cls)] if anomaly_labels[i] == "Yes" else "N/A"
            for i, cls in enumerate(predicted_classes)]

@app.route("/")
def index():
    return render_template("index.html", expected_feature_count=expected_feature_count, tables=None)

@app.route("/predict_file", methods=["POST"])
def predict_file():
    try:
        file = request.files["file"]
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(file)
        if df.shape[1] < expected_feature_count:
            return jsonify({"error": f"Expected {expected_feature_count} features, got {df.shape[1]}"}), 400
        elif df.shape[1] > expected_feature_count:
            df = df.iloc[:, :expected_feature_count]

        df.fillna(0, inplace=True)
        scaled_data = scaler.transform(df)
        lstm_input = scaled_data.reshape((scaled_data.shape[0], 1, expected_feature_count))

        mse_scores, anomaly_status = detect_anomaly(lstm_input)
        attack_categories = classify_attack(scaled_data, anomaly_status)

        results_df = pd.DataFrame()
        # Add original features
        for i in range(min(5, expected_feature_count)):  # Show only first 5 features to save space
            results_df[f"Feature_{i+1}"] = df.iloc[:, i]
        
        # Add prediction results
        results_df["Anomaly Score"] = mse_scores
        results_df["Anomaly Detected"] = anomaly_status
        results_df["Predicted Attack Category"] = attack_categories

        # Save full results for download
        full_df = df.copy()
        full_df["Anomaly Score"] = mse_scores
        full_df["Anomaly Detected"] = anomaly_status
        full_df["Predicted Attack Category"] = attack_categories
        
        output_file = os.path.join(app.static_folder, "predictions.csv")
        full_df.to_csv(output_file, index=False)

        # Store the filename to display it after form submission
        filename = file.filename

        return render_template("index.html", 
                               tables=[results_df.to_html(classes='data', header=True)],
                               download_link=True, 
                               expected_feature_count=expected_feature_count,
                               input_type="file",
                               filename=filename)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    try:
        form_data = request.form
        features = [float(form_data.get(str(i), 0)) for i in range(expected_feature_count)]

        user_input = np.array(features).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)
        user_input_reshaped = user_input_scaled.reshape((1, 1, expected_feature_count))

        anomaly_score, anomaly_status = detect_anomaly(user_input_reshaped)
        attack_category = classify_attack(user_input_scaled, anomaly_status)[0]

        # Create a DataFrame for display
        display_data = {}
        for i in range(min(5, expected_feature_count)):  # Show only first 5 features to save space
            display_data[f"Feature_{i+1}"] = features[i]
            
        display_data["Anomaly Score"] = f"{anomaly_score[0]:.6f}"
        display_data["Anomaly Detected"] = anomaly_status[0]
        display_data["Predicted Attack Category"] = attack_category

        results_df = pd.DataFrame([display_data])

        return render_template("index.html", 
                               tables=[results_df.to_html(classes='data', header=True)],
                               expected_feature_count=expected_feature_count,
                               input_type="form",
                               request=request)  # Pass the request object to the template

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download_predictions")
def download_predictions():
    try:
        output_file = os.path.join(app.static_folder, "predictions.csv")
        return send_file(output_file, as_attachment=True, download_name="predictions.csv")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Create static folder if it doesn't exist
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")):
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"))
    
    app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    app.run(debug=True)

