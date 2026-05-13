# Anomaly Detection in IoT Time-Series Data

A beginner-friendly Flask web application for detecting anomalies in IoT-style time-series or tabular network-flow features and for classifying potential attack categories. Upload a CSV or enter feature values manually; the app combines a deep learning autoencoder with gradient boosting for a practical ML pipeline you can run locally or deploy.

---

## Project Description

This project exposes a trained **LSTM autoencoder** (TensorFlow/Keras) for reconstruction-based anomaly scoring and a **LightGBM** multiclass model for attack-type prediction when a row is flagged as anomalous. The web UI is built with **HTML**, **CSS**, and **JavaScript**, served by **Flask**. If model files are missing, the server starts with a clear warning and uses a safe development fallback so you can still explore the interface.

---

## Features

- **CSV upload**: Batch scoring with results table and optional full-result download (`predictions.csv` in the static folder).
- **Manual input**: Enter one row of numeric features and view scores and predicted attack category.
- **REST-style JSON errors**: Invalid uploads or dimension mismatches return JSON error messages with appropriate HTTP status codes.
- **Clear separation of concerns**: Models under `models/`, notebooks under `notebook/`, UI assets under `static/` and `templates/`.

---

## Technologies Used

| Area        | Technology                          |
|------------|--------------------------------------|
| Language   | Python                               |
| Web        | Flask, Werkzeug                      |
| Deep learning | TensorFlow, Keras               |
| ML / classic | LightGBM, scikit-learn, pandas, numpy |
| Frontend   | HTML, CSS, JavaScript                |

---

## Machine Learning Models Used

1. **LSTM autoencoder** (`models/lstm_autoencoder.h5`) — Encodes and reconstructs sequences; anomaly score is derived from reconstruction error (MSE-style loss on the backend).
2. **LightGBM multiclass booster** (`models/lightgbm_model.txt`) — Predicts attack category for rows classified as anomalies.

Place these files in the `models/` directory (see **Installation Steps**). Expected input width is inferred from the loaded LSTM model when available.

---

## Folder Structure

```text
Anamoly_detection/
├── .vscode/              # VS Code launch/settings (optional)
├── .gitignore            # Python / venv / IDE ignores (mirrors repo root if used alone)
├── assets/               # Extra static branding, diagrams, or screenshots (optional)
├── models/               # Trained model files (.h5, .txt)
├── notebook/             # Jupyter notebooks for training and experiments
├── static/
│   ├── css/              # Stylesheets
│   ├── js/               # Client-side scripts
│   ├── images/           # Image assets for the UI
│   └── predictions.csv   # Generated after file upload (download source)
├── templates/
│   ├── index.html        # Main analysis page
│   └── result.html       # Helper page explaining where results appear
├── cv.csv                # Sample or cross-validation related data (as provided)
├── predictions.csv       # Last or sample predictions export (as provided)
├── requirements.txt      # Python dependencies
├── server.py             # Flask application entry point
└── README.md             # This file
```

If the Git repository root is the parent folder (`anamoly mulyi`), a `.gitignore` there continues to apply to the whole tree (including `.idea/` at the top level).

---

## Installation Steps

1. **Clone the repository** (or copy this project folder).

2. **Create a virtual environment** (recommended; do not commit `.venv` to Git):

   ```bash
   python -m venv .venv
   ```

3. **Activate the environment**

   - Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
   - macOS / Linux: `source .venv/bin/activate`

4. **Install dependencies**

   ```bash
   cd Anamoly_detection
   pip install -r requirements.txt
   ```

5. **Add trained models** (for full ML behavior):

   - `models/lstm_autoencoder.h5`
   - `models/lightgbm_model.txt`

   Without these files, the app still runs using the built-in development fallback described in `server.py`.

---

## How to Run the Project

From the `Anamoly_detection` directory:

```bash
python server.py
```

Then open a browser at **http://127.0.0.1:5000/** (Flask default with `debug=True` in `server.py`).

- **Templates**: Flask resolves `templates/index.html` and `templates/result.html` via `render_template`.
- **Static files**: URLs such as `/static/css/styles.css` and `/static/js/main.js` are served from the `static/` folder automatically.

For the best experience with the included VS Code **Launch Chrome** configuration, open the **`Anamoly_detection`** folder as your workspace root so `webRoot` matches the Flask app location.

---

## Screenshots

Add your UI screenshots under `assets/` or `static/images/` and embed them here, for example:

```markdown
![Home page](assets/screenshot-home.png)
![Results table](assets/screenshot-results.png)
```

*(Replace with your own image paths after you capture the screens.)*

---

## Future Improvements

- Add authentication and rate limiting for a public deployment.
- Persist prediction history in a database instead of only CSV download.
- Add automated tests (pytest) and a CI workflow (GitHub Actions).
- Expose a versioned JSON API (e.g. `/api/v1/predict`) for integrations.
- Dockerize the app with a pinned base image for reproducible deployment.

---

## Author

**Pavani**

---

## Best practices (quick reference)

- Keep **secrets and API keys** in `.env` (listed in `.gitignore`), never in source code.
- Use a **virtual environment** per project; pin dependencies in `requirements.txt` for reproducibility.
- For production, run with a production WSGI server (e.g. **Gunicorn** or **Waitress**) and turn **debug mode off** in `server.py`.
- Track **large model files** with Git LFS or release artifacts instead of bloating the main branch if they exceed GitHub limits.
