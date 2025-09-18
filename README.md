@"
# Bike Counters Project

This repository contains a complete data science project on the **Bike Counters dataset**.
It includes data preprocessing, exploratory analysis, modeling, a **Streamlit app**, automated tests, and a Docker container.

---

## Kaggle Challenge

Read the instructions from the [Kaggle challenge](https://www.kaggle.com/competitions/mdsb-2023/overview).

Your goal is to train your model on `train.parquet` (and optional external datasets) and make predictions on `final_test.parquet`.

---

## Dataset Setup

1. Download the datasets from Kaggle:
   - `train.parquet`
   - `final_test.parquet`
   - Optional external data in `external_data/`
2. Place the files into the `data/` folder.
3. Ensure the folder structure is like:

\```
project/
├── data/
│   ├── train.parquet
│   ├── final_test.parquet
│   └── external_data.csv
├── app.py
├── requirements.txt
├── tests/
└── Dockerfile
\```

---

## Local Environment Setup

It is recommended to use a virtual environment:

\```bash
conda create -n bikes-count python=3.11
conda activate bikes-count
\```

Install dependencies:

\```bash
pip install -r requirements.txt
\```

---

## Running the Streamlit App

To launch the interactive app locally:

\```bash
streamlit run app.py
\```

The app includes:

- Raw data preview
- Interactive Pydeck map of bike locations
- Histograms and correlation heatmaps of numerical features
- Time-series visualization of bike counts
- Weather features over time

---

## Docker Usage

You can run the Streamlit app in a Docker container:

\```bash
# Build the Docker image
docker build -t bike-counter-app .

# Run the container
docker run -p 8501:8501 bike-counter-app
\```

Open your browser at [http://localhost:8501](http://localhost:8501) to access the app.

(Optional) Push the Docker image to DockerHub:

\```bash
docker tag bike-counter-app yourusername/bike-counter-app:latest
docker push yourusername/bike-counter-app:latest
\```

---

## Testing

Automated tests are included in the `tests/` folder:

\```bash
pytest -v
\```

Tests currently cover:

- Data loading functions (`load_data()`, `load_weather_data()`)
- Column existence and data integrity checks

---

## Continuous Integration (CI)

A GitHub Actions workflow is included (`.github/workflows/ci.yml`) that:

- Runs all tests on push or pull requests
- Lints code using `black`
- Builds the Docker image to ensure containerization works

All CI checks must pass before submission.

---

## Kaggle Submission

- Submit your prediction script `.py` via the Kaggle interface.
- The submission CSV must include columns: `"Id"` and `"bike_log_count"`.
- Ensure the length matches `final_test.parquet`.

---

## Notebooks

The starter notebook `bike_counters_starting_kit.ipynb` demonstrates:

- Data loading and preprocessing
- Exploratory visualizations
- Example model training

Launch it using:

\```bash
jupyter lab bike_counters_starting_kit.ipynb
\```
