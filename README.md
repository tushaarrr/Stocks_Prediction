# Nandini_300377653_4260_002_A1
 Stock Price Prediction & Benchmarking Dashboard

Project Overview

This project benchmarks various data storage and processing techniques while predicting stock prices using Machine Learning models. It includes:

Benchmarking of CSV vs. Parquet for data storage efficiency

Comparison of Pandas vs. Polars for data processing performance

Stock price prediction using ML models (Gradient Boosting Regressor)

An interactive Streamlit dashboard for visualization and predictions

 How to Set Up and Run the Project

1️⃣ Set Up a Virtual Environment

Run the following commands in your project directory:

python -m venv .venv

2️⃣ Activate the Virtual Environment

Windows (PowerShell):

.venv\Scripts\Activate

Mac/Linux:

source .venv/bin/activate

3️⃣ Install Required Dependencies

pip install -r requirements.txt

4️⃣ Run the Streamlit Dashboard

streamlit run streamlit.py

This will launch the interactive dashboard in your web browser.

 Benchmarking Analysis & Results

1️⃣ Storage Benchmarking: CSV vs. Parquet

CSV:

Simple and widely used but lacks compression.

Slower for large-scale data operations.

Consumes more storage due to text-based structure.

Parquet:

Columnar storage format optimized for analytical workloads.

Highly compressed and much faster for reading/writing.

Ideal for scaling beyond 10x/100x dataset sizes.

Benchmark Results:

At 1x scale, both CSV and Parquet perform similarly.

At 10x and 100x, Parquet is significantly faster and more efficient.

Decision: Parquet was chosen for large-scale processing, while CSV is maintained for compatibility.

2️⃣ Data Processing Benchmarking: Pandas vs. Polars

Pandas:

Most widely used library in Python.

Good for small-to-medium datasets but struggles with very large datasets.

Chosen for model training and feature engineering due to familiarity.

Polars:

Optimized for parallel execution, making it faster on large datasets.

Performs significantly better at 10x and 100x dataset sizes.

Benchmark Results:

Pandas was used for modeling due to better integration with scikit-learn.

Polars was used for fast large-scale data manipulation before ML training.

3️⃣ Machine Learning: Model Selection

Gradient Boosting Regressor (GBR) was chosen due to:

Better predictive power than linear models.

Handles non-linearity well and reduces overfitting.

Performs well with structured stock data.

Feature Scaling:

StandardScaler was used to normalize stock price data for better model convergence.

Training & Prediction Pipeline:

Pandas was used for feature extraction and preprocessing.

GBR was trained using historical stock prices (Open, High, Low, Volume → Close Price prediction).

4️⃣ Dashboarding: Streamlit

Streamlit was selected because:

Simple and easy to use

Provides an interactive UI for benchmarking and predictions

Integrates seamlessly with Pandas and Polars

 project_directory/
│-- .venv/                 # Virtual environment
│-- requirements.txt       # Dependencies
│-- streamlit.py           # Streamlit dashboard
│-- Assignmnet_1.py        # Benchmarking & ML model training
│-- trained_model.pkl      # Saved ML model
│-- scaler.pkl             # Feature scaler
│-- benchmark_results.csv  # Benchmarking results
│-- NandiniG_screenshots.pdf  # Screenshots of dashboard

