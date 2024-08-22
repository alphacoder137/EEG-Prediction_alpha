import streamlit as st
import pandas as pd
import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score, StratifiedKFold, permutation_test_score
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import butter, filtfilt, welch
import shap
from lifelines import KaplanMeierFitter, CoxPHFitter
import requests
import zipfile
import logging

# Initialize the logger
logging.basicConfig(level=logging.INFO)

st.title("Epi-Sense Visualization with Advanced Statistical Analysis")

# Function to download and extract the zip file from Google Drive
def download_and_extract_zip(url, zip_filename, extract_to):
    logging.info("Downloading the dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logging.info("Extracting the dataset...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_filename)

# Load the EEG data from the extracted folder
def load_eeg_data(base_dir):
    categories = ['preictal', 'interictal', 'ictal']
    labels = {'preictal': 0, 'interictal': 1, 'ictal': 2}
    X = []  # Raw Data Matrix
    y = []  # Label vector
    logging.info(f"Checking base directory: {base_dir}")
    for category in categories:
        cat_dir = None
        for subdir, _, _ in os.walk(base_dir):
            if os.path.basename(subdir) == category:
                cat_dir = subdir
                break
        logging.info(f"Checking category directory: {cat_dir}")
        if cat_dir and os.path.exists(cat_dir):
            logging.info(f"Directory exists: {cat_dir}")
            for file in os.listdir(cat_dir):
                logging.info(f"Processing file: {file}")
                file_path = os.path.join(cat_dir, file)
                if file.endswith('.mat'):
                    mat_data = scipy.io.loadmat(file_path)
                    data = mat_data.get(category)
                    if data is None:
                        logging.error(f"Data not found in {file_path}")
                    else:
                        X.append(data.flatten())  # Flatten the EEG segment
                        y.append(labels[category])
        else:
            logging.error(f"Directory {cat_dir} does not exist!")
    return np.array(X), np.array(y)

# Download and extract dataset
zip_url = "https://drive.google.com/uc?export=download&id=1Y0Cw2emtNxQX0Ei47rbR33Da9Yeqt39L"
zip_filename = "dataset.zip"
extracted_folder = "EEG_Epilepsy_Datasets"
if not os.path.exists(extracted_folder):
    download_and_extract_zip(zip_url, zip_filename, extracted_folder)

# Load the EEG data
X, y = load_eeg_data(extracted_folder)

if X.size == 0 or y.size == 0:
    st.error("No EEG data found. Please check if the files were downloaded correctly.")

# Load y_test
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

def load_results(file_path):
    return pd.read_pickle(file_path)

def calculate_and_display_fusion_prediction(index, results_dfs):
    class_labels = {0: 'Preictal', 1: 'Interictal', 2: 'Ictal'}
    cumulative_probabilities = [0] * len(class_labels)
    
    for df in results_dfs.values():
        probabilities = df.iloc[index]['Probabilities']
        cumulative_probabilities = [sum(x) for x in zip(cumulative_probabilities, probabilities)]
    
    averaged_probabilities = [prob / len(results_dfs) for prob in cumulative_probabilities]
    final_predicted_class_index = averaged_probabilities.index(max(averaged_probabilities))
    return final_predicted_class_index, averaged_probabilities

def display_fusion_prediction(index, results_dfs):
    predicted_class_index, averaged_probabilities = calculate_and_display_fusion_prediction(index, results_dfs)
    
    class_labels = {0: 'Preictal', 1: 'Interictal', 2: 'Ictal'}
    correct_label = class_labels[y_test[index]]
    predicted_class = class_labels[predicted_class_index]
    averaged_probabilities_percent = [prob * 100 for prob in averaged_probabilities]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div style='font-size: 20px; font-family: serif; color: white;'>Correct Label: <strong>{correct_label}</strong></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='font-size: 20px; font-family: serif; color: white;'>Fusion Prediction: <strong>{predicted_class}</strong></div>", unsafe_allow_html=True)
    with col3:
        fusion_predictions = [calculate_and_display_fusion_prediction(i, results_dfs)[0] for i in range(30)]
        accuracy = accuracy_score(y_test[:30], fusion_predictions)
        st.markdown(f"<div style='font-size: 20px; font-family: serif; color: white;'>Fusion Accuracy: {accuracy * 100:.2f}%</div>", unsafe_allow_html=True)
    
    st.title("")  # Add a blank title for vertical space
    
    prob_df = pd.DataFrame({
        'Class': list(class_labels.values()),
        'Probability (%)': averaged_probabilities_percent
    })
    
    fig, ax = plt.subplots(facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    sns.barplot(x='Probability (%)', y='Class', data=prob_df, palette='coolwarm', ax=ax)
    ax.set_ylabel('')
    ax.set_title('Fusion Prediction Probabilities', color='white', fontname='serif')
    ax.set_xlabel('Probability (%)', color='white', fontname='serif')
    ax.set_yticklabels(prob_df['Class'], color='white', fontname='serif')
    ax.tick_params(colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    st.pyplot(fig)

# Load data from all models
models = ['rf', 'ada']
results_dfs = {model: load_results(f"{model}_pred.pkl") for model in models}

font_properties = {'fontname': 'serif', 'fontsize': 14, 'color': 'white'}

# Initialize session state
if 'show_fusion' not in st.session_state:
    st.session_state['show_fusion'] = False

if 'X' in globals() and 'y' in globals():
    sample_index = st.slider('Select Test Sample Index', 0, 29, 0)
    
    if st.button('Show EEG Sample & Fusion Prediction'):
        st.session_state['show_fusion'] = True

    if st.session_state['show_fusion']:
        fig, ax = plt.subplots(facecolor='#0D1117')
        ax.set_facecolor('#0D1117')
        ax.plot(X[sample_index], color='yellow')
        ax.set_title(f"EEG Recording: {sample_index}", fontdict=font_properties)
        ax.set_xlabel("Datapoint (0-1024)", fontdict=font_properties)
        ax.set_ylabel("Voltage", fontdict=font_properties)
        ax.tick_params(colors='white')
        st.pyplot(fig)
        
        st.title("")  # Add a blank title for vertical space

        display_fusion_prediction(sample_index, results_dfs)
else:
    st.write("Please provide the correct path to the EEG dataset.")

# ---- EDA and Statistical Analysis Section ----
st.header("Exploratory Data Analysis (EDA)")
def stationarity_tests(data):
    adf_result = adfuller(data)
    st.write("ADF Statistic:", adf_result[0])
    st.write("p-value:", adf_result[1])
    st.write("Critical Values:", adf_result[4])

    kpss_result = kpss(data, regression='c')
    st.write("\nKPSS Statistic:", kpss_result[0])
    st.write("p-value:", kpss_result[1])
    st.write("Critical Values:", kpss_result[3])

def frequency_domain_analysis(data, fs=256):
    freqs, psd = welch(data, fs=fs)
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, psd, color='blue')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    st.pyplot(plt)

def extract_statistical_features(data):
    mean_val = np.mean(data)
    variance_val = np.var(data)
    st.write(f"Mean: {mean_val}")
    st.write(f"Variance: {variance_val}")

sample_index = st.slider('Select Test Sample Index for EDA', 0, 29, 0)
if st.button('Perform EDA'):
    st.subheader("Time-Series Analysis and Stationarity Tests")
    stationarity_tests(X[sample_index])

    st.subheader("Frequency Domain Analysis")
    frequency_domain_analysis(X[sample_index])

    st.subheader("Statistical Feature Extraction")
    extract_statistical_features(X[sample_index])

# ---- Signal Processing Section ----
st.subheader("Signal Processing and Noise Reduction")
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=256, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

if st.button('Apply Band-Pass Filter'):
    filtered_data = bandpass_filter(X[sample_index])
    st.write("Filtered Data (First 10 Points):", filtered_data[:10])

    fig, ax = plt.subplots(facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    ax.plot(filtered_data, color='cyan')
    ax.set_title(f"Filtered EEG Recording: {sample_index}", fontdict=font_properties)
    ax.set_xlabel("Datapoint (0-1024)", fontdict=font_properties)
    ax.set_ylabel("Voltage", fontdict=font_properties)
    ax.tick_params(colors='white')
    st.pyplot(fig)

# ---- Bayesian Inference Section ----
st.subheader("Bayesian Inference and Probabilistic Models")
def apply_bayesian_inference(X, y):
    model = BayesianRidge()
    model.fit(X, y)
    return model

if st.button('Apply Bayesian Inference'):
    bayesian_model = apply_bayesian_inference(X, y)
    prediction = bayesian_model.predict([X[sample_index]])
    st.write(f"Bayesian Prediction for Sample {sample_index}: {prediction[0]}")

# ---- Survival Analysis Section ----
st.subheader("Survival Analysis")
seizure_times = np.random.exponential(scale=100, size=len(y))
event_occurred = (y == 2).astype(int)

def perform_kaplan_meier_analysis(seizure_times, event_occurred):
    kmf = KaplanMeierFitter()
    kmf.fit(seizure_times, event_occurred)
    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    st.pyplot(fig)

if st.button('Perform Kaplan-Meier Analysis'):
    perform_kaplan_meier_analysis(seizure_times, event_occurred)

# ---- Model Interpretability with SHAP ----
st.subheader("Model Interpretability with SHAP")
rf_model = RandomForestClassifier()
rf_model.fit(X, y)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

if st.button('Visualize SHAP Values'):
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1][sample_index], X[sample_index], matplotlib=True)
    st.pyplot()

# ---- Cross-Validation and Permutation Testing ----
st.subheader("Model Evaluation and Statistical Validation")
def perform_cross_validation(X, y):
    cv = StratifiedKFold(n_splits=5)
    model = RandomForestClassifier()
    cv_scores = cross_val_score(model, X, y, cv=cv)
    st.write(f"Cross-Validation Scores: {cv_scores}")
    st.write(f"Mean CV Score: {np.mean(cv_scores)}")

def perform_permutation_test(X, y):
    model = RandomForestClassifier()
    score, permutation_scores, pvalue = permutation_test_score(model, X, y, cv=5, n_permutations=100)
    st.write(f"Permutation Test Score: {score}")
    st.write(f"Permutation Test p-value: {pvalue}")

if st.button('Perform Cross-Validation'):
    perform_cross_validation(X, y)

if st.button('Perform Permutation Test'):
    perform_permutation_test(X, y)
