import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.subheader("Model Interpretability with SHAP")

# Sample data (replace with actual data)
X_sample = np.random.randn(30, 10)  # Replace with your actual features
y_sample = np.random.randint(0, 2, size=30)  # Replace with your actual labels

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_sample, y_sample)

# SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

if st.button('Visualize SHAP Values'):
    st.write("### SHAP Force Plot for Sample 13")
    shap.initjs()
    sample_index = 13  # Replace with a slider if you want to dynamically select
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][sample_index], X_sample[sample_index]))

    st.write("### Global Feature Importance")
    fig, ax = plt.subplots(facecolor='#1f1f2e')
    ax.set_facecolor('#1f1f2e')
    shap.summary_plot(shap_values[1], X_sample, plot_type="bar", plot_size=(10, 6))
    st.pyplot(fig)

    st.write("### SHAP Waterfall Plot for Detailed Explanation")
    fig, ax = plt.subplots(facecolor='#1f1f2e')
    ax.set_facecolor('#1f1f2e')
    shap.waterfall_plot(shap.Explanation(values=shap_values[1][sample_index], base_values=explainer.expected_value[1], data=X_sample[sample_index], feature_names=[f'Feature {i}' for i in range(X_sample.shape[1])]))
    st.pyplot(fig)

# Function to display SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# Optional: Add a dropdown or slider for sample selection
selected_sample = st.slider("Select Sample Index for SHAP Analysis", 0, 29, 13)
