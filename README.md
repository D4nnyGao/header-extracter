# Clinical Trial Information Extraction Project

This project focuses on extracting structured information from clinical trial documents. It implements and compares two distinct machine learning approaches: a traditional **Random Forest (RF) classifier** and a modern **Large Language Model (LLM)**. The goal is to efficiently identify and extract key data points from raw text data sourced from ClinicalTrials.gov.

The repository includes data processing pipelines, model training scripts, evaluation scripts, and interactive web applications built with Streamlit to demonstrate and test the models.

***

## 🚀 Features

* **Data Pipeline:** A script to preprocess and clean the raw clinical trial data (`ctg-studies.csv`) for machine learning applications.
* **Random Forest Model:** A classical ML model trained to classify or extract specific features. The repository includes a visualization of the resulting feature importances.
* **Large Language Model (LLM):** An LLM-based approach for the same information extraction task, complete with a performance evaluation script.
* **Interactive Demos:** Separate Streamlit applications to demonstrate the live prediction capabilities of both the Random Forest (`rf_app.py`) and LLM (`llm_app.py`) models.
* **Manual Testing App:** A dedicated Streamlit interface (`test_app.py`) for easy manual testing and validation of outputs.

***

## 📂 File Structure

Here is an overview of the key files in this project: