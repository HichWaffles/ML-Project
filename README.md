# Retail Customer Behavioral Analysis

## Project Overview

This project focuses on analyzing customer behavior for an e-commerce gift company. The objective is to leverage a dataset of 52 features to personalize marketing strategies, reduce customer churn, and optimize overall revenue.

## Project Structure

The project follows a standard Machine Learning pipeline architecture:

- **data/**: Contains raw, processed, and split (train/test) datasets
- **notebooks/**: Jupyter notebooks used for prototyping and initial exploration
- **src/**: Production-ready Python scripts for preprocessing, training, and prediction
- **models/**: Saved model files
- **app/**: Flask-based web application for model deployment
- **reports/**: Visualizations and analysis reports

## Machine Learning Pipeline

The project covers the complete data science lifecycle:

- **Exploration**: Analyzing data quality and structure
- **Preparation**: Cleaning, encoding categorical variables, and handling missing values
- **Transformation**: Dimensionality reduction using Principal Component Analysis (PCA) to reduce noise and accelerate calculations
- **Modeling**: Implementation of clustering, classification (for Churn), and regression models
- **Deployment**: A web interface built with Flask to serve the model
