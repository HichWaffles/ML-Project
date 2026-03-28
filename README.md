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

## Segmentation Strategy

Instead of relying on rigid, monolithic frameworks like RFM—which fail to capture the nuances of modern e-commerce behavior and heavily correlate with churn—this project uses a **Multi-Theme Segmentation Architecture**. 

Customers are scored and clustered independently across three distinct behavioral axes using K-Means clustering, providing a rich, multi-dimensional view of the customer base.

### 1. Friction (Operational Overhead)
Focuses on the cost-to-serve and operational friction of the customer:
- **Low-Support**: The silent majority. Normal baseline friction but high silent churn.
- **High-Support**: Engaged users asking for help via tickets. Better retention.
- **Serial-Canceller**: Power buyers who habitually cancel and reorder. 
- **High-Returner**: Customers suffering from product-fit failure.

### 2. Explorer (Catalog Breadth)
Focuses on how diversely the customer interacts with the product catalog:
- **Focused-Buyer**: Purchases a narrow, repetitive band of products.
- **Broad-Explorer**: Explores deep into the catalog across many categories.

### 3. Timing (Temporal Cadence)
Focuses on when the customer prefers to shop:
- **Weekday-Shopper**: Activity concentrated during the workweek.
- **Weekend-Shopper**: Major activity peaks on weekends.

### Notebooks & Production
- **`notebooks/segmentation_exploration.ipynb`**: Contains the robust feature engineering (IQR clipping), K-Means clustering, silhouette scoring, and churn-correlation analysis used to validate the three themes.
- **`src/segment_customers.py`**: The production script that executes the multi-theme pipeline, appending independent cluster assignments to the user base and generating detailed profiles and insights in `data/segments/`.

