# Codsoft

## Overview

This repository contains three machine learning projects:
1. **Movie Genre Classification**: A project that classifies movies into different genres based on their descriptions.
2. **Credit Card Fraud Detection**: A project that detects fraudulent credit card transactions.
3. **Customer Churn Prediction**: A project that predicts customer churn for a bank based on customer data.

## Project 1: Movie Genre Classification

### Description

The goal of this project is to classify movies into their respective genres based on their descriptions. Various machine learning models are implemented and evaluated to determine the best performing model.

### Dataset

- **Training Data**: `dataset/train_data.txt`
- **Test Data**: `dataset/test_data.txt`

### Steps

1. **Data Cleaning**
2. **Data Preprocessing**
3. **Exploratory Data Analysis (EDA)**
4. **TF-IDF Vectorization**
5. **Model Training and Evaluation**

### Models Used

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

### Results

The accuracy of the models is compared, and confusion matrices are plotted to visualize the performance of each model.

## Project 2: Credit Card Fraud Detection

### Description

The goal of this project is to detect fraudulent credit card transactions. Several machine learning models are implemented and evaluated to identify the best performing model for this task.

### Dataset

- **Training Data**: `dataset/fraudTrain.csv`

### Steps

1. **Data Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Data Preparation**
4. **Feature Engineering**
5. **Label Encoding and Scaling**
6. **Model Training and Evaluation**

### Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Neural Networks (MLPClassifier)
- Naive Bayes

### Results

The models are evaluated based on accuracy, precision, recall, and F1 score. Confusion matrices are plotted to visualize the performance of each model.

## Project 3: Customer Churn Prediction

### Description

The goal of this project is to predict customer churn for a bank based on customer data. Various machine learning models are implemented and evaluated to determine the best performing model.

### Dataset

- **Data**: `dataset/Churn_Modelling.csv`

### Steps

1. **Data Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Data Preparation**
4. **Feature Engineering**
5. **Label Encoding and Scaling**
6. **Model Training and Evaluation**

### Models Used

- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Random Forest Classifier

### Results

The models are evaluated based on accuracy, and confusion matrices are plotted to visualize the performance of each model.

#### Random Forest Classifier Results:

- **Accuracy on training data**: High
- **Accuracy on validation data**: High
- **Accuracy on test data**: High

Confusion Matrix and ROC Curve are plotted to further analyze the model performance.
