# FlipItNews Classifier

An NLP-based news classification project for **FlipItNews** (Indian startup) to automate categorization of news articles and support better content organization, search, and personalized delivery.

## Problem Statement

FlipItNews needs to classify large volumes of news articles automatically instead of relying on time-consuming manual labeling.  
This project builds and compares text-classification pipelines and deep learning models to predict article categories from news text.

## Dataset

- Source: [Google Drive Dataset Link](https://drive.google.com/uc?id=1POmzYEQZgZVqJO3F_cyNcd4QmoNE_mCv)
- Expected content: news article text and category labels

## Project Objectives

1. Perform **Exploratory Data Analysis (EDA)**
   - Inspect dataset structure
   - Identify and handle missing values
   - Visualize class/category distribution

2. Perform **Text Preprocessing**
   - Convert text to lowercase
   - Remove punctuation
   - Remove stop words
   - Tokenize text

3. Build **Feature Representations**
   - Bag of Words (BoW)
   - TF-IDF
   - Word2Vec

4. Train and evaluate **Classification Models**
   - Neural Network (NN)
   - Recurrent Neural Network (RNN)

5. Compare model performance using:
   - Accuracy
   - Precision
   - Recall
   - F1-score

6. Deploy the trained **RNN model** via an interactive web-based dashboard for automated news category prediction.

## Suggested Workflow

1. Load dataset and inspect schema
2. Run EDA plots and missing-value analysis
3. Build preprocessing pipeline
4. Train feature extractors (BoW / TF-IDF / Word2Vec)
5. Train NN and RNN models
6. Evaluate and compare metrics
7. Select best model (target: RNN deployment)
8. Integrate model into dashboard UI for inference

## Expected Impact for FlipItNews

- Faster and consistent content categorization
- Improved search and filtering of published content
- Better personalization through reliable category tagging
- Reduced manual operational effort in editorial workflows

## Deliverables

- EDA summary and visualizations
- Cleaned/tokenized text pipeline
- BoW, TF-IDF, and Word2Vec feature experiments
- NN and RNN training/evaluation reports
- Final model comparison table with Accuracy, Precision, Recall, and F1-score
- Interactive web dashboard hosting the trained RNN for real-time classification
