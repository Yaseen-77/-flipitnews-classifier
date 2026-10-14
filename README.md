# FlipItNews Classifier

An end-to-end Natural Language Processing (NLP) project for automatically categorizing Indian news articles. This project is designed to help FlipItNews organize content more efficiently, improve search and discovery, and support personalized news delivery.

## Problem Statement

FlipItNews publishes a large volume of news content across multiple categories. Manually labeling every article is time-consuming, difficult to scale, and vulnerable to inconsistent categorization.

This project builds a text-classification pipeline that learns from previously labeled news articles and predicts the most appropriate category for new article text. The solution covers exploratory data analysis, text preprocessing, feature engineering, neural-network modeling, model evaluation, and interactive deployment.

## Objectives

- Explore the dataset and understand its structure and category distribution.
- Identify missing values, duplicates, and potential data-quality issues.
- Clean and normalize article text for machine-learning workflows.
- Compare Bag of Words, TF-IDF, and Word2Vec representations.
- Develop Neural Network (NN) and Recurrent Neural Network (RNN) classifiers.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Deploy the trained RNN model through an interactive web dashboard.
- Demonstrate how automated categorization can improve FlipItNews content operations.

## Dataset

The dataset is available at:

[Download the FlipItNews dataset](https://drive.google.com/uc?id=1POmzYEQZgZVqJO3F_cyNcd4QmoNE_mCv)

The dataset contains news articles and their corresponding category labels. Before modeling, the data should be inspected for missing values, duplicate records, class imbalance, and inconsistencies in the text or label columns.

> **Note:** Downloaded datasets and trained model artifacts are not included in this repository unless explicitly added later. Update the dataset path in the notebook or application configuration before running the project.

## Project Workflow

```text
Raw news data
      ↓
Exploratory Data Analysis
      ↓
Text cleaning and preprocessing
      ↓
Feature extraction
      ├── Bag of Words
      ├── TF-IDF
      └── Word2Vec
      ↓
Model training
      ├── Neural Network
      └── Recurrent Neural Network
      ↓
Evaluation and comparison
      ↓
Interactive news-category prediction dashboard
```

## Text Preprocessing

The preprocessing pipeline includes:

1. Converting text to lowercase.
2. Removing punctuation and unwanted characters.
3. Handling missing or empty text values.
4. Removing stop words.
5. Tokenizing article text.
6. Preparing sequences and padding them for the RNN model.

## Feature Engineering

The project explores multiple numerical representations of text:

### Bag of Words

Represents each article using word-frequency counts. This provides a simple baseline for comparing traditional machine-learning or neural-network approaches.

### TF-IDF

Weights words according to how important they are within an article and across the complete dataset. Common words receive lower importance, while category-relevant terms receive higher weights.

### Word2Vec

Creates dense word embeddings that capture relationships between words based on their surrounding context. These embeddings can be used as inputs to neural and recurrent models.

## Models

### Neural Network

A feed-forward Neural Network is trained on numerical text features to establish a strong baseline for multi-class news classification.

### Recurrent Neural Network

The RNN processes token sequences while retaining information from earlier words in an article. This makes it suitable for learning patterns in the order and context of news text.

The RNN is also used as the primary model for the interactive deployment dashboard.

## Evaluation

Models are compared using:

- **Accuracy:** Overall proportion of correctly classified articles.
- **Precision:** Proportion of predicted articles in a category that are actually from that category.
- **Recall:** Proportion of articles from a category that are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Category-level view of correct and incorrect predictions.

Use the evaluation results to select the model that provides the best balance across categories rather than relying on accuracy alone, especially if the dataset is imbalanced.

## Interactive Dashboard

The deployment component is intended to provide a simple web interface where a user can:

1. Enter or paste a news article.
2. Submit the text for prediction.
3. View the predicted news category.
4. Review the model confidence or class probabilities when available.

The dashboard can be implemented with a lightweight Python web framework such as Streamlit. The trained tokenizer, label encoder, and RNN model should be loaded when the application starts.

## Business Impact

Automated news categorization can help FlipItNews:

- Reduce the manual effort required to label articles.
- Improve consistency across editorial categories.
- Organize content for faster publishing and retrieval.
- Improve search filters and category-based navigation.
- Support personalized recommendations and content feeds.
- Scale content operations as article volume increases.
- Provide a foundation for future recommendation and analytics systems.

Human review should remain available for low-confidence predictions and newly emerging topics that may not be well represented in the training data.

## Suggested Repository Structure

```text
.
├── README.md
├── notebooks/
│   └── flipitnews_classifier.ipynb
├── data/
│   └── README.md
├── models/
│   ├── rnn_model.*
│   ├── tokenizer.*
│   └── label_encoder.*
├── app.py
├── requirements.txt
└── .gitignore
```

The current repository is primarily notebook-based. The structure above is a recommended organization for extending the project into a reproducible training and deployment application.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Yaseen-77/-flipitnews-classifier.git
cd -flipitnews-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

For notebook execution, the project will typically require packages such as:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk gensim tensorflow jupyter streamlit
```

### 4. Download and configure the dataset

Download the dataset from the link in the [Dataset](#dataset) section and update the dataset path used by the notebook or application.

### 5. Run the notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells in order to perform EDA, preprocessing, feature extraction, model training, and evaluation.

### 6. Run the dashboard

After exporting the trained RNN model and preprocessing artifacts, start the dashboard with:

```bash
streamlit run app.py
```

## Reproducibility Notes

- Use the same preprocessing steps during training and inference.
- Save and reload the tokenizer and label encoder used by the trained model.
- Use a fixed random seed where possible.
- Keep training, validation, and test data separate.
- Track the class labels and model version used for each prediction.
- Do not commit large datasets, credentials, or private deployment configuration.

## Limitations and Future Improvements

Potential next steps include:

- Hyperparameter tuning for the NN and RNN models.
- Addressing class imbalance through class weights or resampling.
- Comparing the RNN with LSTM, GRU, and transformer-based models.
- Adding confidence thresholds and a human-review workflow.
- Monitoring model performance after deployment.
- Adding multilingual support for Indian languages.
- Supporting batch classification for multiple articles.
- Adding prediction explanations using important words or attention scores.

## License

No license has been specified for this repository yet. Add a license before distributing or reusing the project publicly.

## Author

Developed as a FlipItNews NLP classification lab implementation by [Yaseen-77](https://github.com/Yaseen-77).
