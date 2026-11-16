::: {align="center"}
# ✈️ Airline Tweet Sentiment Analyzer

### Natural Language Processing \| Text Classification \| Machine Learning \| Streamlit

**A practical NLP project that classifies airline-related tweets as
Positive, Negative, or Neutral.**
:::

------------------------------------------------------------------------

## 📌 Project at a Glance

  Item                                        Details
  ------------------------------------------- ----------------------------------------------------
  Project                                     Airline Tweet Sentiment Analyzer
  Practical                                   NLP Practical Report 1
  Dataset                                     Twitter US Airline Sentiment
  Original records                            14,640 tweets
  Records after duplicate removal             14,427 tweets
  Sentiment classes                           Negative, Neutral, Positive
  Best model                                  TF-IDF Unigrams + Bigrams with Logistic Regression
  Test Accuracy                               **79.63%**
  Test Macro-F1                               **0.7510**
  Scikit-learn version recorded in notebook   **1.9.0**
  Main notebook                               `NLP_PR1.ipynb`

> **Deployment status:** The notebook trains and saves
> `sentiment_model.joblib` and tests it on sample sentences. It contains
> the Streamlit app plan, but the uploaded notebook does not include a
> completed `app.py`, deployment screenshots, or a live Streamlit URL.

## 📖 Project Overview

The Airline Tweet Sentiment Analyzer uses Natural Language Processing
(NLP) and Machine Learning to understand the sentiment expressed in
airline-related tweets. The project follows an end-to-end workflow,
starting with dataset inspection and text cleaning, followed by feature
extraction, model training, evaluation, error analysis, and preparation
of a model for deployment.

The project compares rule-based sentiment analysis, VADER, traditional
text features, and Word2Vec embeddings. It then selects a TF-IDF and
Logistic Regression model based on Macro-F1 performance.

## 🎯 Objectives

-   Understand the basic concepts and workflow of Natural Language
    Processing.
-   Explore and clean real-world airline tweets.
-   Apply regular-expression text cleaning, tokenization, and stop-word
    handling.
-   Preserve negation words that can change sentiment.
-   Build heuristic and VADER sentiment analysis approaches.
-   Explore Bag of Words, TF-IDF, n-grams, and Word2Vec embeddings.
-   Train and compare text classification models.
-   Evaluate models using accuracy, precision, recall, Macro-F1, and
    confusion matrices.
-   Perform Part-of-Speech (POS) tagging and POS-based feature
    experiments.
-   Save and reload a deployment-ready scikit-learn pipeline.

## 📂 Dataset

**Dataset name:** Twitter US Airline Sentiment

**Source:** [Kaggle: Twitter US Airline
Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

The dataset contains tweets about US airlines and their sentiment
labels. The notebook loads the file as `Tweets.csv`.

### Relevant columns

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `text`                              Original tweet text

  `airline_sentiment`                 Sentiment label: negative, neutral,
                                      or positive

  `airline`                           Airline mentioned in the dataset
  -----------------------------------------------------------------------

The original dataset contains **14,640 records** and 15 columns.

### Data cleaning summary

-   No missing values were found in the selected `text`,
    `airline_sentiment`, and `airline` columns.
-   The notebook found 36 fully duplicated rows.
-   It found 213 repeated tweet texts.
-   Duplicate tweets were removed using the `text` column.
-   The cleaned dataset contains **14,427 records**.

### Sentiment distribution after duplicate removal

  Sentiment         Tweets   Percentage
  ----------- ------------ ------------
  Negative           9,080       62.94%
  Neutral            3,057       21.19%
  Positive           2,290       15.87%
  **Total**     **14,427**     **100%**

The dataset is imbalanced, with negative tweets forming the largest
class. The classification models therefore use `class_weight="balanced"`
in the Logistic Regression experiments.

## 🧹 Project Workflow

``` text
Dataset Loading
      ↓
Data Inspection and EDA
      ↓
Duplicate Removal
      ↓
Text Cleaning with Regular Expressions
      ↓
Tokenization and Stop-word Handling
      ↓
Heuristic and VADER Sentiment Analysis
      ↓
Feature Extraction
(BoW, TF-IDF, N-grams, Word2Vec)
      ↓
Model Training and Comparison
      ↓
POS Tagging and POS Feature Experiment
      ↓
Error Analysis and Cross-validation
      ↓
Save and Test Deployment Pipeline
```

## 🧪 Practical Tasks

### Task 1: NLP Introduction and Exploratory Data Analysis

-   Introduction to NLP, common NLP tasks, and the NLP pipeline.
-   Loaded and inspected `Tweets.csv`.
-   Checked missing values and duplicate records.
-   Removed duplicate tweet text.
-   Explored sentiment distribution and sentiment by airline.
-   Calculated tweet lengths and examined common NLP challenges such as
    informal language, spelling variations, hashtags, and ambiguity.

**Dataset observations:** - Average tweet length: **17.72 words**. -
Median tweet length: **19 words**. - Negative sentiment is the majority
class.

### Task 2: Text Cleaning and Tokenization

-   Practiced regular expressions for URL, mention, number, punctuation,
    and whitespace handling.
-   Created a `clean_text()` function.
-   Created cleaned text and token columns.
-   Applied NLTK tokenization and English stop-word removal.
-   Preserved negation words such as `no`, `not`, `nor`, and `never`.
-   Compared vocabulary sizes and frequent words across sentiment
    classes.

### Task 3: Sentiment Analysis Approaches

Implemented and compared:

1.  **Heuristic approach:** Uses positive and negative word lists with
    basic negation handling.
2.  **VADER:** Uses the NLTK VADER sentiment analyzer and compound-score
    thresholds.

Both approaches were evaluated on the same test split.

  Approach      Macro-F1
  ----------- ----------
  Heuristic       0.5122
  VADER           0.5146

These approaches provide useful baselines, but their scores were lower
than the trained TF-IDF classification models.

### Task 4: Text Feature Extraction

-   Built a Bag-of-Words matrix manually and with `CountVectorizer`.
-   Calculated TF-IDF manually and compared it with scikit-learn.
-   Explored unigram and bigram features.
-   Compared vocabulary sizes and feature limits.
-   Examined important bigrams in negative and positive tweets.
-   Fit vectorizers on training text and transformed test text to avoid
    vocabulary leakage.

The train-test split contains:

  Split             Tweets
  ----------- ------------
  Training          11,541
  Testing            2,886
  **Total**     **14,427**

### Task 5: Word Embeddings with Word2Vec

-   Explained Continuous Bag of Words (CBoW) and Skip-gram.
-   Trained both Word2Vec approaches using training tweets.
-   Used 100-dimensional word vectors.
-   Explored similar words and cosine similarity.
-   Visualized selected word embeddings using PCA.
-   Converted tweets into document vectors by averaging word embeddings.
-   Selected Skip-gram for the document-vector experiment.

### Task 6: Text Classification

Trained and compared five feature-classifier combinations:

-   Bag of Words + Multinomial Naive Bayes
-   Bag of Words + Logistic Regression
-   TF-IDF Unigrams + Logistic Regression
-   TF-IDF Unigrams and Bigrams + Logistic Regression
-   Averaged Word2Vec + Logistic Regression

The project also includes: - Baseline TF-IDF and Logistic Regression
pipeline. - Accuracy and Macro-F1 evaluation. - Per-class precision,
recall, and F1-score. - Confusion matrix. - Comparison of Heuristic,
VADER, BoW, TF-IDF, n-gram, and Word2Vec approaches. - Five-fold
stratified cross-validation. - Misclassified tweet inspection and
probability-based error analysis. - Saving and reloading the final
model.

### Task 7: Part-of-Speech Tagging

-   Used NLTK POS tagging on five sample tweets and ambiguity examples.
-   Explored how a word can be tagged differently depending on context.
-   Tagged a sample of 2,000 training tweets.
-   Calculated average proportions of adjectives, adverbs, verbs, and
    nouns by sentiment.
-   Extracted frequent adjectives for negative and positive tweets.
-   Compared full-text TF-IDF with POS-filtered TF-IDF.

**POS-filtering experiment:**

  Model                   Macro-F1
  --------------------- ----------
  Full-text TF-IDF          0.7510
  POS-filtered TF-IDF       0.6349

POS filtering reduced the Macro-F1 score in this experiment. Removing
nouns, pronouns, and other context words can remove useful information
that helps distinguish sentiment.

### Task 8: Deployment Preparation

-   Planned the Streamlit app layout.
-   Trained a TF-IDF and Logistic Regression deployment pipeline.
-   Saved the pipeline as `sentiment_model.joblib`.
-   Reloaded the saved pipeline and tested three new sentences.
-   Recorded the scikit-learn version used by the notebook.

The notebook includes the deployment model preparation and sample
prediction test. The Streamlit application code and online deployment
still need to be added as separate project files.

## 📊 Model Results

The following results are taken from the notebook's held-out test-set
comparison.

  -------------------------------------------------------------------------
  Feature Method  Classifier                  Accuracy             Macro-F1
  --------------- --------------- -------------------- --------------------
  TF-IDF          Logistic                  **0.7963**           **0.7510**
  Unigrams +      Regression                           
  Bigrams                                              

  TF-IDF Unigrams Logistic                      0.7713               0.7291
                  Regression                           

  BoW             Logistic                      0.7595               0.7182
                  Regression                           

  BoW             Multinomial                   0.7769               0.6995
                  Naive Bayes                          

  Averaged        Logistic                      0.7315               0.6788
  Word2Vec        Regression                           
  -------------------------------------------------------------------------

### Best model

**TF-IDF Unigrams + Bigrams with Logistic Regression**

-   Accuracy: **79.63%**
-   Macro-F1: **0.7510**
-   Negative-class precision: **0.8893**
-   Positive-class recall: **0.7555**
-   Training time recorded in the comparison: approximately **0.912
    seconds**

### Cross-validation

  Evaluation                       Macro-F1
  ------------------------------ ----------
  5-fold cross-validation mean       0.7381
  Held-out test set                  0.7510

Cross-validation standard deviation: **0.0091**.

### Comparison of NLP approaches

  Approach      Macro-F1
  ----------- ----------
  N-gram          0.7510
  TF-IDF          0.7510
  BoW             0.7182
  Word2Vec        0.6788
  VADER           0.5146
  Heuristic       0.5122

## 🔍 Error Analysis and Limitations

The notebook reports **588 misclassified tweets** for the selected model
on the 2,886-tweet test set.

A sample deployment test also shows that the model can make an
unexpected prediction. For example, the sentence:

> The aircraft arrived on time

was predicted as **negative** in the notebook's saved-model test, with
probabilities:

-   Negative: 0.4191
-   Neutral: 0.1976
-   Positive: 0.3833

This illustrates that the model does not always interpret context
correctly, even when a sentence appears positive or neutral to a person.

Other limitations include: - Sarcasm and irony are difficult to detect
using basic TF-IDF features. - Informal language, spelling variations,
and abbreviations can affect predictions. - The dataset has an
imbalanced sentiment distribution. - POS tagging may be less reliable on
short, informal tweets. - The model is trained on airline-related tweets
and may not generalize to unrelated topics.

## 💾 Model Saving and Prediction

The notebook saves the trained pipeline as:

``` text
sentiment_model.joblib
```

The deployment pipeline contains: -
`TfidfVectorizer(ngram_range=(1, 2), min_df=2)` -
`LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`

Example code to load the saved model:

``` python
import joblib

# Load the trained pipeline
model = joblib.load("sentiment_model.joblib")

# Prepare a sample tweet
tweet = "The flight service was excellent"

# Apply the same cleaning function used during training
cleaned_tweet = clean_text(tweet)

# Predict sentiment
prediction = model.predict([cleaned_tweet])[0]

# Get class probabilities
probabilities = model.predict_proba([cleaned_tweet])[0]

print("Prediction:", prediction)
print("Classes:", model.classes_)
print("Probabilities:", probabilities)
```

**Important:** Keep the `clean_text()` implementation consistent with
the one used during training. The saved scikit-learn pipeline handles
TF-IDF transformation and classification, but it does not include the
external text-cleaning function.

## 🛠️ Technologies and Libraries

-   Python
-   Jupyter Notebook
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Regular Expressions (`re`)
-   NLTK
-   Gensim (Word2Vec)
-   Scikit-learn
-   Joblib
-   Streamlit (planned app interface)

## ⚙️ Installation and Usage

### 1. Clone the repository

``` bash
git clone YOUR_PUBLIC_GITHUB_REPOSITORY_URL
cd Airline_Tweet_Sentiment_Analyzer
```

### 2. Create and activate a virtual environment (Windows)

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

Create `requirements.txt` using the package versions from the
environment used for the project. The notebook records **scikit-learn
1.9.0**.

The notebook-generated requirements file currently lists pandas, numpy,
matplotlib, seaborn, scikit-learn, and joblib. For the complete notebook
and Streamlit app, also include the required versions of NLTK, Gensim,
and Streamlit.

``` bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `Tweets.csv` from the [Kaggle dataset
page](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
and place it in the project root, next to `NLP_PR1.ipynb`.

The raw dataset is not included in this repository.

### 5. Run the notebook

Open `NLP_PR1.ipynb` in Jupyter Notebook, JupyterLab, Google Colab, or
VS Code and run the cells in order.

### 6. Generate the saved model

Run the model training and saving cells in Task 8. This creates
`sentiment_model.joblib` in the current working directory.

### 7. Run the Streamlit app

After creating `app.py` and installing Streamlit, run:

``` bash
streamlit run app.py
```

The uploaded notebook does not contain the completed `app.py`, so the
app must be added before this command can run successfully.

## 📁 Repository Structure

Recommended repository structure:

``` text
Airline_Tweet_Sentiment_Analyzer/
│
├── NLP_PR1.ipynb
├── app.py
├── nlp_utils.py
├── sentiment_model.joblib
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── start_screen.png
    ├── positive_prediction.png
    ├── negative_prediction.png
    ├── probability_chart.png
    └── pos_tags.png
```

`Tweets.csv` is downloaded separately from Kaggle and should not be
committed to the repository.

## 🖼️ Screenshots

Add your exported notebook plots and application screenshots to the
`screenshots/` folder. Suggested items:

-   Sentiment distribution
-   Sentiment distribution by airline
-   Tweet length distribution
-   Top words and bigrams
-   Macro-F1 model comparison
-   Confusion matrix
-   Word2Vec PCA visualization
-   POS distribution by sentiment
-   Streamlit start screen and prediction results

The uploaded notebook contains outputs and plots, but these have not
been exported as separate image files in this README package.

## 🚀 Future Improvements

-   Improve sarcasm and context understanding.
-   Explore a Transformer model such as BERT.
-   Improve handling of spelling variations and informal expressions.
-   Add more robust validation and monitoring to the Streamlit
    application.
-   Test the model on newer airline tweets to assess generalization.

## 📝 Conclusion

This project demonstrates an end-to-end NLP workflow for airline tweet
sentiment analysis. It covers dataset exploration, text cleaning,
tokenization, sentiment analysis, feature extraction, Word2Vec
embeddings, classification, POS tagging, model evaluation, error
analysis, and deployment preparation.

Among the evaluated models, TF-IDF with unigrams and bigrams combined
with Logistic Regression achieved the best test Macro-F1 score of
**0.7510** and an accuracy of **79.63%**. The POS-filtering experiment
performed less well than the full-text model, showing that removing
words can also remove useful context.

The project provided practical experience in transforming raw text into
machine-learning features, comparing NLP approaches, evaluating
classification models, and saving a trained pipeline for future
application use.

------------------------------------------------------------------------

## 👤 Author

**Santosh Verma**

NLP Practical Report 1

------------------------------------------------------------------------

::: {align="center"}
**Airline Tweet Sentiment Analyzer \| NLP Project**
:::
