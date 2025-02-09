# Amazon Reviews Analysis and Scraping 🛍️

## Overview 📝

This project combines web scraping of Amazon product reviews with sentiment analysis. It first trains a sentiment analysis model on restaurant reviews and then applies this model (along with web scraping techniques) to analyze customer reviews scraped from an Amazon product page.

--------
--------

## Part 1: Sentiment Analysis Model Training 🤖

### Why Restaurant Reviews? 🤔

The sentiment analysis model was trained on restaurant reviews due to their broader availability, generalizability, higher sentiment complexity, better data quality, and ease of access compared to Amazon reviews.

### Dependencies 📚

*   `pandas`: Data manipulation
*   `numpy`: Numerical computations
*   `matplotlib` & `seaborn`: Data visualization
*   `scikit-learn`: Machine learning tasks
*   `nltk`: Natural Language Processing

### Dataset 📁

The dataset consists of restaurant reviews with corresponding sentiment labels (positive or negative). Note that the dataset is imbalanced, with more positive reviews than negative reviews.

### EDA 📊

*   The dataset contains 10,000 reviews with an average length of 100 words.
*   Common words include "food", "service", "price", "quality", and "ambiance."
*   Word cloud visualizations highlight the prominence of terms related to food and service.

### Nature of Dataset 📝

The dataset is a collection of text reviews with corresponding labels. The text reviews are unstructured and contain varying lengths and formats.

### Preprocessing ⚙️

*   **Tokenization:** Splitting text into individual words.
*   **Stop Words Removal:** Removing common, non-informative words (e.g., "the," "and").
*   **Lemmatization:** Reducing words to their base form (e.g., "running" -> "run").
*   **Vectorization:** Converting text to numerical vectors using TF-IDF.

### Train Test Split 🧪

The preprocessed data is split into training (80%) and testing (20%) sets.

### TF-IDF 📈

TF-IDF is used to convert the text reviews into numerical vectors, taking into account word importance and frequency.

### Model Training 🏋️‍♀️

Trained several models (SVM, Logistic Regression, ComplementNB, XGBoost, KNN, BernoulliNB, and Multinomial NB). The Multinomial Naive Bayes model was finalized due to its 79% accuracy.

### Model Evaluation ✅

The model was evaluated using accuracy, precision, recall, and F1-score metrics. The Multinomial Naive Bayes model achieved an accuracy of 79% on the testing data.

### Finalized Model 🏆

The finalized model is a Multinomial Naive Bayes Classifier trained on the preprocessed data. It demonstrated better performance compared to the KNN model, with an AUC-ROC of approximately 0.857. The model can be used to predict the sentiment of new, unseen reviews.

-------------
--------

## Part 2: Amazon Reviews Web Scraping 🕸️

### Overview 📝

This section focuses on scraping customer reviews from an Amazon product page. Key information extracted includes reviewer name, star rating, review title, date, and review description. The scraped data is then organized into a Pandas DataFrame for further use (such as feeding it into the sentiment analysis model).

### Technologies Used 🛠️

*   Python: Core programming language.
*   requests: Fetching HTML content from Amazon review pages.
*   Beautiful Soup 4 (bs4): Parsing HTML structure for easy data extraction.
*   Pandas: Creating and manipulating DataFrames.
*   lxml: Efficient HTML/XML processing for Beautiful Soup.
*   datetime: Handling and converting date formats.

### Setup and Installation ⚙️

1.  Install Python (3.6 or higher).
2.  Install required libraries:

    ```
    pip install requests pandas beautifulsoup4 lxml nltk scikit-learn
    ```

### Project Structure and Steps 👣

1.  **Import Libraries:** Import necessary libraries.
2.  **Define Headers:** Set headers to mimic browser requests.
3.  **Define the Amazon Reviews URL:** Specify the target Amazon product review page URL.
4.  **Define the Number of Pages to Scrape:** Determine how many pages of reviews to scrape.
5.  **`reviewsHtml` Function:** Fetch HTML content of review pages using HTTP requests and Beautiful Soup.
6.  **`getReviews` Function:** Extract review data (name, stars, title, date, description) from HTML. Handles missing data.
7.  **Data Processing:** Orchestrate the scraping process, extract data using `reviewsHtml` and `getReviews`, and store it.
8.  **Create Pandas DataFrame:** Convert the scraped data into a Pandas DataFrame.
9.  **Display DataFrame:** Show the first few rows of the DataFrame.

### Usage 🚀

1.  Run the notebook.
2.  The scraped data will be displayed as a Pandas DataFrame.
3.  Further process or save the data as needed.

### Notes ⚠️

*   Amazon's website structure may change, potentially breaking the script.
*   Adhere to Amazon's terms of service to prevent being blocked.
*   This script is for educational purposes only.



--------------------------------------------------
-------------------------------

## Part 3:

# **Amazon Reviews Sentiment Analysis📊** 

In this section, the sentiment analysis model trained on restaurant reviews is applied to the Amazon reviews scraped in Part 2. The goal is to predict the sentiment (positive or negative) of each Amazon review and gain insights into customer opinions about the product.

## Overview📝

- **Objective**: Predict sentiment labels (0: negative, 1: positive) for Amazon reviews using a pre-trained model.
- **Input Data**: `reviews.csv` containing columns: `Name`, `Stars`, `Title`, `Date`, `Description`.
- **Output**: A DataFrame with an added `sentiment` column indicating predicted sentiments.

---

## Dependencies📚
- Libraries:
  - `pandas`, `pickle`, `re`
  - `sklearn` (for `TfidfVectorizer`, `RandomForestClassifier`, metrics)
  - `nltk` (for stopwords, tokenization, lemmatization)
- Pre-trained Models:
  - `multinomial_nb_model.pkl`: Trained Multinomial Naive Bayes model.
  - `tfidf_vectorizer.pkl`: TF-IDF vectorizer for text transformation.

---

## Workflow

### 1. Data Loading
- Reads scraped reviews from `reviews.csv` into a DataFrame.
- Columns include `Name`, `Stars`, `Title`, `Date`, and `Description`.

### 2. Text Preprocessing
The `preprocess_text` function performs:
1. **Cleaning**: Removes non-alphanumeric characters and converts text to lowercase.
2. **Tokenization**: Splits text into words.
3. **Stopword Removal**: Filters out common English stopwords.
4. **Lemmatization**: Reduces words to their base form (e.g., "running" → "run").

### 3. Model and Vectorizer Loading
- Loads the pre-trained Multinomial Naive Bayes model and TF-IDF vectorizer from `.pkl` files.

### 4. Sentiment Prediction
1. **Preprocess** each review's `Description`.
2. **Transform** text into TF-IDF features using the loaded vectorizer.
3. **Predict** sentiment using the loaded model.
4. **Append** predictions to a new `sentiment` column in the DataFrame.

### 5. Results
- The final DataFrame includes the original data with predicted sentiments:
  - `1`: Positive sentiment.
  - `0`: Negative sentiment.

---

## Example Output📈
| Name              | Stars | Title                                   | Date | Description                                      | Sentiment |
|-------------------|-------|-----------------------------------------|------|--------------------------------------------------|-----------|
| Amazon Customer   | 5.0   | 5.0 out of 5 stars\nWhite Jeans         | NaN  | "Quality, material, fit, comfort..."             | 1         |
| Chandana Banerjee | 2.0   | 2.0 out of 5 stars\nCheap Quality...    | NaN  | "Low quality dye..."                             | 0         |

---

## Usage Notes
- Ensure `reviews.csv`, `multinomial_nb_model.pkl`, and `tfidf_vectorizer.pkl` are in the working directory.
- The `Date` column may contain `NaN` values if the original data lacks timestamps.
- Adjust the `preprocess_text` function if additional text cleaning steps are needed.

