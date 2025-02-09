## **Amazon-Customer-Reviews-Sentiment-Analysis**

## **why I trained the model on restaurant reviews instead of Amazon reviews ?**

- 'I chose to train the model on restaurant reviews because they are more readily available and easily accessible. Additionally, restaurant reviews are more general and can be applied to a wider range of domains, making it easier to adapt the model to other domains. The sentiment analysis complexity in restaurant reviews is also higher, which allows us to develop a more robust and accurate sentiment analysis system. Furthermore, training on restaurant reviews allows us to develop a more generalizable model that can be applied to other domains, rather than a model that is specific to Amazon reviews. Finally, the data quality in restaurant reviews is often higher, which improves the accuracy of the model.’
 
 ------------

## part 1 : Training model on restaurent reviews.

# **Sentiment Analysis of Restaurant Reviews 📊**

## **Dependencies 📚**

- **`pandas`** for data manipulation
- **`numpy`** for numerical computations
- **`matplotlib`** and **`seaborn`** for data visualization
- **`scikit-learn`** for machine learning tasks
- **`nltk`** for natural language processing tasks

## **Dataset 📁**

- The dataset consists of restaurant reviews with corresponding labels (positive or negative)
- The dataset is imbalanced, with more positive reviews than negative reviews

## **EDA 📊**

- The dataset contains 10,000 reviews with an average length of 100 words
- The most common words in the dataset are "food", "service", "price", "quality", and "ambiance"
- The wordcloud shows that the most common words are related to the restaurant's food and service

## **Nature of Dataset 📝**

- The dataset is a collection of text reviews with corresponding labels
- The text reviews are unstructured and contain varying lengths and formats

## **Preprocessing 📄**

- **Tokenization**: split the text reviews into individual words or tokens
- **Stop Words Removal**: remove common words like "the", "and", "a", etc. that do not add much value to the meaning of the review
- **Lemmatization**: reduce words to their base or root form (e.g. "running" becomes "run")
- **Vectorization**: convert the text reviews into numerical vectors using TF-IDF

## **Train Test Split 📊**

- Split the preprocessed data into training (80%) and testing (20%) sets

## **TF-IDF 📈**

- Used TF-IDF to convert the text reviews into numerical vectors
- TF-IDF takes into account the importance of each word in the review and the frequency of the word in the entire dataset

## **Model Training 📚**

- Trained SVM , Logistic Regression,ComplementNB , XGBoost,knn,BernoulliNB model, **Mutilnominal NB model,** on the training data
- Finalized the Multinomial Naive Bayes Classifier with 79% accuracy.
- The model is trained to predict the label (positive or negative) of a review based on its features

## **Model Evaluation 📊**

- Evaluated the model on the testing data using accuracy, precision, recall, and F1-score metrics
- The model achieved an accuracy of 79% on the testing data

## **Finalized Model 📈**

- In this case, the MNB model has an AUC-ROC of approximately 0.857 and the KNN model has an AUC-ROC of approximately 0.822. This suggests that the MNB model is performing better at distinguishing between the classes compared to the KNN model for the given dataset.
- The finalized model is a Multinomial Naive Bayes Classifier trained on the preprocessed data
- The model can be used to predict the sentiment of new, unseen reviews.


-----------------------------------------------------------------------------------------
----------------------------------
## Part 2: 

# Amazon Reviews Web Scraping

## Overview

This notebook is designed to scrape customer reviews from an Amazon product page. It extracts key information such as reviewer name, star rating, review title, date, and the review description. The scraped data is then organized into a Pandas DataFrame for potential analysis or storage.

## Technologies Used

*   **Python:** The core programming language for the entire project.
*   **requests:** Used to make HTTP requests to fetch the HTML content of the Amazon product review pages.
*   **Beautiful Soup 4 (bs4):** Used for parsing the HTML structure of the web pages, making it easy to navigate and extract specific data elements.
*   **Pandas:** Used to create and manipulate DataFrames, which are ideal for organizing and working with structured data.
*   **lxml:** An efficient XML and HTML processing library that serves as the parser for Beautiful Soup.
*   **datetime:** Used for handling and converting date formats extracted from the reviews.

## Setup and Installation

1.  **Install Python:** Ensure you have Python 3.6 or higher installed.
2.  **Install Libraries:** Use pip to install the required libraries:

    ```
    pip install requests pandas beautifulsoup4 lxml
    ```

## Project Structure and Steps

1.  **Import Libraries:** The necessary libraries are imported to enable HTTP requests, HTML parsing, and data manipulation.
2.  **Define Headers:** A header is defined to mimic browser requests, to avoid being blocked by Amazon.
    ```
    headers = {
     'authority': 'www.amazon.com',
     'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
     'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
     'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
     'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }
    ```
3.  **Define the Amazon Reviews URL:** The specific URL of the Amazon product review page to be scraped is defined.
    ```
    reviews_url = 'https___' 
    ```
4.  **Define the Number of Pages to Scrape:** The script defines how many pages of reviews to scrape.
     ```
    len_page = 4 # Number of pages to scrape
    ```
5.  **`reviewsHtml` Function:** This function is responsible for fetching the HTML content of the Amazon review pages. It takes the URL and number of pages as input, makes HTTP requests, and uses Beautiful Soup to parse the HTML.
6.  **`getReviews` Function:** This function takes the parsed HTML and extracts the relevant review data (name, stars, title, date, description) from each review box. It handles potential missing data by assigning default values.
7.  **Data Processing:** This section orchestrates the scraping process. It calls the `reviewsHtml` function to get the HTML data, then iterates through each page, using the `getReviews` function to extract the data. The extracted data is stored in a list.
8.  **Create Pandas DataFrame:** The scraped data is converted into a Pandas DataFrame for easy manipulation and analysis.
9.  **Display DataFrame:**  The script then displays the first few rows of the DataFrame.

## Usage

1.  Run the notebook.
2.  The scraped data will be displayed as a Pandas DataFrame.
3.  You can further process or save the data as needed.

## Notes

*   Amazon's website structure may change, which could break the script. You may need to update the CSS selectors accordingly.
*   Be respectful of Amazon's terms of service and avoid making excessive requests in a short period. Consider adding delays between requests.
*   This script is for educational purposes only.

--------------------------------------------------
-------------------------------

## Part 3:

# Amazon Reviews Sentiment Analysis 

This Jupyter notebook applies a pre-trained Multinomial Naive Bayes model to predict sentiments for scraped Amazon product reviews. The workflow includes text preprocessing, TF-IDF vectorization, and sentiment prediction.

## Overview
- **Objective**: Predict sentiment labels (0: negative, 1: positive) for Amazon reviews using a pre-trained model.
- **Input Data**: `reviews.csv` containing columns: `Name`, `Stars`, `Title`, `Date`, `Description`.
- **Output**: A DataFrame with an added `sentiment` column indicating predicted sentiments.

---

## Dependencies
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

## Example Output
| Name              | Stars | Title                                   | Date | Description                                      | Sentiment |
|-------------------|-------|-----------------------------------------|------|--------------------------------------------------|-----------|
| Amazon Customer   | 5.0   | 5.0 out of 5 stars\nWhite Jeans         | NaN  | "Quality, material, fit, comfort..."             | 1         |
| Chandana Banerjee | 2.0   | 2.0 out of 5 stars\nCheap Quality...    | NaN  | "Low quality dye..."                             | 0         |

---

## Usage Notes
- Ensure `reviews.csv`, `multinomial_nb_model.pkl`, and `tfidf_vectorizer.pkl` are in the working directory.
- The `Date` column may contain `NaN` values if the original data lacks timestamps.
- Adjust the `preprocess_text` function if additional text cleaning steps are needed.

