# Amazon_Product_Reviews_Sentiment_Analysis

Keywords: nltk, Stopwords, sklearn, matplot, tfidVectorizer, wordCloud, Pandas, filterwarnings, LogisticRegression, ModelFitting, ConfusionMatrix

#####################################################################
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

**warnings:** This module is used to control warnings in Python. The filterwarnings('ignore') command suppresses any warning messages that might arise during code execution, making the output cleaner.
**pandas:** A powerful data manipulation and analysis library, pandas provides data structures like DataFrames to easily work with structured data, often used for tasks like data cleaning, exploration, and analysis.
**sklearn.feature_extraction.text.TfidfVectorizer:** This is part of scikit-learn, a machine learning library. The TfidfVectorizer converts text data into a matrix of TF-IDF features, which reflect how important words are in documents, relative to the entire corpus.
**matplotlib.pyplot:** A plotting library for Python, pyplot is used to create static, interactive, and animated visualizations. It provides a flexible framework to generate graphs and plots.
**wordcloud:** This library generates word clouds, which are visual representations of word frequency in text. The more frequent a word, the larger and bolder it appears in the cloud.

###############################################################
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

**nltk:** The Natural Language Toolkit (NLTK) is a powerful Python library for working with human language data (text). It provides tools for various text processing tasks like tokenization, stemming, tagging, and more.
**nltk.download('punkt'):** This command downloads the 'punkt' tokenizer model, which is used for tokenizing text into sentences or words. It’s essential for breaking down text into smaller pieces for further processing.
**nltk.download('stopwords'):** This command downloads a list of common stopwords (like "and," "the," "in") in various languages. Stopwords are often removed during text preprocessing since they are considered to add little value in many NLP tasks.
**from nltk.corpus import stopwords:** This line imports the stopwords list from the NLTK corpus, which allows you to filter out common stopwords during text processing tasks.

##################################################Loading Data
data = pd.read_csv('/content/Amazon-Product-Reviews-Sentiment-Analysis-in-Python-Dataset.csv')
The dataset, which contains Amazon product reviews, is being loaded into a pandas DataFrame from a CSV file.

#########################################################Initial Data Exploration
data.head()
data.info()
data.head() shows the first 5 rows of the dataset, while data.info() provides information on the DataFrame, such as the number of non-null entries, data types, and memory usage.

#############################################Removing Missing Values
data.dropna(inplace=True)
This removes rows with missing values from the dataset.

#########################################################Reclassifying Sentiment Values
data.loc[data['Sentiment'] <= 3, 'Sentiment'] = 0
data.loc[data['Sentiment'] > 3, 'Sentiment'] = 1
Sentiment values are being reclassified into binary categories:
Reviews with sentiment scores from 1 to 3 are labeled as 0 (negative).
Reviews with sentiment scores of 4 and 5 are labeled as 1 (positive).

#####################################################Cleaning the Review Text
stp_words = stopwords.words('english')
def clean_review(review):
    cleanreview = " ".join(word for word in review.split() if word not in stp_words)
    return cleanreview
data['Review'] = data['Review'].apply(clean_review)
A list of English stopwords is created using NLTK.
The function clean_review removes all stopwords from the review text by splitting the text into words, filtering out stopwords, and joining the remaining words back into a string.
The apply() method is used to apply this cleaning function to the 'Review' column in the dataset.

######################################################Reviewing Results
data.head()
data['Sentiment'].value_counts()
data.head() displays the first 5 rows of the cleaned data again.
data['Sentiment'].value_counts() gives the distribution of the two sentiment classes (0 for negative, 1 for positive), showing how many reviews fall into each category.

############################################TF-IDF Vectorization
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['Review']).toarray()
TfidfVectorizer converts the cleaned text reviews into a matrix of TF-IDF features. Here, you specify max_features=2500, which limits the vocabulary to the top 2,500 words based on importance.
X becomes a matrix where each row represents a review, and each column corresponds to a word from the vocabulary, with TF-IDF values as the elements.

####################################################Splitting Data into Training and Testing Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, data['Sentiment'], test_size=0.25, random_state=42)
The data is split into training and testing sets using train_test_split().
test_size=0.25 means 25% of the data is reserved for testing, while 75% is used for training.
random_state=42 ensures that the split is reproducible.

#####################################################Model Selection (Logistic Regression):
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
A logistic regression model is initialized. This is a widely used classifier in sentiment analysis tasks.

################################################################Training the Model
model.fit(x_train, y_train)
The logistic regression model is trained on the training data (x_train and y_train).

##############################################################Predicting on Test Data
pred = model.predict(x_test)
The trained model predicts the sentiment labels for the test data (x_test).

#########################################################Model Evaluation
print(accuracy_score(y_test, pred))
accuracy_score(y_test, pred) calculates the accuracy of the model by comparing the predicted labels (pred) with the actual labels (y_test).

############################################################Confusion Matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
cm_display.plot()
plt.show()
A confusion matrix is generated using confusion_matrix(y_test, pred), which provides detailed information about the model's performance: true positives, true negatives, false positives, and false negatives.
The confusion matrix is displayed using ConfusionMatrixDisplay, which visualizes the results in a grid.
