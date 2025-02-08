# Databricks notebook source
# File location and type
file_location = "/FileStore/tables/df.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a temporary view
df.createOrReplaceTempView("table")


# COMMAND ----------

# Display the data
df = spark.sql("SELECT * FROM table")
display(df)


# COMMAND ----------

# Rename the columns
df = df.withColumnRenamed("_c0", "Title") \
       .withColumnRenamed("_c1", "Price") \
       .withColumnRenamed("_c2", "Shipping") \
       .withColumnRenamed("_c3", "Link") \
       .withColumnRenamed("_c4", "Seller_Feedback_Rating") \
       .withColumnRenamed("_c5", "Seller_Reviews")


# COMMAND ----------

from pyspark.sql.functions import regexp_replace, col

# Remove non-numeric characters and convert to float
df = df.withColumn("Price", regexp_replace(col("Price"), "[^0-9.]", "").cast("float"))


# COMMAND ----------

# Calculate total cost
df = df.withColumn("Total_Cost", col("Price") + col("Shipping"))


# COMMAND ----------

# Descriptive statistics for cleaned data
df.select("Price", "Shipping", "Total_Cost").describe().show()


# COMMAND ----------

# Convert Seller Feedback Rating to numeric
df = df.withColumn("Seller_Feedback_Rating", regexp_replace(col("Seller_Feedback_Rating"), "[^0-9.]", "").cast("float"))

# Descriptive statistics for Seller Feedback Rating
df.select("Seller_Feedback_Rating").describe().show()


# COMMAND ----------

from pyspark.sql.functions import size, split

# Count the number of reviews per seller
df = df.withColumn("Number_of_Reviews", size(split(col("Seller_Reviews"), ",")))

# Descriptive statistics for Number of Reviews
df.select("Number_of_Reviews").describe().show()


# COMMAND ----------

# Display the final DataFrame with all transformations
display(df)


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# List of numerical columns to plot
numerical_columns = ["Price", "Total_Cost"]

# Create a single figure for histograms
plt.figure(figsize=(10, 5))

# Plot histograms for each numerical column
for column in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df.select(column).toPandas(), kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# COMMAND ----------

import re
import html
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Define the text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ''
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove brackets and their contents
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\{.*?\}', '', text)

    # Remove punctuation and special characters
    text = re.sub(r'[!"\'#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text)
    
    # Remove emoticons and non-ASCII characters
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]', '', text)
    text = ''.join(char for char in text if ord(char) < 128)

    # Convert text to lowercase
    text = text.lower()

    return text

# Create a User Defined Function (UDF) for cleaning text
clean_text_udf = udf(clean_text, StringType())

# Apply the UDF to the 'Seller_Reviews' column to create the 'clean_text' column
df = df.withColumn("clean_text", clean_text_udf(df["Seller_Reviews"]))

# Check if 'clean_text' column was created successfully
df.printSchema()

# Display a few rows to verify the 'clean_text' column
df.select("Title", "Seller_Reviews", "clean_text").show(5, truncate=False)



# COMMAND ----------

from pyspark.ml.feature import Tokenizer

# Tokenize the cleaned text
tokenizer = Tokenizer(inputCol="clean_text", outputCol="tokens")
df = tokenizer.transform(df)

# Display tokens
df.select("clean_text", "tokens").show(5, truncate=False)


# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

# Create a StopWordsRemover instance
remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_no_stopwords")

# Remove stop words
df = remover.transform(df)

# Display tokens without stopwords
df.select("tokens", "tokens_no_stopwords").show(5, truncate=False)


# COMMAND ----------

from collections import Counter
import pandas as pd
import plotly.express as px

# Flatten the list of tokens after stopword removal
all_tokens_no_stopwords = [token for row in df.select("tokens_no_stopwords").collect() for token in row.tokens_no_stopwords]

# Count the frequency of each word
word_counts_no_stopwords = Counter(all_tokens_no_stopwords)

# Get the 10 most common words
top_words_no_stopwords = word_counts_no_stopwords.most_common(10)

# Convert the top words to a DataFrame for easy plotting
top_words_no_stopwords_df = pd.DataFrame(top_words_no_stopwords, columns=['Word', 'Frequency'])

# Create a bar chart using Plotly
fig = px.bar(top_words_no_stopwords_df, x='Word', y='Frequency', title='Top 10 Words After Removing Stopwords',
             labels={'Word': 'Words', 'Frequency': 'Frequency'},
             color='Frequency', color_continuous_scale='Blues')

# Show the plot
fig.show()


# COMMAND ----------

# Install the wordcloud package
%pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Join the tokens into a single string for the word cloud
text_no_stopwords = " ".join(all_tokens_no_stopwords)

# Create a WordCloud instance and generate the word cloud
wordcloud_no_stopwords = WordCloud(width=800, height=400, background_color='white').generate(text_no_stopwords)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_no_stopwords, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.title('Word Cloud After Removing Stopwords')
plt.show()


# COMMAND ----------

from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql import SparkSession

# Create a Spark session (if not already created)
spark = SparkSession.builder.appName("TF-IDF Example").getOrCreate()

# HashingTF for tokenized words
hashingTF = HashingTF(inputCol="tokens_no_stopwords", outputCol="raw_features", numFeatures=1000)
featurizedData = hashingTF.transform(df)

# Fit the IDF model and transform the data
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Show the DataFrame with TF-IDF features
rescaledData.select("features").show(truncate=False)


# COMMAND ----------

import numpy as np
import pandas as pd

# Convert the features (SparseVector) to an array
feature_vectors = rescaledData.select("features").rdd.map(lambda x: x[0].toArray()).collect()

# Create a DataFrame for terms and their corresponding scores
terms = [f"Term_{i}" for i in range(len(feature_vectors[0]))]  # Create dummy terms based on the number of features
tfidf_scores = np.array(feature_vectors)

# Create a DataFrame with terms and their TF-IDF scores
terms_df = pd.DataFrame(tfidf_scores, columns=terms)

# Sum the scores for each term across all documents
sum_tfidf = terms_df.sum(axis=0)

# Create a DataFrame for top terms with their TF-IDF scores
tfidf_results = pd.DataFrame(sum_tfidf, columns=['Score']).reset_index()
tfidf_results.columns = ['Term', 'Score']
tfidf_results = tfidf_results.sort_values(by='Score', ascending=False).head(20)  # Get top 20 terms

# Display the top terms with their TF-IDF scores
display(tfidf_results)


# COMMAND ----------

from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF

# Create a CountVectorizer
count_vectorizer = CountVectorizer(inputCol="tokens_no_stopwords", outputCol="raw_features", vocabSize=1000, minDF=1.0)
cv_model = count_vectorizer.fit(df)
featurizedData = cv_model.transform(df)

# Fit the IDF model and transform the data
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Extract feature names
vocab = cv_model.vocabulary

# Create a DataFrame for the terms and their corresponding scores
tfidf_scores = rescaledData.select("features").rdd.map(lambda x: x[0].toArray()).collect()

# Create a DataFrame for terms and their corresponding scores
terms = [vocab[i] for i in range(len(tfidf_scores[0]))]  # Get the original terms
tfidf_list = []

for i in range(len(tfidf_scores)):
    for j in range(len(tfidf_scores[i])):
        tfidf_list.append((terms[j], tfidf_scores[i][j]))

# Create a DataFrame
tfidf_df = pd.DataFrame(tfidf_list, columns=["Term", "Score"])

# Group by term and sum the scores
tfidf_grouped = tfidf_df.groupby("Term", as_index=False).sum("Score")

# Sort by score in descending order and take the top 20 terms
top_tfidf_terms = tfidf_grouped.sort_values(by="Score", ascending=False).head(20)

# Display the top terms with their TF-IDF scores
display(top_tfidf_terms)


# COMMAND ----------

import plotly.express as px

# Create a bar chart for the top terms
fig = px.bar(top_tfidf_terms, x='Term', y='Score', title='Top 20 Terms Based on TF-IDF Scores',
             labels={'Term': 'Terms', 'Score': 'TF-IDF Score'},
             color='Score', color_continuous_scale='Blues')

# Show the plot
fig.show()


# COMMAND ----------

from pyspark.ml.feature import NGram

# Create bigrams
bigram = NGram(n=2, inputCol="tokens_no_stopwords", outputCol="bigrams")
df_bigrams = bigram.transform(df)

# Create trigrams
trigram = NGram(n=3, inputCol="tokens_no_stopwords", outputCol="trigrams")
df_trigrams = trigram.transform(df)

# Show the bigrams and trigrams
df_bigrams.select("bigrams").show(truncate=False)
df_trigrams.select("trigrams").show(truncate=False)


# COMMAND ----------

from collections import Counter

# Flatten the list of bigrams and trigrams
all_bigrams = [bigram for row in df_bigrams.select("bigrams").collect() for bigram in row.bigrams]
all_trigrams = [trigram for row in df_trigrams.select("trigrams").collect() for trigram in row.trigrams]

# Count the frequency of each bigram and trigram
bigram_counts = Counter(all_bigrams)
trigram_counts = Counter(all_trigrams)

# Get the top 10 bigrams and trigrams
top_bigrams = bigram_counts.most_common(10)
top_trigrams = trigram_counts.most_common(10)

# Convert to DataFrames for plotting
top_bigrams_df = pd.DataFrame(top_bigrams, columns=['Bigram', 'Frequency'])
top_trigrams_df = pd.DataFrame(top_trigrams, columns=['Trigram', 'Frequency'])


# COMMAND ----------

import plotly.express as px

# Create a bar chart for the top bigrams
fig_bigram = px.bar(top_bigrams_df, y='Bigram', x='Frequency',
                    title='Top Bigrams Based on Frequency',
                    labels={'Bigram': 'Bigrams', 'Frequency': 'Frequency'},
                    color='Frequency', color_continuous_scale='Blues')

# Show the plot
fig_bigram.show()


# COMMAND ----------

# Create a bar chart for the top trigrams
fig_trigram = px.bar(top_trigrams_df, y='Trigram', x='Frequency',
                     title='Top Trigrams Based on Frequency',
                     labels={'Trigram': 'Trigrams', 'Frequency': 'Frequency'},
                     color='Frequency', color_continuous_scale='Blues')

# Show the plot
fig_trigram.show()