import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Pre-process data
clickbaitDF = pd.read_csv('data/clickbait-train1.csv')

countVectorizer = CountVectorizer()
bagOfWords = countVectorizer.fit_transform(clickbaitDF.loc[:, 'headline'])
bagOfWordsDF = pd.DataFrame(bagOfWords.toarray(), columns=countVectorizer.get_feature_names_out())

isClickbait = clickbaitDF['clickbait']
headlines = bagOfWordsDF

isClickbait_train, isClickbait_temp, headlines_train, headlines_temp = train_test_split(isClickbait, headlines, test_size=0.3, random_state=440)
isClickbait_validate, isClickbait_test, headlines_validate, headlines_test = train_test_split(isClickbait_temp, headlines_temp, test_size=0.5, random_state=440)

# Train model
naiveBayesMod = MultinomialNB()
naiveBayesMod = naiveBayesMod.fit(headlines_train, isClickbait_train)

# Validate model
isClickbait_pred = naiveBayesMod.predict(headlines_validate)
print("Accurately predicting clickbait percentage (validation):", metrics.accuracy_score(isClickbait_validate, isClickbait_pred) * 100)

# Test model
isClickbait_test_pred = naiveBayesMod.predict(headlines_test)
print("Accurately predicting clickbait percentage (test):", metrics.accuracy_score(isClickbait_test, isClickbait_test_pred) * 100)

# Test model with new dataset
clickbaitDFTest = pd.read_csv('data/clickbait-train2.csv')

bagOfWordsTest = countVectorizer.transform(clickbaitDFTest.loc[:, 'title'])
bagOfWordsDFTest = pd.DataFrame(bagOfWordsTest.toarray(), columns=countVectorizer.get_feature_names_out())

isClickbait_test_2 = clickbaitDFTest['label']
isClickbait_test_2_temp = isClickbait_test_2.to_numpy()
isClickbait_test_2_temp = np.where(isClickbait_test_2 == 'clickbait', 1, 0)
isClickbait_test_2 = pd.DataFrame(isClickbait_test_2_temp, columns=['label'])
headlines_test_2 = bagOfWordsDFTest

isClickbait_test_pred_2 = naiveBayesMod.predict(headlines_test_2)
print("Accurately predicting clickbait percentage (test 2):", metrics.accuracy_score(isClickbait_test_2, isClickbait_test_pred_2) * 100)