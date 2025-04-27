import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt

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
theTree = DecisionTreeClassifier(max_depth=30)
theTree = theTree.fit(headlines_train, isClickbait_train)

# Validate model
isClickbait_pred = theTree.predict(headlines_validate)
print("Accurately predicting clickbait percentage (validation):", metrics.accuracy_score(isClickbait_validate, isClickbait_pred) * 100)

# Test model
isClickbait_test_pred = theTree.predict(headlines_test)
print("Accurately predicting clickbait percentage (test):", metrics.accuracy_score(isClickbait_test, isClickbait_test_pred) * 100)

# Plot tree
figure = plt.figure(figsize=(25, 20))
tree.plot_tree(theTree, feature_names=countVectorizer.get_feature_names_out(), max_depth=3)
figure.savefig("figs\decision-tree-figure.png")

# Plot the whole tree
tree.plot_tree(theTree, feature_names=countVectorizer.get_feature_names_out())
figure.savefig("figs\whole-decision-tree-figure.png")


# Test model with new dataset
clickbaitDFTest = pd.read_csv('data/clickbait-train2.csv')

bagOfWordsTest = countVectorizer.transform(clickbaitDFTest.loc[:, 'title'])
bagOfWordsDFTest = pd.DataFrame(bagOfWordsTest.toarray(), columns=countVectorizer.get_feature_names_out())

isClickbait_test_2 = clickbaitDFTest['label']
isClickbait_test_2_temp = isClickbait_test_2.to_numpy()
isClickbait_test_2_temp = np.where(isClickbait_test_2 == 'clickbait', 1, 0)
isClickbait_test_2 = pd.DataFrame(isClickbait_test_2_temp, columns=['label'])
headlines_test_2 = bagOfWordsDFTest

isClickbait_test_pred_2 = theTree.predict(headlines_test_2)
print("Accurately predicting clickbait percentage (test 2):", metrics.accuracy_score(isClickbait_test_2, isClickbait_test_pred_2) * 100)