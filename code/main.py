"""
Toxic comment classifier

Taking from the document we will likely use SVM to classify comments

SCI-KIT SVM documentation
https://scikit-learn.org/stable/modules/svm.html#svm-classification

sci-kit svm example with visualization
https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/

Project base
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


sci-kit naive bayes blog post with examples
https://blog.sicara.com/naive-bayes-classifier-sklearn-python-example-tips-42d100429e44

http://www.nltk.org/

https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a?gi=a4f8d9daa68b

https://www.analyticsvidhya.com/blog/2015/10/6-practices-enhance-performance-text-classification-model/

http://blog.chapagain.com.np/machine-learning-sentiment-analysis-text-classification-using-python-nltk/
"""
#imports
from sklearn import svm
import pandas as pd
import numpy as np
from nltk import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
import nltk.classify.util, nltk.metrics
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC

#datapaths global vars

testLables = "../data/test_labels.csv"
testData = "../data/test.csv"
trainData = "../data/train.csv"
submission = "../data/sample_submission.csv"
swearWords = "../data/swearWords.csv"

stopwords = set(stopwords.words('english'))
#line for debug mode flag
debug_mode = False
NB_Mode = False
SVM_Mode = True


def word_feats(words):    
    return dict([(word, True) for word in words])

#Filters stop words out of a list of words
def processwords(words):
    wordsFiltered = []

    for w in words:
        if w not in stopwords:
            wordsFiltered.append(w)
    return wordsFiltered

#Assembles a feature list usable with NLTK for a given raw dataset
def AssembleFeatureList(dataset):
    featureList = list()
    for data in dataset:
        i = 2
        label = 0
        for i in range(2,7):
            if (data[i] == 1):
                label = 1
            i += 1
            if i == 8:
                break
        filteredComment = processwords(data[1].split())
        testtuple = (word_feats(filteredComment), label)
        featureList.append(testtuple)

    return featureList

#Assembles test feature list that accounts for data that is to be excluded in testing
def AssembleTestFeatureList(dataset):
    featureList = list()
    for data in dataset:
        i = 2
        label = 0
        for i in range(2,7):
            if (data[i] == 1):
                label = 1
            if (data[i] == -1):
                label = -1
                break
            i += 1
            if i == 8:
                break
        if (label == -1):
            continue
        filteredComment = processwords(data[1].split())
        testtuple = (word_feats(filteredComment), label)
        featureList.append(testtuple)

    return featureList

def main():
    csvTestLabels = pd.read_csv(testLables) #read in csv
    csvTraining = pd.read_csv(trainData)
    csvTest = pd.read_csv(testData)
    #swearWordsList = pd.read_csv(swearWords)



    IDs = csvTraining["id"]
    comments = csvTraining["comment_text"]
    toxic_status = csvTraining["toxic"]
    severe_toxic_status = csvTraining["severe_toxic"]
    obscene_status = csvTraining["obscene"]
    threat_status = csvTraining["threat"]
    insult_status = csvTraining["insult"]
    identity_hate_status = csvTraining["identity_hate"]

    #zip training data into list of tuples
    parsed_train_data = list()
    for ids,comment,tx,svtx,ob,tr,ins,idh in zip(IDs,comments,toxic_status,severe_toxic_status,obscene_status,threat_status,insult_status,identity_hate_status):
        parsed_train_data.append((ids,comment,tx,svtx,ob,tr,ins,idh))

    IDs = csvTestLabels["id"]
    comments = csvTest["comment_text"]
    toxic_status = csvTestLabels["toxic"]
    severe_toxic_status = csvTestLabels["severe_toxic"]
    obscene_status = csvTestLabels["obscene"]
    threat_status = csvTestLabels["threat"]
    insult_status = csvTestLabels["insult"]
    identity_hate_status = csvTestLabels["identity_hate"]
    
    #zip data into tuple list of parsed label
    parsed_label_data = list() #parsed out list that contains all the test label data in a list of tuples
    for ids,comment,tx,svtx,ob,tr,ins,idh in zip(IDs, comments,toxic_status,severe_toxic_status,obscene_status,threat_status,insult_status,identity_hate_status):
        parsed_label_data.append((ids,comment,tx,svtx,ob,tr,ins,idh))


    IDs = csvTest["id"]
    comments = csvTest["comment_text"]


    #zip training data into list of tuples
    parsed_test_data = list()
    for ids,comment in zip(IDs,comments):
        parsed_test_data.append((ids,comment))
    
    #this debug print shows the data if enabled for verification
    if debug_mode:
        i = 0
        for data in parsed_label_data:
            print(data)
            if i == 5:
                break
            i+=1
        print()
        print()
        print()

    if debug_mode:
        i=0
        for data in parsed_train_data:
            print(data,"\n")
            if i==5:
                break
            i+=1
        print()
        print()
        print()

    if debug_mode:
        i=0
        for data in parsed_test_data:
            print(data,"\n")
            if i==5:
                break
            i+=1
        print()
        print()
        print()




    
    if NB_Mode:
        trainingFeatures = AssembleFeatureList(parsed_train_data)

        classifier = NaiveBayesClassifier.train(trainingFeatures)
        accuracy = nltk.classify.util.accuracy(classifier, trainingFeatures)
        print("Training Pass Naive Bayes")
        print("Accuracy: ", accuracy)
        print("Testing Pass Naive Bayes")

        testingFeatures = AssembleTestFeatureList(parsed_label_data)
        #print(testingFeatures)
        testcount = 0.0
        hitcount = 0.0
        for features,labels in testingFeatures:
            testcount += 1
            result = classifier.classify(features)
            if result == labels:
                hitcount += 1
        
        accuracy = hitcount/testcount
        print("Accuracy: ",accuracy)

    if SVM_Mode:
        trainingFeatures = AssembleFeatureList(parsed_train_data)

        classifier = SklearnClassifier(LinearSVC())
        accuracy = nltk.classify.util.accuracy(classifier, trainingFeatures)
        print("Training Pass Linear SVM")
        print("Accuracy: ", accuracy)
        print("Testing Pass Linear SVM")

        testingFeatures = AssembleTestFeatureList(parsed_label_data)
        #print(testingFeatures)
        testcount = 0.0
        hitcount = 0.0
        for features,labels in testingFeatures:
            testcount += 1
            result = classifier.classify(features)
            if result == labels:
                hitcount += 1
        
        accuracy = hitcount/testcount
        print("Accuracy: ",accuracy)

if __name__ == '__main__':
    main()



    #We need to split the data across 7 feature sets
    #           negfeats = [(featx(f), 'neg') for f in word_split(negdata)]
    #           posfeats = [(featx(f), 'pos') for f in word_split(posdata)]
    #This is from the article here: http://blog.chapagain.com.np/machine-learning-sentiment-analysis-text-classification-using-python-nltk/
    #
    #We need ot split the data into 7 feature sets 
    #The 7 feature sets will be processed out of the features present in the data
    #If a comment is marked as both toxic and severe-toxic it will be included in both of those feature sets
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    