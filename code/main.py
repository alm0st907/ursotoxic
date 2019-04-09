"""
Toxic comment classifier

Taking from the document we will likely use SVM to classify comments

SCI-KIT SVM documentation
https://scikit-learn.org/stable/modules/svm.html#svm-classification

sci-kit svm example with visualization
https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/


sci-kit naive bayes blog post with examples
https://blog.sicara.com/naive-bayes-classifier-sklearn-python-example-tips-42d100429e44
"""
#imports
from sklearn import svm
import pandas as pd

#datapaths global vars

testLables = "data/test_labels.csv"
testData = "data/test.csv"
trainData = "data/train.csv"
submission = "data/sample_submission.csv"

#line for debug mode flag
debug_mode = True

def main():
    csvTestLabels = pd.read_csv(testLables) #read in csv
    csvTraining = pd.read_csv(trainData)
    csvTest = pd.read_csv(testData)

    IDs = csvTestLabels["id"]
    toxic_status = csvTestLabels["toxic"]
    severe_toxic_status = csvTestLabels["severe_toxic"]
    obscene_status = csvTestLabels["obscene"]
    threat_status = csvTestLabels["threat"]
    insult_status = csvTestLabels["insult"]
    identity_hate_status = csvTestLabels["identity_hate"]
    
    #zip data into tuple list of parsed label
    parsed_label_data = list() #parsed out list that contains all the test label data in a list of tuples
    for ids,tx,svtx,ob,tr,ins,idh in zip(IDs,toxic_status,severe_toxic_status,obscene_status,threat_status,insult_status,identity_hate_status):
        parsed_label_data.append((ids,tx,svtx,ob,tr,ins,idh))

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

    if debug_mode:
        i=0
        for data in parsed_train_data:
            print(data,"\n")
            if i==5:
                break
            i+=1
        print()

    if debug_mode:
        i=0
        for data in parsed_test_data:
            print(data,"\n")
            if i==5:
                break
            i+=1
        print()

if __name__ == '__main__':
    main()