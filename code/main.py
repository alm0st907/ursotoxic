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
    # testTuples = (csvTestLabels["id"],csvTestLabels["toxic"],csvTestLabels["obscene"],csvTestLabels["threat"],csvTestLabels["insult"],csvTestLabels["identity_hate"])

    IDs = csvTestLabels["id"]
    toxic_status = csvTestLabels["toxic"]
    severe_toxic_status = csvTestLabels["severe_toxic"]
    obscene_status = csvTestLabels["obscene"]
    threat_status = csvTestLabels["threat"]
    insult_status = csvTestLabels["insult"]
    identity_hate_status = csvTestLabels["identity_hate"]

    parsed_label_data = list() #parsed out list that contains all the test label data in a list of tuples

    for ids,tx,svtx,ob,tr,ins,idh in zip(IDs,toxic_status,severe_toxic_status,obscene_status,threat_status,insult_status,identity_hate_status):
        parsed_label_data.append((ids,tx,svtx,ob,tr,ins,idh))
    
    #this debug print shows the data if enabled for verification
    if debug_mode:
        i = 0
        for data in parsed_label_data:
            print(data)
            if i == 5:
                break
            i+=1

    print()

if __name__ == '__main__':
    main()