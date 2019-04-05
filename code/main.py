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

#datapaths global vars

testLables = "data/test_labels.csv"
testData = "data/test.csv"
trainData = "data/train.csv"
submission = "data/sample_submission.csv"

def main():
    print("lets do this bois")

if __name__ == '__main__':
    main()