import numpy as np
import csv
import math
import re

from ast import literal_eval

from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

GENRES = {
    "Blues" :1.,
    "Country":2.,
    "Electronic":3.,
    "International":4.,
    "Jazz":5.,
    "Latin":6.,
    "Pop/Rock":7.,
    "R&B":8.,
    "Rap":9.,
    "Reggae":10.,
}

def tryeval(val):

    if val is '' or val == 'nan' or val ==  "None":
        return 0
    else:
        try:
            val = literal_eval(val)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return val


def processData(file):
    lengths = [len(x) for x in file]
    avg = max(set( lengths ), key=lengths.count)
    file = [ x for x in file if len(x) is avg  ]
    X = map(lambda x: x[:-1], file)
    X = map(lambda x: x[:15], X)
    Y = map(lambda x: x[-1], file)
    Y = map(lambda x: GENRES[x],Y)
    for x in X:
        x = map(tryeval,x)
    for b in X:
        for i,a in enumerate(b):
            if a is None:
                b[i] = 0
    return (X,Y)


if __name__ == "__main__":
    genres        = [] #hardcoded all possible genres
    train_size    = 5000
    file          = list(csv.reader(open("out.csv",'rb'),delimiter='|'))[1:]
    processedData = processData(file) #(X,Y)
    trainingSet   = (processedData[0][:train_size],processedData[1][:train_size])
    testingSet    = (processedData[0][train_size:],processedData[1][train_size:])

    X_train, X_test, y_train, y_test = train_test_split(processedData[0], processedData[1], test_size=.5)
#classify
    svr = OneVsRestClassifier(SVR(kernel='rbf'))
    svr.fit(trainingSet[0],trainingSet[1])
    predicted = svr.predict(testingSet[0])

#accuracy
    correct =0.
    for x,y in zip(testingSet[1],predicted):
        print "expected {0} got {1}".format( x,y)
        if x == y:
            correct +=1.
    print "percent accuracy {0}".format(correct/len(predicted))



