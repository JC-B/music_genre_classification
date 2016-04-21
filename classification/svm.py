#!/usr/bin/env python
import numpy as np
import csv
import math
import re
import os
import scipy.sparse as sp
import pylab as pl
import random

from ast import literal_eval

from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import *
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, f_regression,f_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

GENRES = {
    "Blues"              :1.,
    "Country"            :2.,
    "Electronic"         :3.,
    "International"      :4.,
    "Jazz"               :5.,
    "Latin"              :6.,
    "Pop/Rock"           :7.,
    "R&B"                :8.,
    "Rap"                :9.,
    "Reggae"             :10.,
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
    file          = list(csv.reader(open("../out222.csv",'rb'),delimiter='|'))[1:]
    processedData = processData(file) #(X,Y)
    X_train, X_test, y_train, y_test = train_test_split(processedData[0], processedData[1], test_size=.25)

#classify


#This is to figure out what parameters are the best
#    anova_filter = SelectKBest(f_regression, k=2)
#    svr = OneVsRestClassifier(SVR(kernel='rbf',C=1,epsilon=.01,gamma=.01))
#    pipeline = Pipeline([
#        ("anova",anova_filter),
#        ("svr",svr)
#    ])
#
#    parameters = {
#        "anova__k":[2,5,10,15],
#        "anova__k":[2,3,4,5,6,7,8,9,10]
#        "anova__score_func":[f_regression,f_classif],
#        "svr__estimator__C": [1,2,5,10,100,100],
#        "svr__estimator__epsilon":[.01,.02,.05,.1,.5],
#        "svr__estimator__gamma":[.01,.05,.25,.5,.9]
#    }
#    model_tuner = GridSearchCV(pipeline, parameters)
#    model_tuner.fit(X_train, y_train)
#    print model_tuner.best_score_
#    print model_tuner.best_params_

#This is the application of the results from above
    anova_filter = SelectKBest(f_regression, k=2)
    svr = OneVsRestClassifier(SVR(kernel='rbf',C=1000,epsilon=.01,gamma=.01),n_jobs=2)
    svr.fit(X_train,y_train)
    predicted = svr.predict(X_test)

#accuracy
    print classification_report(y_test,predicted, target_names=["Blues", "Country", "Electronic", "International", "Jazz", "Latin", "Pop/Rock", "R&B", "Rap", "Reggae"]) 
    cm = confusion_matrix(y_test,predicted)

    print "confusion matrix:"
    print cm

#    pl.matshow(cm)
#    pl.title("confusion_matrix for onevsrest classifer")
#    pl.colorbar()
#    pl.show()



    writeFile = open("results.txt", 'wb')
    correct =0.
    for x,y in zip(y_test,predicted):
        writeFile.write( "expected {0} got {1}".format(x, y))
        if x == y:
            correct +=1.
    writeFile.write("percent accuracy {0}".format(correct/len(predicted)))
    writeFile.close()



