import matplotlib.pyplot as plt
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
from sklearn.preprocessing  import *
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, f_regression,f_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

from operator import itemgetter

GENRES = {
    "Blues"              :0.,
    "Country"            :1.,
    "Electronic"         :2.,
    "International"      :3.,
    "Jazz"               :4.,
    "Latin"              :5.,
    "Pop/Rock"           :6.,
    "R&B"                :7.,
    "Rap"                :8.,
    "Reggae"             :9.,
    "test"               :10.
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
#        if type(val) is float:
#            val = int(round(val))
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
    bin = Binarizer()
    X = bin.fit_transform(X)
    X = map(lambda x: map(lambda y: float(y), x), X)
    return (X,Y)


if __name__ == "__main__":
    file          = list(csv.reader(open("out2.csv",'rb'),delimiter='|'))[1:]
    processedData = processData(file) #(X,Y)
    X_train, X_test, y_train, y_test = train_test_split(processedData[0], processedData[1], test_size=.25)

    with open("out2.csv", "rb") as f:
        reader = csv.reader(f)
        i = reader.next()
        rest = [row for row in reader]
#classify


#This is to figure out what parameters are the best
#    anova_filter = SelectKBest(f_regression, k=2)
#    svr = OneVsRestClassifier(SVR(kernel='rbf'))
#    pipeline = Pipeline([
#        ("anova",anova_filter),
#        ("svr",svr)
#    ])
#
#    parameters = {
#        "anova__k":[2, 5, 10, 15],
#        "anova__k":[2, 3, 4, 5, 6, 7, 8, 9, 10]
#        "anova__score_func":[f_regression, f_classif],
#        "svr__estimator__C": [1, 2, 5, 10, 100, 1000],
#        "svr__estimator__epsilon":[.01, .02, .05, .1, .5],
#        "svr__estimator__gamma":[.01, .05, .25, .5, .9]
#    }
#    model_tuner = GridSearchCV(pipeline, parameters)
#    model_tuner.fit(X_train, y_train)
#    print model_tuner.best_score_
#    print model_tuner.best_params_

#This is the application of the results from above
    anova_filter = SelectKBest(f_regression, k=2)
#    X_train = CCA(n_components=38).fit(X_train,y_train).transform(X_train)
#    X_train = PCA(n_components=38).fit_transform(X_train)

    X_train = anova_filter.fit_transform(X_train,y_train)

    print "======"
    print sorted(zip(anova_filter.scores_,i),key=lambda x: x[0])
    print "======"

    onevsrest = SVC(kernel='rbf',C=10,gamma=.01).fit(X_train,y_train)
    onevsone = SVR(kernel='rbf',C=10,epsilon=.01,gamma=.01).fit(X_train,y_train)










    X = X_train
    y = y_train

    h = .02  # step size in the mesh

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVR',"SVC"]


    for i, clf in enumerate((onevsone,onevsrest)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()
