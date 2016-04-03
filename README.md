# music_genre_classification
Project for music genre classification for datamining


## [Yajie Hu and Mitsunori Ogihara Million Song Dataset Genre Tags](http://web.cs.miami.edu/home/yajiehu/resource/genre/) ##
### General ###

This genre tag is added to a tab delemeted file that is a 1% subset of the dataset in the file combined.txt. This is a good place to start testing classification algorithms.

### Possible Genres ###
- Blues
- Country
- Electronic
- International
- Jazz
- Latin
- Pop/Rock
- R&B
- Rap
- Reggae

> Yajie Hu and Mitsunori Ogihara, “Genre Classification for Million Song Dataset Using Confidence-Based Classifiers Combination”, in Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval, Portland, Oregon, USA, 2012, pages 1083 - 1084.


cleanCSV.py
----------
### General ###
This script allows for pruning of features from our sample data set.

### How to run ###
Add all column names you want to prune to the array in cleanCSV.py

`python cleanCSV.py <input file> <output file>`

classification/svm.py
---------------------
### Prereqs ###
You need to have the follow libraries installed:
- sklearn
- panda
- numpy

### How to run###
`python classification/svm.py`
