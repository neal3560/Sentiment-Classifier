import numpy as np
import string
import random
import math
import re
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

def read(dataFile):
    train_data = []
    file = open(dataFile, encoding='utf8')
    for l in file:
        train_data.append(l)
    return train_data

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    #print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def read_files(tarfname):
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name

    class Data: pass
    sentiment = Data()
    #print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    #print(len(sentiment.train_data))

    #print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    #print(len(sentiment.dev_data))
    #print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer
    sentiment.count_vect = CountVectorizer()
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def read_unlabeled(tarfname, sentiment):
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []

    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name

    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)


    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def train_classifier(X, y, c):
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(C=c,random_state=0, solver='lbfgs', max_iter=10000)
    cls.fit(X, y)
    return cls

def tokenize(dataset, mode):
    termsCount = defaultdict(int)
    documentCount = defaultdict(int)

    for d in dataset:
        lower = ''.join([c for c in d.lower()])
        r = re.split(r'\W+', lower)
        voc = set()
        for i in range(len(r)):
            w = r[i]
            voc.add(w)
            termsCount[w] += 1

            if i != len(r) - 1 and mode == 0:
                b = r[i] + ' ' + r[i + 1]
                voc.add(b)
                termsCount[b] += 1

        for t in voc:
            documentCount[t] += 1

    terms = [t for t in termsCount]
    termsId = dict(zip(terms, range(len(terms))))

    idf = defaultdict(float)
    for t in terms:
        idf[t] = math.log(len(dataset) / (documentCount[t]))
    return (termsId, idf)

def tf(document, mode):
    tf_table = defaultdict(int)
    lower = ''.join([c for c in document.lower()])
    r = re.split(r'\W+', lower)
    for i in range(len(r)):
        w = r[i]
        tf_table[w] += 1

        if i != len(r) - 1 and mode == 0:
            b = r[i] + ' ' + r[i + 1]
            tf_table[b] += 1

    for t in tf_table:
        tf_table[t] = math.log(1 + tf_table[t])
    return tf_table

def tfidf_matrix(dataset, termsId, idf, mode):
    row = []
    col = []
    data = []
    index = 0
    for document in dataset:
        tf_table = tf(document, mode)
        for t in tf_table:
            if t in termsId:
                row.append(index)
                col.append(termsId[t])
                data.append(tf_table[t] * idf[t])
        index += 1
    return csr_matrix((data, (row, col)), shape=(len(dataset), len(termsId)))

class TC:
    def __init__(self, mode=0):
        self.mode = mode
        if mode == 0:
            tarfname = "./text_classifier/data/sentiment.tar.gz"
            sentiment = read_files(tarfname)
            train_data = sentiment.train_data
            train_y = sentiment.trainy
        else:
            train_data = read("./text_classifier/data/datafile.txt")
            train_y = [int(label) for label in read("./text_classifier/data/label.txt")]
        self.wordsId, self.idf_table = tokenize(train_data, mode)
        trainX = tfidf_matrix(train_data, self.wordsId, self.idf_table, mode)
        self.cls = train_classifier(trainX, train_y, 0.1)

    def classify(self, review):
        test_vector = tfidf_matrix([review], self.wordsId, self.idf_table, self.mode)[0]
        lower = ''.join([c for c in review.lower()])
        r = re.split(r'\W+', lower)
        uni = []
        bi = []
        for i in range(len(r)):
            w = r[i]
            uni.append(w)
            if i != len(r) - 1 and self.mode == 0:
                b = r[i] + ' ' + r[i + 1]
                bi.append(b)
        """
        print("========unigram=======")
        for w in uni:
            if w in self.wordsId:
                print(w, round(self.cls.coef_[0][self.wordsId[w]],2))
            else:
                print(w, "is unseen")
        print("========bigram========")
        for w in bi:
            if w in self.wordsId:
                print(w, round(self.cls.coef_[0][self.wordsId[w]],2))
            else:
                print(w, "is unseen")
        print("confidence score:", self.cls.decision_function(test_vector)[0])
        """
        label = self.cls.predict(test_vector)[0]
        confidence = round(self.cls.predict_proba(test_vector)[0].max(), 2)
        return (label, confidence)
