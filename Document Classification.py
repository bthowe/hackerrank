import pandas as pd
import numpy as np
import itertools

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

def load_data():
    df = pd.read_csv('trainingdata.txt')
    name = df.columns
    # y = df[name[0]].map(lambda x: x[0])
    y = df[name[0]].str[0].map(lambda x: int(x))
    X = df[name[0]].str[2:]
    return X.values, y.values

def lemmatize(articles, type):

    if type=='porter':
        porter = PorterStemmer()
        stem = [' '.join([porter.stem(word.lower()) for word in str(article).split()]) for article in articles]

    if type=='snowball':
        snowball = SnowballStemmer('english')
        stem = [' '.join([snowball.stem(word.lower()) for word in str(article).split()]) for article in articles]

    if type=='lemma':
        wordnet = WordNetLemmatizer()
        stem = [' '.join([wordnet.lemmatize(word.lower()) for word in str(article).split()]) for article in articles]

    if type==None:
        stem = articles

    return stem

def tfidf_fun(stem, sw):
    tfidf_vectorizer = TfidfVectorizer(stop_words=sw)
    tfidf = tfidf_vectorizer.fit(stem)
    return tfidf

def classify(X, y):
    naive_bayes = MultinomialNB()
    gsCV = GridSearchCV(naive_bayes, {'alpha': [.01, .02, .03, .04, .05, .06, .07, .08, .09, .10]}, scoring = 'accuracy', n_jobs = -1)
    gsCV.fit(X, y)
    print gsCV.best_params_
    print gsCV.grid_scores_

    # return naive_bayes.fit(X, y)

if __name__=="__main__":
    X, y = load_data()

    # stopwords = ['english', None]
    # stopwords = ['english']
    # # prep = ['snowball', 'lemma', 'porter', None]
    # prep = [None]
    # for i in itertools.product(prep, stopwords):
    #     print '{0}: '.format(i)
    #     stem = lemmatize(X, i[0])
    #     tfidf = tfidf_fun(stem, i[1])
    #     classify(tfidf.transform(stem), y)

    # pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('mnb', MultinomialNB())])
    # param_dict = {'tfidf__stop_words': ['english'],
    #     'tfidf__max_df': [.25],
    #     # 'tfidf__min_df': [0, .0125, .025],
    #     'tfidf__min_df': [0],
    #     'tfidf__max_features': [1250],
    #     'mnb__alpha': [.095, .1, .11, .15, .2]}
    # gsCV = GridSearchCV(pipeline, param_dict, scoring='accuracy', n_jobs = -1)
    # gsCV.fit(X, y)
    # print gsCV.best_params_
    # print gsCV.best_score_


    pipe = Pipeline([('tfidf', TfidfVectorizer(stop_words = 'english', min_df = 0, max_df = .25, max_features = 1250)), ('mnb', MultinomialNB(alpha = .095))])
    pipe.fit(X, y)


    num = int(raw_input('Enter Now!'))
    x_test = []
    for i in range(num):
        x_test.append(raw_input())
    predictor = pipe.predict(x_test)
    for i in range(num):
        print predictor[i]

# num = int(raw_input("Enter Number: "))
# for i in xrange(num):
#     string = raw_input()
#     stem_test = lemmatize(string)
#     print nb.predict(tfidf.transform([stem_test]))
#     print nb.predict(tfidf.transform([string]))





    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    # scores = []
    # stemmer_type = ['porter', 'snowball', 'lemma']
    # for stemmer in stemmer_type:
    #     stem_train = lemmatize(X_train, stemmer)
    #     tfidf = tfidf_fun(stem_train)
    #     nb = classify(tfidf.transform(stem_train), y_train)
    #
    #     stem_test = lemmatize(X_test, stemmer)
    #     y_pred = nb.predict(tfidf.transform(stem_test))
    #
    #     scores.append(accuracy_score(y_pred, y_test))
    #
    # print "\n\n\nThe accuracy scores for the different featurization techniques are...\n"
    # for i in xrange(3):
    #     print "{0}: {1}".format(stemmer_type[i], scores[i])
