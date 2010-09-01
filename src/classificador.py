# encoding: UTF-8
import os
from lxml import etree
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import re
import unicodedata
from soccers_sentences import read_files
from nltk.classify import apply_features
from random import shuffle

def prepare_text(words):
    if not words:
        words = ""
    #retira acentos
    #words = unicodedata.normalize("NFKD", words.decode('utf-8')).encode('ascii', 'ignore')
    
    # retira espaços desnecessarios
    words = re.sub(r"(\s|\W)+", " ", words.lower())
    # corrige palavra gol
    #words = re.sub(r"(g)+(o)+(l+)", "gol", words)
    return words

_get_features = None

def call_feature(w):
    return _get_features(w)

def features_all_words(sentence):
    if sentence == None:
        sentence = ""
    
    features = {}
    for w in prepare_text(sentence).split(' '):
        features[w] = True
    return features

def select_features_best_words(sentence):
    if sentence == None:
        sentence = ""
    features = {}
    for w in sentence.split(' '):
        if w in select_features_best_words._best:
            features[w] = True
    return features

def train_set(featuresset, factor):
    partial = {}
    tam = 0
    for c in featuresset:
        sentences = featuresset[c]
        n = int(len(sentences)*factor)
        partial[c] = sentences[n:]
        tam += len(partial[c])
    print "train set: %d" % tam
    return partial

def test_set(featuresset, factor):
    partial = {}
    tam = 0
    for c in featuresset:
        sentences = featuresset[c]
        n = int(len(sentences)*factor)
        partial[c] = sentences[:n]
        tam += len(partial[c])
    print "test set: %d" % tam
    return partial

def extract_features_and_run(featuresset, feature_func, factor):
    print "preparing..."
    train = [(feature_func(s), classe) for classe in train_set(featuresset, factor) for s in featuresset[classe]]
    
    print "training..."
    classifier = NaiveBayesClassifier.train(train)
    train = None # libera a memória
    #classifier.show_most_informative_features(10)
    
    print "accuracying..."
    test = [(feature_func(s), classe) for classe in test_set(featuresset, factor) for s in featuresset[classe]]
    print 'accuracy:', nltk.classify.util.accuracy(classifier, test)
    return classifier

def use_best_features(classifier, informacoes, n):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for classe in classifier._labels:
        sentences = informacoes[classe]
        for sentence in sentences:
            for word in features_all_words(sentence):
                word_fd.inc(word)
                label_word_fd[classe].inc(word)

    total_word_count = 0
    for classe in informacoes.keys():
        total_word_count += label_word_fd[classe].N()

    word_scores = {}
    for word, freq in word_fd.iteritems():
        word_scores[word] = 0
        for classe in informacoes:
            word_scores[word] += BigramAssocMeasures.chi_sq(label_word_fd[classe][word],
            (freq, label_word_fd[classe].N()), total_word_count)

    best = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:n]
    bestwords = set([w for w, s in best])
    return bestwords

informacoes = read_files()
print "classes: ", informacoes.keys()

# reduz e embaaralha dataset
#for classe in informacoes:
    #shuffle(informacoes[classe])
#    informacoes[classe] = informacoes[classe][:len(informacoes[classe])/10]


classifier = extract_features_and_run(informacoes, features_all_words, 0.25)

for num_words in [5,10,30,50,80,100, 200, 300, 400, 500, 800, 900, 1000, 3000, 5000, 10000]:
	print "calculando com num_words = %d ----------------------" % num_words
	select_features_best_words._best = use_best_features(classifier, informacoes, num_words)
	classifier2 = extract_features_and_run(informacoes, select_features_best_words, 0.25)






