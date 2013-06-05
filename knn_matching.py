from __future__ import division
import praw
from praw.objects import Comment, MoreComments
from itertools import chain
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from utils import get_message_replies, MessageReply, get_submission_reply_pairs, read_raw_data_from_cache, write_data_to_cache
from scipy.spatial.distance import euclidean
from fisher import pvalue
import random
import math
import pickle
import logging
import sys
import settings
import re
import collections
from nltk.stem.porter import PorterStemmer

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))

MIN_WORDS = 2
MAX_COMMENT_LENGTH = 250

class SpellCorrector(object):
    """
    Taken and slightly adapted from peter norvig's post at http://norvig.com/spell-correct.html
    """

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    punctuation = [".", "!", "?", ","]

    def __init__(self):
        self.NWORDS = self.train(self.words(file('big.txt').read()))

    def words(self, text):
        return re.findall('[a-z]+', text.lower())

    def train(self, features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def edits1(self, word):
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
        replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts    = [a + c + b     for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

    def known(self, words): return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        suffix = ""
        for p in self.punctuation:
            if word.endswith(p):
                suffix = p
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        word = max(candidates, key=self.NWORDS.get)
        return word + suffix

class Vectorizer(object):
    def __init__(self):
        self.fit_done = False

    def fit(self, input_text, input_scores):
        self.spell_corrector = SpellCorrector()
        self.stemmer = PorterStemmer()
        input_text = [t + self.generate_new_text(t) for t in input_text]
        self.vectorizer1 = CountVectorizer(ngram_range=(1,2), min_df = 3/len(input_text), max_df=.4)
        self.vectorizer1.fit(input_text)
        self.vocab = self.get_vocab(input_text, input_scores)
        self.vectorizer = CountVectorizer(ngram_range=(1,2), vocabulary=self.vocab)
        self.fit_done = True

    def spell_correct_text(self, text):
        text = text.lower()
        split = text.split(" ")
        corrected = [self.spell_corrector.correct(w) for w in split]
        return corrected

    def generate_new_text(self, text):
        no_punctuation = re.sub("[^A-Za-z0-9]", " ", text.lower())
        no_punctuation = re.sub("\s+", " ", no_punctuation)
        corrected = self.spell_correct_text(no_punctuation)
        corrected = [self.stemmer.stem(w) for w in corrected]
        new = " ".join(corrected)
        return new

    def get_vocab(self, input_text, input_scores):
        train_mat = self.vectorizer1.transform(input_text)
        input_score_med = np.median(input_scores)
        new_scores = [0 if i<=input_score_med else 1 for i in input_scores]
        pvalues = []
        for i in xrange(0,train_mat.shape[1]):
            lcol = np.asarray(train_mat.getcol(i).todense().transpose())[0]
            good_lcol = lcol[[n for n in xrange(0,len(new_scores)) if new_scores[n]==1]]
            bad_lcol = lcol[[n for n in xrange(0,len(new_scores)) if new_scores[n]==0]]
            good_lcol_present = len(good_lcol[good_lcol > 0])
            good_lcol_missing = len(good_lcol[good_lcol == 0])
            bad_lcol_present = len(bad_lcol[bad_lcol > 0])
            bad_lcol_missing = len(bad_lcol[bad_lcol == 0])
            pval = pvalue(good_lcol_present, bad_lcol_present, good_lcol_missing, bad_lcol_missing)
            pvalues.append(pval.two_tail)
        col_inds = list(xrange(0,train_mat.shape[1]))
        p_frame = pd.DataFrame(np.array([col_inds, pvalues]).transpose(), columns=["inds", "pvalues"])
        p_frame.sort(['pvalues'], ascending=True)
        getVar = lambda searchList, ind: [searchList[int(i)] for i in ind]
        vocab = getVar(self.vectorizer1.get_feature_names(), p_frame['inds'][:2000])
        return vocab

    def get_features(self, text):
        if not self.fit_done:
            raise Exception("Vectorizer has not been created.")
        itext=text
        if isinstance(text, list):
            itext = text[0]
        new_text = self.generate_new_text(itext)
        if isinstance(text, list):
            text = [text[0] + new_text]
        else:
            text = text + new_text
        return (self.vectorizer.transform(text).todense())

class KNNCommentMatcher(object):
    def __init__(self, train_data):
        self.train = train_data
        self.vectorizer = Vectorizer()
        self.messages = [t['message'] for t in self.train]
        self.message_scores = [t['message_score'] for t in self.train]

    def fit(self):
        self.vectorizer.fit(self.messages, self.message_scores)
        self.train_mat = self.vectorizer.get_features(self.messages)

    def find_nearest_match(self, text):
        test_vec = np.asarray(self.vectorizer.get_features(text))
        distances = [euclidean(u, test_vec) for u in self.train_mat]
        nearest_match = distances.index(min(distances))
        return nearest_match

    def find_suitable_reply(self, text):
        if isinstance(text, dict):
            text = text['message']
        if not isinstance(text, list):
            text = [text]
        self.validate_reply(text)
        nearest_match = self.find_nearest_match(text)
        raw_data = self.train[nearest_match]
        reply = self.get_highest_rated_comment(raw_data)
        if isinstance(reply, list):
            reply = reply[0]
        reply = " " .join(self.vectorizer.spell_correct_text(reply))
        return self.validate_reply(reply)

    def get_highest_rated_comment(self, raw_data):
        rep_frame =  pd.DataFrame(np.array([raw_data['scores'], raw_data['replies']]).transpose(), columns=["scores", "replies"])
        rep_frame.sort(['scores'], ascending=False)
        rand_int = random.randint(0,min([10, len(raw_data['replies'])-1]))
        return rep_frame['replies'][rand_int]

    def validate_reply(self, reply):
        print reply
        if isinstance(reply, list):
            reply = reply[0]
        if len(reply.split(" "))<MIN_WORDS:
            return None
        elif len(reply)> MAX_COMMENT_LENGTH:
            return None
        elif ".com" in reply or ".net" in reply or "http://" in reply or "www." in reply:
            return None

        for word in settings.BAD_WORD_LIST:
            if word in reply:
                return None

        return reply

def train_knn_matcher(raw_data):
    knn_matcher = KNNCommentMatcher(raw_data)
    knn_matcher.fit()
    return knn_matcher

def test_knn_matcher(knn_matcher, test_data):
    return knn_matcher.find_suitable_reply(test_data)

def cross_validate(input_data, train_function, predict_function, num_folds=3):
    random.seed(1)
    id_seq = list(xrange(0,len(input_data)))
    random.shuffle(id_seq)
    fold_size = int(math.floor(len(id_seq)/num_folds))
    start_index = 0
    all_test_inds = []
    all_test_results = []
    for i in xrange(0, num_folds):
        end_index = int(start_index + fold_size)
        if i==(num_folds-1):
            end_index = len(id_seq)
        test_inds = id_seq[start_index:end_index]
        sel_train = list(xrange(0,start_index)) + list(xrange(end_index,len(id_seq)))
        train_inds = [id_seq[m] for m in sel_train]
        train_data = [input_data[m] for m in train_inds]
        test_data = [input_data[m] for m in test_inds]
        model = train_function(train_data)
        tr = []
        for td in test_data:
            tr.append(predict_function(model, td))
        all_test_inds += test_inds
        all_test_results += tr
        start_index = end_index

    sorted_test_results = [x for (y,x) in sorted(zip(all_test_inds,all_test_results))]
    return sorted_test_results

def test_accuracy(test_results, raw_data):
    correct = []
    for i in xrange(0,len(raw_data)):
        if test_results[i] in raw_data[i]['replies']:
            correct.append(True)
        else:
            correct.append(False)
    return correct

if __name__ == '__main__':
    message_replies = get_message_replies(subreddit = "funny", max_replies= 500, submission_count = 300, min_reply_score = 20)

    raw_data = list([mr.get_raw_data() for mr in message_replies])

    raw_data = write_data_to_cache(raw_data, "raw_data_cache.p")

    test_results = cross_validate(raw_data,train_knn_matcher, test_knn_matcher)

    for i in xrange(0,len(test_results)):
        print i
        print raw_data[i]['message']
        print test_results[i]
        print "------------------"





