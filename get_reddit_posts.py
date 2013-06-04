from __future__ import division
import praw
from praw.objects import Comment, MoreComments
from itertools import chain
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

MAX_REPLIES = 100

def get_submission_reply_pairs(submission):
    message_replies = []
    forest_comments = [c for c in submission.comments if isinstance(c, Comment)]
    while len(forest_comments)>0 and len(message_replies)<MAX_REPLIES:
        actual_replies = []
        for comment in forest_comments:
            replies = comment.replies
            actual_replies = []
            for reply in replies:
                if isinstance(reply, MoreComments):
                    mc_replies = reply.comments()
                    for mc_reply in mc_replies:
                        if isinstance(mc_reply, Comment):
                            actual_replies.append(mc_reply)
                elif isinstance(reply, Comment):
                    actual_replies.append(reply)
            reply_text = [ar.body for ar in actual_replies]
            scores = [ar.score for ar in actual_replies]
            if len(reply_text)>0:
                message_replies.append(MessageReply(comment.body, comment.score, reply_text, scores))
        forest_comments = [ar for ar in actual_replies if isinstance(actual_replies, Comment)]
    return message_replies

class MessageReply:
    def __init__(self, message, message_score, replies, scores):
        self.message = message
        self.message_score = message_score
        self.replies = replies
        self.scores = scores

    def get_table_data(self):
        rows = []
        for i in xrange(0,len(self.replies)):
            rows.append([self.message, int(self.message_score), self.replies[i], int(self.scores[i])])
        return rows

def get_message_replies():
    r = praw.Reddit(user_agent='comment_matcher by /u/vikparuchuri github.com/VikParuchuri/comment_matcher/')
    subreddit = r.get_subreddit('funny')

    message_replies = []
    for submission in subreddit.get_hot(limit=10):
        message_replies += get_submission_reply_pairs(submission)
        if len(message_replies)>=MAX_REPLIES:
            break
    return message_replies

def subtract_if_not_zero(input_val, subtract_num):
    if input_val==0:
        return input_val

    return input_val - subtract_num

def create_match_matrix(table_data, pd_frame):

    mean_score = np.mean(pd_frame['reply_score'])

    vectorizer = CountVectorizer(ngram_range=(1,2), min_df = 2/len(table_data), max_df=.6)
    all_text = list(set([t[0] for t in table_data] + [t[2] for t in table_data]))
    vectorizer.fit(all_text)

    count_row = vectorizer.transform([table_data[0][0], table_data[1][0]])
    match_mat = np.empty((count_row.shape[1], count_row.shape[1]))

    for z in xrange(0,pd_frame.shape[0]):
        row=pd_frame.iloc[z]
        message_count = np.asarray(vectorizer.transform([row['message']]).todense())[0]
        reply_count = np.asarray(vectorizer.transform([row['reply']]).todense())[0]
        for i in xrange(0,message_count.shape[0]):
            if message_count[i]==0:
                continue
            else:
                match_mat[i] +=  [subtract_if_not_zero(m, mean_score) for m in ((reply_count) * row['reply_score']).tolist()]
    return match_mat, vectorizer

def find_input_match_vec(input_text, match_mat, vectorizer):
    if not isinstance(input_text, list):
        input_text = [input_text]
    count_text = np.asarray(vectorizer.transform(input_text).todense())[0]
    match_mat_indices = [i for i in xrange(0,match_mat.shape[0]) if count_text[i]>0]
    return match_mat_indices

def find_cosine_match(match_mat, match_mat_rows, cos_mat):
    max_val = 0
    max_ind = -1
    small_mat = match_mat[[match_mat_rows]]
    for i in xrange(0,cos_mat.shape[0]):
        cos_row = np.asarray(cos_mat.getrow(i).todense())[0]
        match_mat_cols = [m for m in xrange(0,match_mat.shape[1]) if cos_row[m]>0]
        match_sum = 0
        for z in xrange(0,small_mat.shape[0]):
            match_sum += np.sum(small_mat[z,match_mat_cols])
        if match_sum>= max_val:
            max_val = match_sum
            max_ind = i
    return max_ind

def find_input_match(input_text, reply_matrix, match_mat, vectorizer, reply_text):
    match_mat_rows = find_input_match_vec(input_text, match_mat, vectorizer)
    max_cos = find_cosine_match(match_mat, match_mat_rows, reply_matrix)
    return max_cos

def create_reply_matrix(reply_text, vectorizer):
    return vectorizer.transform(reply_text)

message_replies = get_message_replies()

table_data = list(chain.from_iterable([mr.get_table_data() for mr in message_replies]))
pd_frame = pd.DataFrame(np.array(table_data),columns=["message", "message_score", "reply", "reply_score"])

match_mat, vectorizer = create_match_matrix(table_data, pd_frame)

reply_matrix = create_reply_matrix(pd_frame['reply'], vectorizer)

input_set = list(set(pd_frame['message']))
replies = []
reply_accurate = []
for z in xrange(0,5):
    input_text = input_set[z]
    input_replies = pd_frame['reply'][pd_frame['message']==input_text]
    input_match = find_input_match(input_text, reply_matrix, match_mat, vectorizer, pd_frame['reply'])
    reply_message = pd_frame['reply'][input_match]
    replies.append(reply_message)
    reply_accurate.append(reply_message in input_replies.tolist())




