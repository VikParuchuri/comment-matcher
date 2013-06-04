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

r = praw.Reddit(user_agent='comment_matcher by /u/vikparuchuri github.com/VikParuchuri/comment_matcher/')
subreddit = r.get_subreddit('funny')

message_replies = []
for submission in subreddit.get_hot(limit=10):
    message_replies += get_submission_reply_pairs(submission)
    if len(message_replies)>=MAX_REPLIES:
        break

table_data = list(chain.from_iterable([mr.get_table_data() for mr in message_replies]))
pd_frame = pd.DataFrame(np.array(table_data),columns=["message", "message_score", "reply", "reply_score"])

mean_score = np.mean(pd_frame['reply_score'])

vectorizer = CountVectorizer(ngram_range=(1,2), min_df = 2/len(table_data), max_df=.6)
all_text = list(set([t[0] for t in table_data] + [t[2] for t in table_data]))
vectorizer.fit(all_text)

count_row = vectorizer.transform([table_data[0][0], table_data[1][0]])

match_mat = np.empty((count_row.shape[1], count_row.shape[1]))
for z in xrange(0,pd_frame.shape[0]):
    row=pd_frame.iloc[z]
    message_count = vectorizer.transform([row['message']]).todense()
    reply_count = vectorizer.transform([row['reply']]).todense()
    for i in xrange(0,message_count.shape[1]):
        if message_count[i]==0:
            continue




