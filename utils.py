from __future__ import division
import praw
from praw.objects import Comment, MoreComments
import random
import logging
import pickle
import sys
import settings

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))

MAX_REPLIES = 500
MIN_REPLY_SCORE = 2
MAX_FEATURES = 2000
MAX_REPLY_LENGTH = 250
SUBMISSION_COUNT = 100
SUBREDDIT = "funny"

def read_raw_data_from_cache(filename):
    try:
        raw_data_cache = pickle.load(open(filename, "r"))
    except Exception:
        raw_data_cache = []
    return raw_data_cache

def write_data_to_cache(raw_data, filename, unique_key="message"):
    raw_data_cache = read_raw_data_from_cache(filename)
    raw_data_messages = [r[unique_key] for r in raw_data_cache]
    for r in raw_data:
        if r[unique_key] in raw_data_messages:
            del_index = raw_data_messages.index(r[unique_key])
            del raw_data_cache[del_index]
            del raw_data_messages[del_index]
    raw_data_to_write = [r for r in raw_data if r not in raw_data_cache]
    raw_data_cache += raw_data_to_write

    with open(filename, "w") as openfile:
        pickle.dump(raw_data_cache, openfile)
    return raw_data_cache

def get_single_comment(subreddit_name):
    comment_found = False
    index = 0
    while not comment_found:
        r = praw.Reddit(user_agent=settings.BOT_USER_AGENT)
        subreddit = r.get_subreddit(subreddit_name)
        hourly_top = list(subreddit.get_top_from_hour(limit=(index+1)))
        comments = [c for c in hourly_top[index].comments if isinstance(c, Comment)]
        index += 1
        if len(comments)>2:
            rand_int = random.randint(0,len(comments))
            random_comment = comments[rand_int]
            comment_found = True
        if index>10:
            return None
    return random_comment

def get_submission_reply_pairs(submission, max_replies = MAX_REPLIES, min_reply_score = MIN_REPLY_SCORE):
    message_replies = []
    forest_comments = [c for c in submission.comments if isinstance(c, Comment)]
    while len(forest_comments)>0 and len(message_replies)<MAX_REPLIES:
        actual_replies = []
        for comment in forest_comments:
            try:
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
                actual_replies = [ar for ar in actual_replies if ar.score>=MIN_REPLY_SCORE]
                reply_text = [ar.body for ar in actual_replies]
                scores = [ar.score for ar in actual_replies]
                if len(reply_text)>0:
                    message_replies.append(MessageReply(comment.body, comment.score, reply_text, scores))
            except Exception:
                log.exception("Could not pull single comment data.")
                continue
        forest_comments = [ar for ar in actual_replies if isinstance(actual_replies, Comment)]
    return message_replies

class MessageReply(object):
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

    def get_raw_data(self):
        return {
            'message' : self.message,
            'message_score' : self.message_score,
            'replies' : self.replies,
            'scores' : self.scores
            }

def get_message_replies(subreddit = SUBREDDIT, max_replies = MAX_REPLIES, submission_count = SUBMISSION_COUNT, min_reply_score = MIN_REPLY_SCORE):
    r = praw.Reddit(user_agent=settings.BOT_USER_AGENT)
    subreddit = r.get_subreddit(subreddit)

    message_replies = []
    for submission in subreddit.get_top_from_day(limit=submission_count):
        message_replies += get_submission_reply_pairs(submission, max_replies = max_replies, min_reply_score = min_reply_score)
        if len(message_replies)>=max_replies:
            break
    return message_replies