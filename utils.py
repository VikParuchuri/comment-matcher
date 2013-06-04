from __future__ import division
import praw
from praw.objects import Comment, MoreComments

MAX_REPLIES = 500
MIN_REPLY_SCORE = 2
MAX_FEATURES = 2000
MAX_REPLY_LENGTH = 250
SUBMISSION_COUNT = 100
SUBREDDIT = "funny"

def get_submission_reply_pairs(submission, max_replies = MAX_REPLIES, min_reply_score = MIN_REPLY_SCORE):
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
            actual_replies = [ar for ar in actual_replies if ar.score>=MIN_REPLY_SCORE]
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

def get_message_replies(subreddit = SUBREDDIT, max_replies = MAX_REPLIES, submission_count = SUBMISSION_COUNT, min_reply_score = MIN_REPLY_SCORE):
    r = praw.Reddit(user_agent='comment_matcher by /u/vikparuchuri github.com/VikParuchuri/comment_matcher/')
    subreddit = r.get_subreddit(subreddit)

    message_replies = []
    for submission in subreddit.get_top_from_month(limit=submission_count):
        message_replies += get_submission_reply_pairs(submission, max_replies = max_replies, min_reply_score = min_reply_score)
        if len(message_replies)>=max_replies:
            break
    return message_replies