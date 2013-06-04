from __future__ import division
import praw
from praw.objects import Comment, MoreComments
from itertools import chain
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from utils import get_message_replies, MessageReply, get_submission_reply_pairs

message_replies = get_message_replies(subreddit = "funny", max_replies= 500, submission_count = 300, min_reply_score = 20)
