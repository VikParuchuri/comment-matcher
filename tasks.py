from celery import Celery
from knn_matching import write_data_to_cache, train_knn_matcher, test_knn_matcher
from utils import get_message_replies
from celery.task import periodic_task
from datetime import timedelta

celery = Celery('tasks', broker='redis://localhost:6379/93', backend='redis://localhost:6379/93')
SUBREDDIT_LIST = ["funny", "pics", "gaming"]

@periodic_task(run_every=timedelta(minutes = 30))
def get_reddit_posts():
    all_message_replies = []
    for subreddit in SUBREDDIT_LIST:
        message_replies = get_message_replies(subreddit =subreddit, max_replies= 500, submission_count = 300, min_reply_score = 20)
        all_message_replies += message_replies
    raw_data = list([mr.get_raw_data() for mr in all_message_replies])
    write_data_to_cache(raw_data)

