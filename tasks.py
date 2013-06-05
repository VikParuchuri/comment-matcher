from celery import Celery
from knn_matching import write_data_to_cache, train_knn_matcher, test_knn_matcher
from utils import get_message_replies
from celery.task import periodic_task
from datetime import timedelta
import functools
import os
import time

celery = Celery('tasks', broker='redis://localhost:6379/93', backend='redis://localhost:6379/93')
SUBREDDIT_LIST = ["funny", "pics", "gaming"]

class FileCache(object):
    def __init__(self, lock_name, lock_time = 3600):
        self.lock_name = lock_name
        self.lock_time = lock_time

    def acquire_lock(self):
        try:
            with open(self.lock_name):
                created = os.path.getctime(self.lock_name)
                cur_time = time.time()
                if (cur_time - created)> self.lock_time:
                    os.unlink(self.lock_name)
                    return True
                return False
        except IOError:
            open(self.lock_name, "w+")
            return True

    def release_lock(self):
        os.unlink(self.lock_name)

def single_instance_task(timeout):
    def task_exc(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            lock_id = "celery-single-instance-" + func.__name__
            filecache = FileCache(lock_id, timeout)
            if filecache.acquire_lock():
                try:
                    func(*args, **kwargs)
                finally:
                    filecache.release_lock()
        return wrapper
    return task_exc

@periodic_task(run_every=timedelta(minutes = 30))
@single_instance_task(timeout=3 * 60 * 60)
def get_reddit_posts():
    all_message_replies = []
    for subreddit in SUBREDDIT_LIST:
        message_replies = get_message_replies(subreddit =subreddit, max_replies= 500, submission_count = 300, min_reply_score = 20)
        all_message_replies += message_replies
    raw_data = list([mr.get_raw_data() for mr in all_message_replies])
    write_data_to_cache(raw_data)
