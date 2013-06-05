from celery import Celery
from knn_matching import write_data_to_cache, train_knn_matcher, test_knn_matcher
from utils import get_message_replies, read_raw_data_from_cache, write_data_to_cache, get_single_comment
from celery.task import periodic_task
from datetime import timedelta
import functools
import os
import time
import logging
import time
import random

log = logging.getLogger(__name__)

celery = Celery('tasks', broker='redis://localhost:6379/7', backend='redis://localhost:6379/7')
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
    try:
        all_message_replies = []
        for subreddit in SUBREDDIT_LIST:
            message_replies = get_message_replies(subreddit =subreddit, max_replies= 500, submission_count = 300, min_reply_score = 20)
            all_message_replies += message_replies
        raw_data = list([mr.get_raw_data() for mr in all_message_replies])
        write_data_to_cache(raw_data, "raw_data_cache.p")
    except:
        log.exception("Could not save posts.")

@periodic_task(run_every=timedelta(minutes = 5))
@single_instance_task(timeout=3 * 60 * 60)
def pull_down_comments():
    sleep_time = random.randint(0,300)
    time.sleep(sleep_time)
    raw_data = read_raw_data_from_cache("raw_data_cache.p")
    items_done = read_raw_data_from_cache("items_done.p")
    comments = [c['text'] for c in items_done]
    replies = [c['reply'] for c in items_done]
    knn_matcher = train_knn_matcher(raw_data)
    for subreddit in SUBREDDIT_LIST:
        comment = get_single_comment(subreddit)
        text = comment.body
        cid = comment.id
        reply = test_knn_matcher(knn_matcher, comment)
        if text in comments or reply in replies:
            continue
        data = {'comment' : text, 'reply' : reply, 'comment_id' : cid}
        items_done.append(data)
        print "Subreddit: {0}".format(subreddit)
        print "Comment: {0} {1}".format(cid, text)
        print "Reply: {0}".format(reply)
        print "-------------------"
    write_data_to_cache(items_done, "items_done.p", "comment_id")
