import os
from path import path
import sys
import json

BAD_WORD_LIST = []

BOT_USER_AGENT = ''

BROKER_URL = 'redis://localhost:6379/7'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/7'

REPO_PATH = path(__file__).dirname()
ENV_ROOT = REPO_PATH.dirname()

try:
    with open(REPO_PATH/"secret.json") as secret_file:
        SECRET_KEYS = json.load(secret_file)
        BAD_WORD_LIST = SECRET_KEYS.get("BAD_WORD_LIST", BAD_WORD_LIST)
        BOT_USER_AGENT= SECRET_KEYS.get("BOT_USER_AGENT", BOT_USER_AGENT)
except IOError:
    pass