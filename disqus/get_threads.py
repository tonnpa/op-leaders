#!/usr/bin/env python3
"""
This script retrieves all threads pertaining to the forum specified
since FROM_DATE (yyyy-mm-dd).
Constraints:
    1. the maximum number of queries is 1000 (Disqus API limit)
    2. DST_DIR_PATH, the directory where retrieved files are stored, has to exist
To continue from previous run, specify
    1. FORUM
    2. FROM_DATE
    3. FIRST_QUERY - the successive query number relative to the last run
    3. CURSOR_NEXT (obtain from cursor.next in last .json file)
Type check
    1. CURSOR_NEXT is a string
"""
__author__ = 'tonnpa'

import time

from disqus.fetch import *


DST_DIR_PATH = '/home/tonnpa/hvghu/2014/threads/'
FORUM = 'hvg'
FROM_DATE = '2014-01-01'
FIRST_QUERY = 154
CURSOR_NEXT = '1411312101841430:0:0'
MAX_QUERY_WARNING = 800

has_next = True
num_queries = 1
since_date = int(time.mktime(time.strptime(FROM_DATE, '%Y-%m-%d')))

while num_queries < MAX_QUERY_WARNING and has_next:
    # get url
    url_threads = get_url_list_threads(forum=FORUM, since=since_date, cursor=CURSOR_NEXT)
    # query url to get json response
    json_threads = get_json(url_threads)
    # save json data
    outfile_path = DST_DIR_PATH + FROM_DATE + '_' + str(FIRST_QUERY).zfill(4) + '.json'
    with open(outfile_path, 'w') as outfile:
        json.dump(json_threads, outfile)
    # prepare next iteration
    FIRST_QUERY += 1
    num_queries += 1
    has_next = json_threads['cursor']['hasNext']
    CURSOR_NEXT = json_threads['cursor']['next']

    # feedback on progress
    if num_queries % 20 == 0:
        print('Iteration: ' + str(num_queries))
