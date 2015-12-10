#!/usr/bin/env python3
"""
This script retrieves all the posts pertaining to the threads in
the .json files.
Constraints
    1. the files in the SRC_DIR_PATH has to follow a specific naming convention:
        FROM_DATE in (yyyy-mm-dd) _ file_number [0-9999] .json
    2. the maximum number of queries is 1000 (Disqus API limit)
    3. only threads with more than MIN_POST_CNT number of posts is queried
    4. DST_DIR_PATH, the directory where retrieved files are stored, has to exist
To continue from previous run, specify
    1. FROM_DATE
    2. FIRST_FILE the number of file that should be processed
    3. LAST_THREAD_ID the number of thread ID that was last processed
"""
__author__ = 'tonnpa'

import os

from disqus.fetch import *


FROM_DATE = '2014-01-01'
SRC_DIR_PATH = '/home/tonnpa/hvghu/2014/threads/'
DST_DIR_PATH = '/home/tonnpa/hvghu/2014/posts/'
FIRST_FILE = 207
LAST_THREAD_ID = 3418529550
MAX_QUERY_WARNING = 995
MIN_POST_CNT = 5

# count the number of files in source directory
num_files = len(os.listdir(SRC_DIR_PATH))
num_queries = 0

for file_num in range(FIRST_FILE, num_files+1):
    # open JSON file and read threads into data
    with open(SRC_DIR_PATH + FROM_DATE + '_' + str(file_num).zfill(4) + '.json') as file:
        data = json.loads(file.read())

    # process each thread
    for thread in data['response']:
        # skip previously processed files
        if file_num == FIRST_FILE and int(thread['id']) <= LAST_THREAD_ID:
            continue

        # if thread has more than 5 posts, then query for all its posts
        if thread['posts'] > MIN_POST_CNT:
            # get url
            url_posts = get_url_list_posts(thread=thread['id'])
            # query url to get json data
            json_posts = get_json(url_posts)
            num_queries += 1
            # save json data
            outfile_path = DST_DIR_PATH + FROM_DATE + '_' + str(file_num).zfill(4) + '_' + str(thread['id'] + '.json')
            with open(outfile_path, 'w') as outfile:
                json.dump(json_posts, outfile)

            segment_num = 1
            # save all further comments
            while json_posts['cursor']['hasNext']:
                cursor_next = json_posts['cursor']['next']
                url_posts = get_url_list_posts(thread=thread['id'], cursor=cursor_next)
                json_posts = get_json(url_posts)
                num_queries += 1

                segment_num += 1
                outfile_path = DST_DIR_PATH + FROM_DATE + '_' + str(file_num).zfill(4) + '_' + \
                               str(thread['id'] + '_' + str(segment_num) + '.json')
                with open(outfile_path, 'w') as outfile:
                    json.dump(json_posts, outfile)

        if num_queries % 20 == 0:
            print('File: ' + str(file_num).zfill(4) + ' Iteration: ' + str(num_queries))
        if num_queries > MAX_QUERY_WARNING:
            print('Ending process. Last Thread ID: ' + str(thread['id']))
            break  # looping at threads in a file

    if num_queries > MAX_QUERY_WARNING:
        print('Ending process. Last File Number: ' + str(file_num))
        break  # looping at files
