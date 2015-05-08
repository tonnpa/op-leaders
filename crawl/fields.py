#!/usr/bin/env python3
"""
This script lists the fields of a Disqus json response.

SRC_THREAD_FILE: the json file retrieved using the disqus API listThreads method
SRC_POST_FILE: the json file retrieved using the disqus API listPosts method
"""
__author__ = 'tonnpa'

import json

SRC_THREAD_FILE = '... json file path here ...'
SRC_POST_FILE = '... json file path here ...'

# forums/listThreads
with open(SRC_THREAD_FILE) as json_data:
    data = json.loads(json_data.read())

print('Fields of a JSON Response: START')
for field in data:
    print(field)
print('Fields of a JSON Response: END' + '\n')

print('Fields of Forum/listThreads: START')
for field in data['response'][0]:
    print(field)
print('Fields of Forum/listThreads: END' + '\n')

# threads/listPosts
with open(SRC_POST_FILE) as posts_444:
    data = json.loads(posts_444.read())

print('Fields of Thread/listComments: START')
for field in data['response'][0]:
    print(field)
print('Fields of Thread/listComments: END' + '\n')
