#!/usr/bin/env python3
"""
This script lists the fields of a Disqus json response.
"""
__author__ = 'tonnpa'

import json

# forums/listThreads
with open('./fields/threads.json') as json_data:
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
with open('./fields/posts.json') as posts_444:
    data = json.loads(posts_444.read())

print('Fields of Thread/listComments: START')
for field in data['response'][0]:
    print(field)
print('Fields of Thread/listComments: END' + '\n')
