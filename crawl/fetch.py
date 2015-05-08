__author__ = 'tonnpa'

from urllib.request import urlopen
import json

# static variables common to all URLs
api_version = '3.0'
output_type = '.json?'
api_key = 'api_key=OjL90VsZnaWjtuJwYhWEC6RJKN2lNucCCCUSkygOJCc6gWtjHjZdDKcceBRbL4V2'


def get_url_list_threads(forum, since='', cursor='', limit=100, order='asc'):
    """
    Creates URL link to query threads of a specified forum
    :param forum: forum id
    :return:URL in string
    """
    resource_path = 'forums/listThreads'

    url = "https://disqus.com/api/" + api_version + '/' + resource_path + output_type + api_key + \
          '&forum='  + forum + \
          '&since='  + str(since) + \
          '&cursor=' + str(cursor) + \
          '&limit='  + str(limit) + \
          '&order='  + order

    return url


def get_url_list_posts(thread, since='', cursor='', limit=100, order='asc'):
    """
    Creates URL link to query posts of a specified thread
    :param thread: thread id
    :return: URL in string
    """
    resource_path = 'threads/listPosts'

    url = "https://disqus.com/api/" + api_version + '/' + resource_path + output_type + api_key + \
          '&thread=' + str(thread) +\
          '&since='  + str(since) + \
          '&cursor=' + str(cursor) + \
          '&limit='  + str(limit) + \
          '&order='  + order
    return url


def get_json(url):
    """
    Get JSON data by invoking the url
    1. open url
    2. read the http response and decode it as utf-8
    3. interpret the decoded content as a json file
    :return:JSON formatted data
    """
    return json.loads(urlopen(url).read().decode('utf-8'))