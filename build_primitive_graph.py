__author__ = 'tonnpa'

import re
import os
import json
import networkx as nx


def build_graph(graph, adjacency_list):
    """
    Builds the graph by appending the nodes and edges defined
    in the adjacency_list
    """
    # nodes
    for node in adjacency_list.keys():
        graph.add_node(node, joinedAt=adjacency_list[node]['joinedAt'],
                             inactiveSince=adjacency_list[node]['inactiveSince'])
# edges
    for node in adjacency_list.keys():
        edges = adjacency_list[node]['edges']
        for to_node, attributes in edges.items():
            graph.add_edge(node, to_node, createdAt=attributes['createdAt'], stoppedAt=attributes['stoppedAt'])


def build_adjacency_list(adj_list, comments, comm_authors):
    """
    Builds the adjacency list by appending the authors and their
    relations to other authors extracted from the comments
    :param comm_authors: dictionary that contains which comments
           were written by whom
    """
    for comment in comments:
        if not comment['author']['isAnonymous']:
            author_uname = comment['author']['username']
            author_reg_date = comment['author']['joinedAt'].split('T')[0]
            created_at = comment['createdAt'].split('T')[0]

            comm_authors[comment['id']] = comment['author']['username']
        else:
            # anonymous posts do not contribute to author relations
            continue

        if author_uname not in adj_list.keys():
            adj_list[author_uname] = {}
            adj_list[author_uname]['joinedAt'] = author_reg_date
            adj_list[author_uname]['inactiveSince'] = author_reg_date
            adj_list[author_uname]['edges'] = {}

        try:
            # TypeError: parent id is not surrounded by quote marks, thus it
            #            is not a string like the other fields, requires casting
            parent_id = str(comment['parent'])
            parent_author = comm_authors[parent_id]
        except KeyError:
            # parent is None: alternative way to handle is to check for None value
            # no parent-child relationship between posts -> no relation detected
            continue
        else:
            try:
                # add author and other details as a dictionary
                # update date of last comment
                if adj_list[author_uname]['inactiveSince'] < created_at:
                    adj_list[author_uname]['inactiveSince'] = created_at
                if adj_list[author_uname]['edges'][parent_author]['createdAt'] > created_at:
                    adj_list[author_uname]['edges'][parent_author]['createdAt'] = created_at
                if adj_list[author_uname]['edges'][parent_author]['stoppedAt'] < created_at:
                    adj_list[author_uname]['edges'][parent_author]['stoppedAt'] = created_at
            except KeyError:
                # parent_author was unknown
                adj_list[author_uname]['edges'][parent_author] = {}
                adj_list[author_uname]['edges'][parent_author]['createdAt'] = created_at
                adj_list[author_uname]['edges'][parent_author]['stoppedAt'] = created_at


DIR_PATH = '/media/sf_Ubuntu/444hu/2014/posts'
files = os.listdir(DIR_PATH)
files.sort()
print('Number of files: ' + str(len(files)))

MIN_RANGE = 0
MAX_RANGE = len(files)

p_graph = nx.DiGraph()
comments_authors = {}
authors_linkage = {}

os.chdir(DIR_PATH)
for file in files[MIN_RANGE:MAX_RANGE]:
    print(file)
    with open(file, 'r') as json_file:
        data = json.loads(json_file.read())

    if re.match('.*_\d\.json$', file) is None:
        # free memory
        comments_authors.clear()
        authors_linkage.clear()

    # 1. phase
    build_adjacency_list(authors_linkage, data['response'], comments_authors)

    # 2. phase
    build_graph(p_graph, authors_linkage)

print('===================')
print('number of nodes: ' + str(len(p_graph.nodes())))
print('number of edges: ' + str(len(p_graph.edges())))
nx.write_graphml(p_graph, '/media/sf_Ubuntu/p_graph_v2.graphml')