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
    graph.add_nodes_from(list(adjacency_list.keys()))
    for node, to_nodes in adjacency_list.items():
        for to_node in to_nodes:
            target_author = to_node[0]
            comment_created_at = to_node[1]
            graph.add_edge(node, target_author, createdAt=comment_created_at)


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
            created_at = comment['createdAt'].split('T')[0]

            comm_authors[comment['id']] = comment['author']['username']
        else:
            # anonymous posts do not contribute to author relations
            continue

        try:
            # TypeError: parent id is not surrounded by quote marks, thus it
            #            is not a string like the other fields, requires casting
            parent_id = str(comment['parent'])
            parent_author = comm_authors[parent_id]
        except KeyError:
            # parent is None: alternative way to handle is to check for None value
            # no parent-child relationship between posts -> no relation detected
            if not author_uname in adj_list.keys():
                adj_list[author_uname] = set()
        else:
            try:
                # add author and date of comment as a tuple
                # alternative: use a dictionary data structure for robustness
                adj_list[author_uname].add((parent_author, created_at))
            except KeyError:
                adj_list[author_uname] = set()
                adj_list[author_uname].add((parent_author, created_at))

DIR_PATH = '/home/tonnpa/444hu/2014/posts/'
files = os.listdir(DIR_PATH)
files.sort()
print('Number of files: ' + str(len(files)))

MIN_RANGE = 0
MAX_RANGE = 100

p_graph = nx.MultiDiGraph()
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
nx.write_graphml(p_graph, '/media/sf_Ubuntu/p_graph.graphml')