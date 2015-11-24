__author__ = 'tonnpa'


def build_adjacency_list(adj_list, comments, comm_authors):
    """
    Builds the adjacency list by appending the authors and their
    relations to other authors extracted from the comments
    :param comm_authors: dictionary that contains which comments
           were written by whom
    """
    for comment in comments:
        # anonymous posts do not contribute to author relations
        if not comment['author']['isAnonymous']:
            author_uname = comment['author']['username']
            author_reg_date = comment['author']['joinedAt'].split('T')[0]
            created_at = comment['createdAt'].split('T')[0]

            comm_authors[comment['id']] = author_uname

            if author_uname not in adj_list:
                adj_list[author_uname] = {}
                adj_list[author_uname]['joinedAt'] = author_reg_date
                adj_list[author_uname]['inactiveSince'] = author_reg_date
                adj_list[author_uname]['edges'] = {}
                adj_list[author_uname]['likes'] = 0
                adj_list[author_uname]['dislikes'] = 0

            # add number of likes and dislikes
            adj_list[author_uname]['likes'] += comment['likes']
            adj_list[author_uname]['dislikes'] += comment['dislikes']
            # update user activity
            adj_list[author_uname]['inactiveSince'] = max(created_at, adj_list[author_uname]['inactiveSince'])

            # TypeError: parent id is not surrounded by quote marks, thus it
            #            is not a string like the other fields, requires casting
            parent_id = str(comment['parent'])
            if parent_id != 'None' and parent_id in comm_authors:
                parent_author = comm_authors[parent_id]
                try:
                    # add author and other details as a dictionary
                    # update date of last comment
                    adj_list[author_uname]['edges'][parent_author]['createdAt'] = \
                        min(created_at, adj_list[author_uname]['edges'][parent_author]['createdAt'])
                    adj_list[author_uname]['edges'][parent_author]['stoppedAt'] = \
                        max(created_at, adj_list[author_uname]['edges'][parent_author]['stoppedAt'])

                    adj_list[author_uname]['edges'][parent_author]['interactions'] += 1
                except KeyError:
                    # parent_author was unknown
                    adj_list[author_uname]['edges'][parent_author] = {}
                    adj_list[author_uname]['edges'][parent_author]['createdAt'] = created_at
                    adj_list[author_uname]['edges'][parent_author]['stoppedAt'] = created_at
                    adj_list[author_uname]['edges'][parent_author]['interactions'] = 1


def build_graph(graph, adjacency_list):
    """
    Builds the graphs by appending the nodes and edges defined
    in the adjacency_list
    :param graph: the graphs to which elements are appended
    """
    # nodes
    for node in adjacency_list:
        graph.add_node(node,
                       joinedAt=adjacency_list[node]['joinedAt'],
                       inactiveSince=adjacency_list[node]['inactiveSince'],
                       likes=adjacency_list[node]['likes'],
                       dislikes=adjacency_list[node]['dislikes'])
    # edges
    for node in adjacency_list:
        edges = adjacency_list[node]['edges']
        for to_node, attributes in edges.items():
            graph.add_edge(node, to_node,
                           createdAt=attributes['createdAt'],
                           stoppedAt=attributes['stoppedAt'],
                           interactions=attributes['interactions'])