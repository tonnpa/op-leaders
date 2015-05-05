__author__ = 'tonnpa'

from graph import discussion_graph as DG

SRC_DIR = '/tmp/posts'

g = DG.build_graph(SRC_DIR, g_path='/tmp/test.graphml')

DG.write_egonets('/tmp/test.graphml', '/tmp/egonets')

print(len(g.nodes()))
print(len(g.edges()))