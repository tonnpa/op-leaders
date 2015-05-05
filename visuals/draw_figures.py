__author__ = 'tonnpa'

from matplotlib import pyplot as plt
import csv
import os

SOURCE_FILE = '/media/sf_Ubuntu/opleaders/egonet_features.csv'
# TARGET_DIR = '/media/sf_Ubuntu/opleaders/figures'
TARGET_DIR = '/tmp'
OUT_FILE = '/tmp/commenters.txt'

os.chdir(TARGET_DIR)

csvfile = open(SOURCE_FILE, 'r')
reader = csv.reader(csvfile, delimiter=';')

txtfile = open(OUT_FILE, 'w')

all_rows = [row for row in reader]
header = all_rows.pop(0)

periods = set([row[4] for row in all_rows])
rows_by_period = {}

for period in periods:
    rows = [row for row in all_rows if row[4] == period]
    rows_by_period[period] = rows

for period in sorted(periods):
    nfeat = [int(r[2]) for r in rows_by_period[period]]
    efeat = [int(r[3]) for r in rows_by_period[period]]
    labels = [r[1] for r in rows_by_period[period]]

    nmax = max(nfeat)
    emax = max(efeat)

    plt.suptitle('Node vs Edge Feature for Quarters: ' + period)
    plt.xlabel('#Nodes')
    plt.ylabel('#Edges')
    plt.axis([0, int(max(nfeat))*1.2, 0, int(max(efeat))*1.2])
    # plt.axis([0, 1000, 0, 10000])
    plt.scatter(nfeat, efeat)

    for label, x, y in zip(labels, nfeat, efeat):
        threshold = 0.6
        if x > nmax*threshold or y > emax*threshold:
            plt.annotate(
                label,
                xy=(x, y), xytext = (-15, 15),
                textcoords = 'offset points',
                horizontalalignment = 'right',
                verticalalignment = 'bottom',
                arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
            txtfile.write(label + ' #nodes: ' + str(x) + ' #edges: ' + str(y) + ' quarter: ' + period + '\n')

    plt.savefig('figure_nvse_' + period)
    plt.clf()

    print('Figure ' + period + ' created.')

csvfile.close()
txtfile.close()