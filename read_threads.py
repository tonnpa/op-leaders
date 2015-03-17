__author__ = 'tonnpa'

import os
import re

os.chdir('./test_case/')
files = os.listdir('.')
files.sort()
print(files)
it = iter(files)

file = next(it)
while True:
    try:
        file2 = next(it)
        if re.match('.*_\d.json$',file2) is not None:
            print('Keep it going')
        print(file + file2)
        file = file2
    except StopIteration:
        break
