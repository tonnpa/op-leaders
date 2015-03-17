#!/usr/bin/env python3
"""
This script renames all the posts files that have more than one parts
and the separator _ is missing between the Thread ID and the segment number.
"""
__author__ = 'tonnpa'

import os

dir_path = './posts/'
files = os.listdir(dir_path)

os.chdir(dir_path)

for file in files:
    name_segs = file.split('_')
    try:
        # 16 only occurs when the separator _ is missing
        if name_segs[2].endswith('json') and len(name_segs[2]) == 16:
            print(name_segs[2])
            seg2_p1 = name_segs[2][:10]
            seg2_p2 = name_segs[2][10:16]
            new_name = name_segs[0] + '_' + name_segs[1] + '_' + seg2_p1 + '_' + seg2_p2
            print(new_name)
            # os.rename(file, new_name)
    except IndexError:
        # folders are expected to show up
        print(name_segs)