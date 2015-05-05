__author__ = 'tonnpa'

import os
import os.path


def src_dir_exists(path):
    exists = os.path.exists(path)
    if not exists:
        print('ERROR: Source directory does not exist: ' + path)
    return exists


def src_dir_empty(path):
    if not os.listdir(path):
        print('ERROR: Empty source directory: ' + path)
        return True
    else:
        return False


def tgt_dir_exists(path):
    exists = os.path.exists(path)
    if not exists:
        print('ERROR: Target directory does not exist: ' + path)
    return exists


def src_file_exists(path):
    exists = os.path.exists(path)
    if not exists:
        print('ERROR: Source file does not exist: ' + path)
    return exists

def tgt_file_exists(path):
    exists = os.path.exists(path)
    if exists:
        print('WARNING: Target file ' + os.path.basename(path) + ' will be overwritten')
    return exists


def file_extension_match(path, expected_extension):
    extension = os.path.splitext(path)[1]
    match = extension == expected_extension
    if not match:
        print('ERROR: Invalid target file extension: ' + extension)
    return match
