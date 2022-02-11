import os

def file_existed(filepath):
    return os.path.exists(filepath)

def create_dir_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def create_file_dir_not_exist(file_path):
    create_dir_not_exist(os.path.dirname(file_path))