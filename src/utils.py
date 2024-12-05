import os


def get_data_from(filename):
    curr_dir = os.getcwd()
    project_dir = os.path.dirname(os.path.dirname(curr_dir))
    data_dir = "".join([project_dir, '\data\, filename])