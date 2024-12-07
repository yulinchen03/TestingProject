import os


def load_data_from(file_path):
    curr_dir = os.getcwd()
    project_dir = os.path.dirname(os.path.dirname(curr_dir))
    data_dir = os.path.join(project_dir, file_path)
    return data_dir
