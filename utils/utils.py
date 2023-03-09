import json
import os

def save_json(file, name, output_path = ''):
    with open(output_path + name, "w") as outfile:
        json.dump(file, outfile)


def read_json(f_path):
    f = open(f_path)
    data = json.load(f)
    return data


def find_json_files(path):
    f_path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith('.json'):
                f_path.append(os.path.join(root, name))

    return f_path

def create_new_folder(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

def no_intersection_lists(list1, list2):
    no_inter_list = []
    for l in list1:
        if l not in list2:
            no_inter_list.append(l)

    return no_inter_list