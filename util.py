import os
import json
from typing import List


def read_jointslu_labels():
    """return a list of BIO labels for jointslu dataset"""
    with open("../data/jointslu.json") as f:
        data = json.load(f)
        jointslu_labels = data["jointslu_labels"]
    return jointslu_labels


def save_jointslu_labels(new_labels):
    """Save all labels appeared in dataset into file
    :paras new_labels: a list of label lists"""
    data = {"jointslu_labels": list(new_labels)}
    print("label set", data)
    with open("../data/jointslu.json", 'w') as f:
        json.dump(data, f, indent=4)


def append_to_json(file_path, new_data):
    """append new data to current json file. data is a list of json dicts"""
    f = open(file_path, 'r')
    data = []
    if os.path.getsize(file_path) > 0:
        data = json.load(f)
    else:
        print(f"{file_path} file is empty")
    f.close()
    if isinstance(new_data, List):
        data = [*data, *new_data]
    else:
        data.append(new_data)
    f = open(file_path, 'w')
    json.dump(data, f, indent=2)
    f.close()


def read_keys_from_json(path, *args):
    """return content of keys (args) of json file"""
    data = read_from_json(path)
    values = {}
    for arg in args:
        try:
            values[arg] = [item[arg] for item in data]
            # print("add item to values: ", values)
        except KeyError:
            print(f"Missing key {arg} in {path} file")
    return values


def read_from_json(path):
    """read and return the json file content"""
    file = open(path, 'r')
    data = json.load(file)
    file.close()
    return data


def get_gpt3_params(version):
    f = open("model_version.json")
    gpt3_data = json.load(f)["gpt3"]
    for item in gpt3_data:
        if item["version"] == version:
            return item["w_intent"], item["prompt"], item["model"], item["select"]
    return None, None, None, None


def get_parsing_params(version):
    f = open("model_version.json")
    parsing_data = json.load(f)["parsing"]
    for item in parsing_data:
        if item["version"] == version:
            return item["shuffle"]
    return None


def get_pretrain_params(version):
    f = open("model_version.json")
    parsing_data = json.load(f)["pre-train"]
    for item in parsing_data:
        if item["version"] == version:
            return item["lr"], item["max_epoch"], item["batch_size"]
    return None, None, None


def get_pretrain_checkpoint(version):
    f = open("model_version.json")
    parsing_data = json.load(f)["pre-train"]
    for item in parsing_data:
        if item["version"] == version:
            return item["checkpoint"]
    return None


def find_ckpt_in_dir(checkpoint):
    # find .ckpt file in folder
    ckpt_files = []
    ckpt_index = 0
    import os
    for root, dirs, files in os.walk(checkpoint):
        for file in files:
            if file.endswith('.ckpt'):
                ckpt_files.append(os.path.join(root, file))
    # todo: if more than one ckpt file available, ask which one
    if len(ckpt_files) == 1:
        return ckpt_files[0]
    elif len(ckpt_files) > 1:
        ckpt_index = input(f"There are {len(ckpt_files)} files, please input the index\n{ckpt_files}")
        return ckpt_files[int(ckpt_index)]
    else:
        return None


def get3output_paths(dataset, parsing_v, pretrain_v, gpt3_v):
    return get_output_path("parsing", dataset, parsing_v), \
           get_output_path("pre-train", dataset, pretrain_v), \
           get_output_path("gpt3", dataset, gpt3_v)


def get_output_path(model_name, dataset, model_version, scenario):
    """return output path given model name and version,
    model versions are defined in model_version.json file"""
    return create_output_file(model_name, dataset, model_version, scenario)


def create_output_file(model, dataset, v, scenario):
    """create or check file existance according to version and return file location
    """
    path = "data/" + dataset + "/"
    if model == "parsing":
        path = path + "parsing/parsing_outputs/"
    elif model == "pre-train":
        path = path + "pre-train/pre-train_outputs/"
    elif model == "gpt3":
        path = path + "gpt3/gpt3_outputs/"
    else:
        print(f"wrong model name [{model}] is given")
        return
    # create file name by version
    from datetime import datetime
    now = datetime.now()  # current date and time
    day = now.strftime("%d")
    month = now.strftime("%m")
    file_name = month + day + 'v' + str(v) + '.json'
    if scenario:
        file_name = month + day + 'v' + str(v) + "_" + scenario + '.json'
    output_path = path + file_name
    from os.path import exists
    file_index = 1
    while exists(output_path):
        print(f"path already exists {output_path}")
        file_name = month + day + 'v' + str(v) + '_' + str(file_index) + '.json'
        output_path = path + file_name
        file_index = file_index + 1
    open(output_path, 'a+').flush()
    print(f"{output_path} created")
    return output_path


def get_output_folder(model, dataset):
    """At evaluation step, need to get output files for each method, here returns the
    output folder given model name and dataset"""
    path = "data/" + dataset + "/"
    if model == "parsing":
        return path + "parsing/parsing_outputs/"
    elif model == "pre-train":
        return path + "pre-train/pre-train_outputs/"
    elif model == "gpt3":
        return path + "gpt3/gpt3_outputs/"
    else:
        print(f"wrong model name [{model}] is given")
        return None
