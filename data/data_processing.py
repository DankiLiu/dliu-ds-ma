import json
import os
from typing import List

import pandas as pd
from pandas.core.indexing import IndexingError
from scipy.ndimage import label

import util


def read_jointslu_lines(file_path=None):
    if not file_path:
        file_path = "../data/sample.iob"
    with open(file_path) as f:
        lines = f.readlines()
    return lines


def jointslu_per_line(line: str):
    """
    process data from JointSLU dataset
    :param line: one line of data read from .iob file
           path: path
    :return: sentence: a string without EOS and BOS
            words: a list of words without EOS and BOS
            labels: a list of labels, BIO tags
    words and labels should have same length (one to one match)
    """
    # Split the sentence and labels by tab
    sentence, labels = line.split('\t')
    sentence = sentence.strip()
    labels = labels.strip()
    # Strip BOS, EOS labels, first and last labels
    words_list, labels_list = sentence.split(' '), labels.split(' ')
    words_len, labels_len = len(words_list), len(labels_list)
    assert words_len == labels_len

    words = words_list[1: words_len - 1]
    labels = labels_list[1: labels_len - 1]
    sentence = " ".join(words)
    return sentence, words, labels


def find_all_labels(llist, boi=False):
    import itertools
    labels = list(itertools.chain(*llist))
    labels = list(filter(lambda l: l != 'O', labels))

    if not boi:
        labels = [label.split('-')[1] for label in labels]
    labels = set(labels)
    print(f"{len(labels)} labels: {labels}")
    util.save_jointslu_labels(labels)


def store_jointslu_labels():
    with open("jointslu/training_data/train.json") as f:
        data = json.load(f)
    labels_set = []
    print(len(data), " data in train.json")
    for ele in data:
        labels = ele["labels"]
        for label in labels.split(' '):
            labels_set.append(label)
    labels_set = set(labels_set)
    print(f"{len(labels_set)} labels are [{labels_set}]")
    util.save_jointslu_labels(labels_set)


def construct_jointslu_data(type_name: str, lines):
    """Construct the raw data for training usage.
    type: train/test/val
    lines: data lines from jointslu dataset
    train.iob (raw data) into train.json"""

    # Construct raw data into json format
    def strip_extra(old_str):
        new_str = old_str.strip()
        new_str = new_str.strip('\n')
        return new_str

    data = []
    for i in range(len(lines)):
        text, labels = lines[i].split('\t')
        new_text = strip_extra(text)
        new_labels = strip_extra(labels)
        if new_text and new_labels is not None:
            print(f"text length {len(new_text.split(' '))}")
            print(f"labels length {len(new_labels.split(' '))}")
            print(new_text.split(' '))
            print(new_labels.split(' '))
            assert len(new_text.split(' ')) == len(new_labels.split(' '))
            data.append({"id": i + 1,
                         "text": new_text,
                         "labels": new_labels})
    # Store data by type
    store_path = "../data/jointslu/" + type_name + ".json"
    outfile = open(store_path, 'w')
    json.dump(data, outfile, indent=4)


def set_cls_sep_tokens():
    with open("jointslu/temp/val.json") as f:
        data = json.load(f)
    new_data = []
    for dic in data:
        labels = dic["labels"].split(' ')
        labels[0] = "[CLS]"
        labels[-1] = "[SEP]"
        new_labels = ' '.join(labels)
        dic["labels"] = new_labels
        new_data.append(dic)
    f.close()
    outfile = open("jointslu/temp/val.json", 'w')
    print(data[0])
    print(new_data[0])
    json.dump(new_data, outfile, indent=4)
    outfile.close()


def get_labels_dict(labels_version):
    """return labels in dictionary form, one label matches to one index number"""
    global labels_path, f
    file_path = "data/jointslu/pre-train/labels" + "labels_version" + ".json"
    if os.path.exists(file_path):
        f = open(file_path)
        labels_dict = json.load(f)
        f.close()
        return labels_dict
    else:
        # if for labels_v, labels file does not exist, create and return
        try:
            labels_path = "data/jointslu/labels/labels" + labels_version + ".csv"
            f = open(labels_path, 'r')
        except FileNotFoundError:
            print(f"{labels_path} not found, pls check again.")
        # read second column without index and store them in sjon format
        simplified_df = pd.read_csv(labels_path, usecols=[1])
        simplified_list = simplified_df['simplified label'].tolist()
        simplified_set = list(set(simplified_list))
        print(simplified_list)
        labels_dictionary = dict(zip(simplified_set, range(len(simplified_set))))
        # store in file
        with open(file_path, 'w') as infile:
            json.dump(labels_dictionary, infile, indent=4)
            infile.close()
        return labels_dictionary


def gpt3_from_bert_dataset():
    f = open("jointslu/temp/train.json")
    data = json.load(f)
    output_file = open("../data/jointslu/gpt3/train.json", 'a')
    for example in data:
        text = example["text"]
        text = text.replace('EOS', '')
        text = text.replace('BOS', '')
        text = text.strip() + '.'
        print(text)
        output_file.write(text + '\n')
    output_file.close()
    f.close()


def create_training_data(shuffle=False):
    # todo: parsing use same training data as other two models but has preprocessing step
    """create data for parsing from "bert_***.json" without bos and eos
    and store them in parsing folder"""
    # read train.json
    f = open("jointslu/temp/train.json", 'r')
    import json
    data = json.load(f)
    # remove bos and eos
    new_data = []
    for i in data:
        tok_len = len(i["text"].split(' '))
        new_text = ' '.join(i["text"].split(' ')[1:tok_len - 1])
        new_label = ' '.join(i["labels"].split(' ')[1:tok_len - 1])
        new_i = {
            "id": i["id"],
            "text": new_text,
            "labels": new_label
        }
        new_data.append(new_i)
    f.close()
    # store in file
    with open("jointslu/parsing/train.json", 'w') as f:
        json.dump(new_data,
                  f,
                  indent=4)
    f.close()


def simp_label(ori, label_path):
    """ return the matched simplified label give original label """
    import pandas as pd
    ori_labels, sim_labels = [], []
    try:
        df = pd.read_csv(label_path)
        ori_labels, sim_labels = df['original label'].tolist(), df['simplified label'].tolist()
    except KeyError or IndexingError:
        print(f"dateframe reading error")
    labels_dict = dict(zip(ori_labels, sim_labels))
    try:
        if labels_dict[ori]:
            return labels_dict[ori]
        else:
            return "O"
    except KeyError or IndexingError:
        print(f"read {ori} from dict error")


def get_labels_file_path(labels_version):
    file_name = "labels" + labels_version + ".csv"
    return "data/jointslu/labels/" + file_name


def get_simplified_labels(oris: List, labels_version):
    """ return a list of labels that matched from the original labels
    to simplified labels for one example"""
    label_path = get_labels_file_path(labels_version)
    labels = []
    for label in oris:
        sim = simp_label(label, label_path)
        labels.append(sim)
    return labels


def construct_data(data, labels_version):
    """change the labels to simplified labels and add bos and eos to the label"""
    new_data = []
    i = 0
    for ele in data:
        # labels list
        labels = ele["labels"].split(' ')
        new_labels = get_simplified_labels(labels[1: len(labels) - 1], labels_version)
        new_labels.insert(0, '[CLS]')
        new_labels.append('[SEP]')
        assert len(labels) == len(new_labels)

        new_data.append({
            'id': i,
            'text': ele['text'].split(' '),
            'labels': new_labels
        })
        i = i + 1
    return new_data


def construct_training_data(data_type, labels_version):
    """change the annotated data for training pre-trained model to simplified labels"""
    path = "data/jointslu/training_data/" + data_type + ".json"
    new_path = "data/jointslu/training_data/labels" + labels_version + "/" \
               + data_type + ".json"
    file = open(path)
    data = json.load(file)
    new_data = construct_data(data, labels_version)
    with open(new_path, 'w') as f:
        json.dump(new_data, f, indent=4)
        f.close()


def generate_gpt3_examples_file(dataset, datatype, labels_version, in_file):
    """select an example for each label and store into file, number suggests the labels_version"""
    examples = []
    label_dict = get_labels_dict(labels_version)
    labels = list(label_dict.keys())
    # open pre-train training data file
    path = data_path_by_lv(dataset=dataset,
                           data_type=datatype,
                           labels_version=labels_version)
    f = open(path, 'r')
    data = json.load(f)
    for idx, label in enumerate(labels):
        example = None
        for item in data:
            # if contains the target label, save this example
            if label in item["labels"]:
                example = {
                    "id": idx,
                    "text": item["text"],
                    "labels": item["labels"]
                }
                break
        examples.append({
            "label": label,
            "example": example
        })
        print(f"example of {label} is\n{example}")
    f_in = open(in_file, 'w')
    json.dump(examples, f_in, indent=4)
    f_in.close()
    f.close()
    print(f"created examples file under {in_file}")


def data_path_by_lv(dataset, data_type, labels_version):
    """return path
    dataset: name of a dataset, [jointslu, ...]
    data_type: [train, test, val]"""
    folder_path = 'data/' + dataset + '/training_data/labels' + labels_version + '/'
    file_name = data_type + '.json'
    return folder_path + file_name


def check_training_data(labels_version):
    """given labels_version, the existence of training data should be checked,
    if not exist, then create training data"""
    folder_path = "data/jointslu/training_data/labels" + labels_version
    print(f"lv{labels_version}: training data should be stored in {folder_path}")
    # if folder path exist and not empty, then the data exists and return True
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if os.listdir(folder_path):
        return True
    else:
        # construct training data
        for data_type in ["train", "test", "val"]:
            try:
                construct_training_data(data_type=data_type,
                                        labels_version=labels_version)
            except FileNotFoundError:
                print(f"fail to construct training data for {data_type}")
                return False
        return True