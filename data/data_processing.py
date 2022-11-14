import json
from typing import List

from pandas.core.indexing import IndexingError

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
    with open("../data/jointslu/train.json") as f:
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
    with open("jointslu/bert_val.json") as f:
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
    outfile = open("jointslu/bert_val.json", 'w')
    print(data[0])
    print(new_data[0])
    json.dump(new_data, outfile, indent=4)
    outfile.close()


def read_jointslu_labels_dict():
    f = open("../data/jointslu/pre-train/labels.json")
    labels_dict = json.load(f)
    f.close()
    return labels_dict


def gpt3_from_bert_dataset():
    f = open("../data/jointslu/bert_train.json")
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
    """create data for parsing from "bert_***.json" without bos and eos
    and store them in parsing_eval folder"""
    # read bert_train.json
    f = open("../data/jointslu/bert_train.json", 'r')
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
    with open("../data/jointslu/parsing_eval/train.json", 'w') as f:
        json.dump(new_data,
                  f,
                  indent=4)
    f.close()


def simp_label(ori):
    """ return the matched simplified label give original label """
    import pandas as pd
    df = pd.read_csv("../data/jointslu/labels.csv",
                     index_col=False)

    try:
        df2 = df.loc[df['original label'] == ori, 'simplified label']
    except KeyError or IndexingError:
        print(f"Can not find the label matching [{ori}]")
        return ["O"]
    if not df2.values:
        return ["O"]
    return df2.values


def get_simplified_labels(oris: List):
    """ return a list of labels that matched from the original labels
    to simplified labels """
    labels = []
    for label in oris:
        sim = simp_label(label)
        labels.append(sim)
    # flatten the list
    labels = [i for sublist in labels for i in sublist]
    return labels


def construct_data(data):
    new_data = []
    i = 0
    for ele in data:
        # labels list
        labels = ele["labels"].split(' ')
        new_labels = get_simplified_labels(labels[1: len(labels) - 1])
        new_labels.insert(0, '[CLS]')
        new_labels.append('[SEP]')
        print(labels)
        print(len(labels))
        print(new_labels)
        print(len(new_labels))
        assert len(labels) == len(new_labels)

        new_data.append({
            'id': i,
            'text': ele['text'].split(' '),
            'labels': new_labels
        })
        i = i + 1
    return new_data


def change_labels2simplified():
    """change the annotated data for training pre-trained model to simplified labels"""
    path_train = '../data/jointslu/bert_train.json'
    path_test = '../data/jointslu/bert_test.json'
    path_val = '../data/jointslu/bert_val.json'

    new_train_path = '../data/jointslu/pre-train/b_train.json'
    new_test_path = '../data/jointslu/pre-train/b_test.json'
    new_val_path = '../data/jointslu/pre-train/b_val.json'
    """
    train_f = open(path_train)
    train_data = json.load(train_f)
    new_train = construct_data(train_data)
    with open(new_train_path, 'w') as f:
        json.dump(new_train, f, indent=4)
        f.close()
    
    test_f = open(path_test)
    test_data = json.load(test_f)
    new_test = construct_data(test_data)
    with open(new_test_path, 'w') as f:
        json.dump(new_test, f, indent=4)
        f.close()
    """
    val_f = open(path_val)
    val_data = json.load(val_f)
    new_val = construct_data(val_data)
    with open(new_val_path, 'w') as f:
        json.dump(new_val, f)
        f.close()


def gpt3_select_examples():
    """select an example for each label and store into file"""
    in_file = "../data/jointslu/gpt3/examples.json"
    examples = []
    label_dict = read_jointslu_labels_dict()
    labels = list(label_dict.keys())
    # open pre-train training data file
    f = open("../data/jointslu/pre-train/b_train.json", 'r')
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


if __name__ == '__main__':
    # gpt3_from_bert_dataset()
    # oris = ["I-depart_date.today_relative","B-arrive_time.start_time","atis_distance","O"]
    # get_simplified_labels(oris)
    # change_labels2simplified()
    gpt3_select_examples()