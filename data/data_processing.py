import json
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
    f = open("../data/jointslu_dict.json")
    labels_dict = json.load(f)
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


def label_set():
    labels = ['depart date',
              'fare amount',
              'from state',
              'atis distance',
              'ground fare',
              'day name',
              'stop state',
              'atis airport',
              'arrive time start',
              'flight time',
              'arrive time',
              'month',
              'relative to today',
              'to airport',
              'days code',
              'return time',
              'to state',
              'airline name',
              'aircraft code',
              'airport name',
              'stop location',
              'return date',
              'state code',
              'transport type',
              'flight stop',
              'economy',
              'atis aircraft',
              'flight days',
              'atis restriction',
              'relative time',
              'restriction code',
              'alternative',
              'airline code',
              'O',
              'flight number',
              'airport code',
              'state name',
              'atis quantity',
              'to city',
              'ground service',
              'in city',
              'connect',
              'from city',
              'period of day',
              'fare basis code',
              'relative cost',
              'from airport',
              'atis abbreviation',
              'atis airline',
              'depart time',
              'arrive time end',
              'atis capacity',
              'atis meal',
              'at time',
              'round trip',
              'superlative',
              'number of days',
              'atis cheapest',
              'class type',
              'from location',
              'to state code',
              'meal code',
              'stop city',
              'atis flight',
              'atis airfare',
              'meal description',
              'meal',
              'arrive date',
              'superlative',
              'to contuntry']
    return labels


if __name__ == '__main__':
    gpt3_from_bert_dataset()
