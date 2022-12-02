from typing import List

import pandas as pd

import util
from evaluation.evaluation_utils import get_std_gt, get_std_output_parsing

from parse.parsing import dependency_parsing
from parse.parsing import name_entity_recognition as ner
from parse.parsing import part_of_speech_parsing as pos

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import json

# same labels as in pre-train
f = open("data/jointslu/pre-train/labels.json")
data = json.load(f)
LABELS = list(data.keys())
f.close()


def phrases_from_dep_graph(words, dep_graph):
    """Construct phrases from parsing result.
    (dependency parsing, POS and NER)"""
    phrases_dict = {}
    for i in range(len(words)):
        phrases_dict[i] = []
        # for loop start from 1 to sentence length
        dep_word = dep_graph.get_by_address(i + 1)
        deps = dep_word['deps']
        # todo: here check the pos of the word and apply some rules
        dep_idxs = [ele - 1 for sublist in deps.values() for ele in sublist]
        for idx in dep_idxs:
            phrase = words[idx] + ' ' + words[i]
            phrases_dict[i].append(phrase)
    return phrases_dict


def plotting(labels=None, acc: List = None, f1: List = None):
    df = pd.read_csv("../data/jointslu/parsing/scores.csv")
    print(df.head())
    ndf = df.loc[df["num of examples"] == 1000]
    plt.ylim(0, 1.0)

    sns.barplot(x='num of examples',
                y='scores',
                data=ndf,
                palette='Paired',
                hue="type")

    plt.title('Accuracy and F1 scores of the parsing method')
    plt.show()


def main():
    pass
    # todo: save evaluation scores into a .csv data


def evaluation_per_label(num, shuffle):
    """"""
    # read results from file and evaluate results per label
    pass


def get_parsed_phrases(num, shuffle):
    """some examples are not usable due to length of dependency graph, this function
    returns num of examples: text, gpts and its generated phrases"""
    text, labels = get_examples("test", num=num, do_shuffle=shuffle)
    gts = [ground_truth(label.split(' ')) for label in labels]
    assert len(text) == len(gts)
    utext, ugts, parsed_phrases = [], [], []
    dep_result, _, ner_result = parse_examples(text)
    # ith_exp = 0
    # num_examples = 0
    # is_done = False
    for ith_exp in range(len(text)):
        assert len(text[ith_exp].split(' ')) == len(gts[ith_exp])
        dp_graph = dep_result[ith_exp]
        ner_labels = ner_result[ith_exp]
        # compare dependency graph length with labels length
        min_add = len(gts[ith_exp]) - 3
        address = min_add
        while dp_graph.contains_address(min_add):
            address = min_add
            min_add = min_add + 1
        # if length not match, skip this example
        if address != len(gts[ith_exp]):
            print("length not match")
            # ith_exp = ith_exp + 1
            continue
        phrases = interpret_dep(dp_graph, ner_labels, with_ner=True)
        parsed_phrases.append(phrases)
        utext.append(text[ith_exp])
        ugts.append(gts[ith_exp])
        '''
        num_examples = num_examples + 1
        ith_exp = ith_exp + 1
        if num_examples == num:
            is_done = True
        elif ith_exp == len(text):
            # if num of example is less than demand, but generated examples are all used, generate 20 more examples
            text, labels = get_examples("test", num=20, do_shuffle=True)
            gts = [ground_truth(label.split(' ')) for label in labels]
            ith_exp = 0
        '''
    return utext, ugts, parsed_phrases


def testing(num, shuffle):
    """return the parsing predictions and used text and gts,
    text: a list of input texts,
    gts: a list of simplified labels as gts of the text"""
    results = []
    utext, ugts, parsed_phrases = get_parsed_phrases(num, shuffle)
    print(f"parsing predicting {len(utext)} tests. ")
    length = len(utext)
    # load sbert model
    sbert = sbert_model()
    from datetime import date
    timestamp = str(date.today())
    for n in range(length):
        # find label for each phrase with similarity
        cosine_scores = cos_sim_per_example(parsed_phrases[n], LABELS, sbert)
        label_idxs = [torch.argmax(i_score).item() for i_score in cosine_scores]
        prediction = [LABELS[idx] for idx in label_idxs]
        std_output = get_std_output_parsing(parsed_phrases[n], prediction)
        std_gt = get_std_gt(utext[n], ugts[n])
        # todo: construct ground truth
        result = {
            "time_stamp": timestamp,
            "text": utext[n],
            "gt": ugts[n],
            "phrase": parsed_phrases[n],
            "prediction": prediction,
            "std_output": std_output,
            "std_gt": std_gt
        }
        print(f"result for {n}th example ", result)
        results.append(result)
    util.append_to_json("data/jointslu/parsing/parsing_output.json", results)
    print(f"{len(results)} results appended to parsing_output.json")


def evaluation(num=1, shuffle=True, pos=True, ner=True):
    """Evaluate parsing with all labels"""
    # read results from file and evaluate the parsing method
    pass


def f1_score(prediction: List, labels: List, mapping: dict):
    tp, fn, fp, tn = prediction_metrics(prediction,
                                        labels,
                                        mapping)
    recall = get_recall(tp, fn, fp, tn)
    precision = get_precision(tp, fn, fp, tn)
    return F1(precision, recall)


def accuracy_score(prediction: List, labels: List, mapping: dict):
    tp, fn, fp, tn = prediction_metrics(
        prediction,
        labels,
        mapping)

    print(f"tp {tp}, fn {fn}, fp {fp}, tn {tn}")
    return accuracy(tp, fn, fp, tn)


def F1(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2 * (recall * precision) / (recall + precision)


def accuracy(tp, fn, fp, tn):
    return (tp + tn) / (tp + fp + fn + tn)


def get_precision(tp, fn, fp, tn):
    if tp == 0:
        return 0
    return tp / (tp + fp)


def get_recall(tp, fn, fp, tn):
    if tp == 0:
        return 0
    return tp / (tp + fn)


def prediction_metrics_per_label(predictions: List, labels: List, label):
    """return TP, FN, FP, TN of n selected label.
    preditions: predicted labels for n examples,
    labels: ground truth labels for n examples,
    label: the label to be evaluated"""
    TP, FN, FP, TN = 0, 0, 0, 0
    num = len(predictions)
    for i in range(num):
        cur_prediction, cur_labels = predictions[i], labels[i]
        print(cur_prediction)
        print(cur_labels)
        # if the label is in labels, check if it is predicted
        if label in cur_labels:
            if label in cur_prediction:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            # if the label is not in labels, check if it is wrongly predicted
            if label in cur_prediction:
                FP = FP + 1
            else:
                TN = TN + 1
    return TP, FN, FP, TN


def prediction_metrics(prediction: List, labels: List, mapping: dict):
    """return TP, FN, FP, TN of the prediction (one prediction)
    prediction: a list of of labels for phrases
    labels: a list of simplified labels for current sentence (gt),
    mapping: a dict that maps phrase index to a list of words (idxs in
    original sentence)"""
    tp, fn, fp, tn = 0, 0, 0, 0
    keys, values = mapping.keys(), mapping.values()
    # idxs that are in the phrase and classified
    labeled_idx = list(set([item for sublist in values for item in sublist]))

    for i in range(len(prediction)):
        ''' for predicted words, check whether it is classified right '''
        if i in keys:
            # words in a phrase
            word_idxs = mapping.get(i)
            gt = []
            for idx in word_idxs:
                if labels[idx] != 'O':
                    gt.append(labels[idx])
            ''' if the phrase is classified right, then fn plus one,
            for each not classified label, fn plus one '''
            # todo: partial classification
            # todo: if this label is classified else where

            if prediction[i] != 'O' and prediction[i] in list(set(gt)):
                tp = tp + 1
                fn = fn + len(set(gt)) - 1
            else:
                fn = fn + len(set(gt))
        elif i not in labeled_idx:
            ''' word at this index are not classified,
            if the original label is 'O', the tn plus one,
            if the original label if not 'O', the fn plus one '''
            if labels[i] == 'O':
                tn = tn + 1
            else:
                fn = fn + 1
    return tp, fn, fp, tn


def ground_truth(labels):
    """return a list of labels as ground truth"""
    f = open("data/jointslu/labels.csv", 'r')
    import csv
    data = csv.reader(f)
    # generate labels-gt match with dict
    label2gt = {}
    for line in data:
        label2gt[line[0]] = line[1]
    gt = []
    for label in labels:
        try:
            gt.append(label2gt[label])
        except KeyError:
            gt.append('O')
    f.close()
    return gt


def parse_examples(text: List):
    """return the parsing result as a list"""
    # load nlp server
    if not util.server_is_running("http://localhost:9000/"):
        util.corenlp_server_start()
    dep_result = []
    pos_result = []
    ner_result = []
    for sentence in text:
        parsed_dep = dependency_parsing(sentence)
        parsed_pos = pos(sentence)
        parsed_ner = ner(sentence)
        dep_result.append(parsed_dep)
        pos_result.append(parsed_pos)
        ner_result.append(parsed_ner)
    return dep_result, pos_result, ner_result


def plot_sim_score(phrase, labels, cosine_scores):
    """plot cosine scores on 2D image"""
    x = torch.linspace(-5, 5, 100)
    x_squared = x * x

    plt.plot(x, x_squared)  # Fails: 'Tensor' object has no attribute 'ndim'

    torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it

    plt.plot(x, x_squared)  # Works now
    plt.show()


def get_label_set():
    """return a set (list type) of labels from labels.csv file"""
    f = open("../data/jointslu/labels.csv", 'r')
    import csv
    data = csv.reader(f)
    labels = set([line[1] for line in data])
    print(f"{len(labels)} labels: {labels}")
    f.close()
    return list(labels)


def interpret_dep(dependency_graph, ner_labels, with_ner=False):
    """return a list of phrases constructed from dependency graph"""
    phrases = []
    address = 1
    # if current index has phrase, create a list of the words in the phrase
    # and store under the index key
    matching_dict = {}
    while True:
        word_dict = dependency_graph.get_by_address(address)
        if word_dict['address'] is None:
            break
        phrase = ""
        gt = ""
        # skip this word or not
        if not pos_skip(word_dict):
            # check dependency type, dep is a dep_dict
            idxs = []
            for key, value in word_dict['deps'].items():
                if not dep_skip(key):
                    # if not skip, construct phrase
                    for idx in value:
                        idxs.append(idx)
            idxs.append(address)
            idxs.sort()
            phrase = construct_phrase(dependency_graph, idxs)
        if with_ner and phrase != "":
            if ner_labels[address - 1][1] != "O":
                phrase = phrase + ' ' + ner_labels[address - 1][1]
        phrases.append(phrase)
        address = address + 1
    return phrases


def construct_phrase(dependency_graph, idxs):
    words = []
    for i in idxs:
        words.append(dependency_graph.get_by_address(i)['word'])
    return ' '.join(words)


def pos_skip(word_dict):
    """return True if the word has no dependency or labeled
    as VPB, otherwise False"""
    if len(word_dict["deps"]) == 0:
        return True
    if word_dict["ctag"] == "VBP" or word_dict["ctag"] == "VB":
        return True
    # if word_dict["ctag"] == "VB" -> goal action
    return False


def dep_skip(dep):
    skip_list = ["nsubj", "conj", "mark", "obl", "cc", "nmod", "iobj", "cop", "det"]
    if dep in skip_list:
        return True
    return False


def get_examples(data_type, num=0, do_shuffle=False):
    """get example from parsing/train.json"""
    file_name = "data/jointslu/parsing/train.json"
    if data_type == "test":
        file_name = "data/jointslu/parsing/test.json"
    elif data_type == "val":
        file_name = "data/jointslu/parsing/val.json"

    f = open(file_name, 'r')
    import json
    from random import shuffle
    # todo: not a real shuffle
    data = json.load(f)
    length = len(data) if num == 0 or num > len(data) else num
    print("get ", length, " examples")
    if do_shuffle:
        shuffle(data)
    text = [i["text"] for i in data[0:length]]
    labels = [i["labels"] for i in data[0:length]]
    return text, labels


def cos_sim_per_example(phrases: List, labels: List, sbert):
    """
    return similarity dict of sentence phrases and labels.
    input: phrases: a list of phrases
           labels: a list of labels
    output: a matrix of cosine similarities
    """
    # encode labels
    label_embs = sbert.encode(labels, convert_to_tensor=True)
    phrase_embs = sbert.encode(phrases, convert_to_tensor=True)
    cosine_scores = cos_sim(phrase_embs, label_embs)
    return cosine_scores


def sbert_encoding(phrases: list):
    # create sentence bert model
    if not phrases:
        return None
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode(phrases)
    return embs


def sbert_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def accuracy_sim(predicted, gt):
    """return accuracy of the model in percentage """
    assert len(predicted) == len(gt)
    length = len(predicted)
    po_num = 0
    for i in range(length):
        if predicted[i] in gt[i] or predicted[i] == gt[i]:
            po_num = po_num + 1
    return po_num / length


def accuracy_gt(predicted, gt):
    """return accuracy of the model in percentage """
    assert len(predicted) == len(gt)
    length = len(predicted)
    po_num = 0
    for i in range(length):
        if predicted[i] == gt[i]:
            po_num = po_num + 1
    return po_num / length


