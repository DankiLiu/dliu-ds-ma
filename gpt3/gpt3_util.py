import json
from typing import List

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def load_examples():
    """
    load examples selected from training data
    examples: List
        {"id": *,
        "text": *,
        "labels": *}
    """
    file = open("../data/jointslu/gpt3/examples.json", 'r')
    data = json.load(file)
    examples = []
    for exp in data:
        if exp["example"] is None:
            continue
        text = exp["example"]["text"]
        labels = exp["example"]["labels"]
        # remove BOS and EOS
        text = text[1: len(text) - 1]
        labels = labels[1: len(labels) - 1]
        example = {"id": exp["example"]["id"],
                   "text": text,
                   "labels": labels}
        examples.append(example)
    return examples


def get_example_by_sim(texts: List, examples):
    """given a list of input text,
    return a list of examples similar with the input text"""
    sentences = [''.join(i["text"]) for i in examples]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text_en = model.encode(texts)
    sens_en = model.encode(sentences)
    cosine_scores = cos_sim(text_en, sens_en)
    # score should be shape (len(texts), len(sentences))
    print("score has shape ", cosine_scores.shape)
    # todo: matrix is tensor or array
    idxs = []
    for i in range(len(texts)):
        # for each text, find an example for the text
        max_score = -1
        max_idx = -1
        for idx in range(len(sentences)):
            if cosine_scores[i][idx] > max_score:
                max_score = cosine_scores[i][idx]
                max_idx = idx
        print(f"text {texts[i]}\nexample{sentences[max_idx]}")
        idxs.append(max_idx)
    return [examples[idx] for idx in idxs]


def get_examples_gpt3(data_type="train", num=1, do_shuffle=True):
    """return a list of text and a list of its corresponding labels,
    return one example by default"""
    # get example from parsing_eval/train.json by default
    file_name = "../data/jointslu/pre-train/b_train.json"
    if data_type == "test":
        file_name = "../data/jointslu/pre-train/b_test.json"
    elif data_type == "val":
        file_name = "../data/jointslu/pre-train/b_val.json"

    f = open(file_name, 'r')
    import json
    from random import shuffle
    data = json.load(f)
    length = len(data) if num == 0 else num
    print("get ", length, " examples")
    if do_shuffle:
        shuffle(data)
    # remove BOS and EOS
    text = [i["text"][1: len(i["text"]) - 1] for i in data[0:length]]
    labels = [i["labels"][1: len(i["text"]) - 1] for i in data[0:length]]
    print(f"text: {text}\nlabels: {labels}")
    return text, labels


def get_example_keyword_pair(data_type, num=0, do_shuffle=False):
    text, labels = get_examples_gpt3(data_type, num, do_shuffle)
    outputs = []
    for i, label in enumerate(labels):
        text_split = text[i].split(' ')
        output = ""
        # todo: merge the same label together
        label_tok = label.split(',')
        for idx, tok in enumerate(label_tok):
            if tok != 'O':
                output = output + text_split[idx] + ':' + tok + ";"
        print(f"{i}th text {text_split[i]}\noutput {output}")
        outputs.append(output)
    return text, outputs


def construct_oneshot_example(examples: List):
    """return a list of one shot examples given number"""
    t_list, l_list = [], []

    for i in range(len(examples)):
        text = [t for t in examples[i]["text"]]
        labels = [l for l in examples[i]["labels"]]
        t_list.append(text)
        l_list.append(labels)
    examples = []
    for i in range(len(t_list)):
        input_text, output_labels = label4phrase(t_list[i], l_list[i])
        print("input_text: ", input_text)
        print("output_labels: ", output_labels)
        example = "input: " + input_text + '\n' + "output: " + output_labels
        examples.append(example)
    return examples


def label4phrase(text, labels):
    """
    return text and constructed output
    example: two Lists
    output: input_text: Str; output_labels: Str
        output_labels is in form label1: phrase1; label2: phrase2
    """
    # example:
    # input_text:
    #   flights from pittsburgh to baltimore arriving between 4 and 5 pm
    # output_labels:
    #   from city: pittsburgh; to city: baltimore; arrive time: 4; arrive time end: 5 pm;
    if not len(text) == len(labels):
        return None
    d = {}
    for i in range(len(text)):
        if labels[i] == 'O':
            continue
        if labels[i] in d.keys():
            d[labels[i]] = d[labels[i]] + ' ' + text[i]
        else:
            d[labels[i]] = text[i]
    input_text = ' '.join(text)
    output_labels = ''
    for k, v in d.items():
        output_labels = output_labels + str(k) + ': ' + str(v) + '; '
    return input_text, output_labels


if __name__ == '__main__':
    pass