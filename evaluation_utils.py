"""
1. define input and output
2. load model
3. model output post-processing
4. evaluation functions
"""
from typing import List


def get_std_gt(text: str, labels: List):
    """
    Construct standard ground truth of the given example, given input text and original labels (simplified).
    :text: (str) A input text string
    :labels: (list) A list of simplified labels
    :return standard ground truth of the given example
    """
    # example:
    # input_text:
    #   flights from pittsburgh to baltimore arriving between 4 and 5 pm
    # output_labels:
    #   from city: pittsburgh; to city: baltimore; arrive time: 4; arrive time end: 5 pm;
    text = text.split(' ')
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
    output_labels = ''
    for k, v in d.items():
        output_labels = output_labels + str(k) + ': ' + str(v) + '; '
    return output_labels


def get_std_output_parsing(phrases: List, prediction: List):
    """
    Construct standard ground truth of the given example for Parsing method,
    given input text and original labels (simplified).
    :phrases: (list) A list of phrases generated from Parsing
    :prediction: (list) A list of simplified labels predicted by Parsing
    :return standard ground truth of the given example
    """
    assert len(phrases) == len(prediction)
    output = []
    for i, p in enumerate(phrases):
        if p != "":
            output.append(prediction[i] + ': ' + p)
    output = '; '.join(output)
    output = output.strip()
    return output


def std_gpt3_example(example):
    text = example["text"]
    labels = example["labels"]
    text = ' '.join(text)
    gt = get_std_gt(text, labels)
    return text, gt