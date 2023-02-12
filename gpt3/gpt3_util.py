import json
import os
from typing import List

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from data.data_processing import generate_gpt3_examples_file, get_samples
from evaluation.evaluation_utils import get_std_gt

PROMPT_1 = "Extract the important information from this text and show them using key-value pairs.\n"
PROMPT_2 = "Extract the intention of the user from the input text, an example is given: \n"
PROMPT_3 = "Please summary the intention of the user in keyword form from the input text\n{oneshot}input>{" \
           "input_sentence}\noutput> "

ONESHOT_PROMPT_3 = ", an example is given:\ninput>{exp_text}\noutput>{exp_gt}"
ROBOT_TRANSLATOR = "A robot is given a command in natural language sentence, " \
                   "but it can only help the user when the important information that shows user's intention is " \
                   "translated to key-value pairs, such as \"someKey: someValue; ...\", where values are a word " \
                   "or a phrase in the command sentence and keys are labels describe the values.\n{oneshot}" \
                   "given command sentence:{input_sentence}\nI want to translate the command sentence so that the " \
                   "robot understands it.\ntranslation:"

ONESHOT_ROBOT_TRANSLATOR = "for example:\ncommand sentence:{exp_text}\ntranslation:{exp_gt}\n"


def load_examples(dataset, labels_version, scenario):
    """
    load examples selected from training data
    examples: List
        {"id": *,
        "text": *,
        "labels": *}
    """
    path = check_example_file(dataset, labels_version, scenario=scenario)

    file = open(path, 'r')
    data = json.load(file)
    examples = []
    for exp in data:
        if exp["example"] is None:
            continue
        intent = exp["example"]["intent"]
        text = exp["example"]["text"]
        labels = exp["example"]["labels"]
        # remove BOS and EOS
        text = text[1: len(text) - 1]
        labels = labels[1: len(labels) - 1]
        example = {"id": exp["example"]["id"],
                   "intent": intent,
                   "text": text,
                   "labels": labels}
        examples.append(example)
    return examples


def check_example_file(dataset, labels_version, scenario):
    # check the existence of examples file with labels_version
    folder_name = 'data/' + dataset + '/gpt3/'
    file_name = ""
    if dataset == "jointslu":
        file_name = 'examples' + labels_version + '.json'
    elif dataset == "massive":
        file_name = scenario + "_examples" + labels_version + ".json"
    path = folder_name + file_name
    if not os.path.exists(path):
        generate_gpt3_examples_file(dataset=dataset,
                                    labels_version=labels_version,
                                    in_file=path,
                                    scenario=scenario)
    return path


def get_example_by_sim(texts: List, examples):
    """given a list of input text,
    return a list of examples similar with the input text"""
    sentences = [''.join(i["text"]) for i in examples]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text_en = model.encode(texts)
    sens_en = model.encode(sentences)
    cosine_scores = cos_sim(text_en, sens_en)
    # score should be shape (len(texts), len(sentences))
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
        idxs.append(max_idx)
    return [examples[idx] for idx in idxs]


def get_labels_ts_stdgts(testing_file, num, do_shuffle=False):
    """return texts (str) and standard ground truth for selected samples"""
    texts, labels, intents = get_samples(file_path=testing_file,
                                         model_name="gpt3",
                                         num=num,
                                         do_shuffle=do_shuffle)
    std_gts = []
    ts = [' '.join(text) for text in texts]
    for i in range(len(texts)):
        std_gt_w = get_std_gt(texts[i], labels[i], intents[i])
        std_gts.append(std_gt_w)
    return labels, ts, std_gts


def construct_zeroshot_prompt(prompt, sentence):
    """return the constructed prompt and stop sequence"""
    if prompt == "PROMPT_3":
        return PROMPT_3.format(oneshot="",
                               input_sentence=sentence)
    elif prompt == "ROBOT_TRANSLATOR":
        return ROBOT_TRANSLATOR.format(oneshot="",
                                       input_sentence=sentence)


def construct_oneshot_prompt(prompt, exp_text, exp_gt, sentence):
    """return the constructed prompt and stop sequence"""
    if prompt == "PROMPT_3":
        return PROMPT_3.format(oneshot=ONESHOT_PROMPT_3.format(exp_text=exp_text,
                                                               exp_gt=exp_gt),
                               input_sentence=sentence)
    elif prompt == "ROBOT_TRANSLATOR":
        return ROBOT_TRANSLATOR.format(oneshot=ONESHOT_ROBOT_TRANSLATOR.format(exp_text=exp_text,
                                                                               exp_gt=exp_gt),
                                       input_sentence=sentence)


def construct_oneshot_example(examples: List):
    """return a list of one shot examples given number"""
    t_list, l_list, i_list = [], [], []

    for i in range(len(examples)):
        text = [t for t in examples[i]["text"]]
        labels = [l for l in examples[i]["labels"]]
        intent = examples[i]["intent"]
        t_list.append(text)
        l_list.append(labels)
        i_list.append(intent)
    exp_texts = [' '.join(t_list[i]) for i in range(len(t_list))]
    exp_gts = [get_std_gt(t_list[i], l_list[i], i_list[i]) for i in range(len(t_list))]
    return exp_texts, exp_gts


def read_output_from_file(path="data/jointslu/gpts/gpt3_output.json"):
    file = open(path, 'r')
    data = json.load(file)
    file.close()
    return data
