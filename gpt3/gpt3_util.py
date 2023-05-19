import json
import os
from typing import List

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from data.data_processing import generate_gpt3_examples_file, get_samples, get_labels_dict, get_intents_dict
from evaluation.evaluation_utils import get_std_gt

PROMPT_1_ZEROSHOT = "Extract the intent and slots for the following user utterance and " \
                    "show them in key-value pairs format. {keys}\nutterance:{utterance}\noutput:"

PROMPT_1_1_ZEROSHOT = "Extract the intent and slots for the following user utterance and " \
                    "show them in key-value pairs format separated by \";\". {keys}\nutterance:{utterance}\noutput:"

PROMPT_1_ONESHOT = "Extract the intent and slots for the following user utterance and " \
                   "show them in key-value pairs format. {keys}\n{oneshot_example}" \
                   "\nutterance:{utterance}\noutput:"

KEYS = "The possible intents are {intents}, and the possible slots are {slots}."
ONESHOT_EXAMPLE = "An example is given: \nutterance:{exp_text}\noutput:{exp_gt}"

PROMPT_2_ZEROSHOT = "A machine receives a user utterance as a command and detects the user intent and extracts " \
                    "specific pieces of information or entities from the utterance. " \
                    "These pieces of information or entities are referred to as slots. " \
                    "The machine outputs them in key-value pairs format, " \
                    "such as \"intent:user_intent;slot1:entity1;slot2:entity2;...\"\n" \
                    "{keys}\nGiven the following user utterance, what is the machine output?\n" \
                    "utterance:{utterance}\noutput:"

PROMPT_2_ONESHOT = "A machine receives user utterances as commands and detects the user intent and extracts " \
                   "specific pieces of information or entities from the utterance. " \
                   "These pieces of information or entities are referred to as slots. " \
                   "The machine outputs them in key-value pairs format, " \
                   "such as \"intent:user_intent;slot1:entity1;slot2:entity2;...\"\n" \
                   "{keys}\n{oneshot_example}\nGiven the following user utterance, " \
                   "what is the machine output?\n" \
                   "utterance:{utterance}\noutput:"

ROBOT_TRANSLATOR = "A robot receives user utterances as commands, " \
                   "but it can only assist the user when user's intent and other key information in the utterance " \
                   "are translated into key-value pairs such as \"someKey:someValue;...\". In this format, each key describes a value, " \
                   "which is a word or phrase from the original sentence. {keys}" \
                   "\n{oneshot_example}" \
                   "Given user utterance:{utterance}\nI want to translate the utterance so that the " \
                   "robot understands it.\ntranslation:"

ONESHOT_ROBOT_TRANSLATOR = "for example:\nuser utterance:{exp_text}\ntranslation:{exp_gt}\n"


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


def construct_zeroshot_prompt(prompt, sentence, withkeys, dataset_info):
    """return the constructed prompt and stop sequence"""
    keys = get_KEY_by_task(dataset_info["dataset"],
                           dataset_info["labels_version"],
                           dataset_info["scenario"]) if withkeys else ""
    if prompt == "PROMPT_1_ZEROSHOT":
        return PROMPT_1_ZEROSHOT.format(keys=keys,
                                        utterance=sentence)
    elif prompt == "PROMPT_2_ZEROSHOT":
        return PROMPT_2_ZEROSHOT.format(keys=keys,
                                        utterance=sentence)
    elif prompt == "ROBOT_TRANSLATOR":
        return ROBOT_TRANSLATOR.format(keys=keys,
                                       oneshot_example="",
                                       utterance=sentence)


def get_KEY_by_task(dataset, labels_version, scenario):
    labels_dict = get_labels_dict(dataset, labels_version, scenario)
    intents_dict = get_intents_dict(dataset, labels_version, scenario)
    intents = '[' + ",".join(list(intents_dict.keys())) + ']'
    slot_list = list(labels_dict.keys())
    slot_list.remove('O')
    slots = '[' + ",".join(slot_list) + ']'
    return KEYS.format(intents=intents, slots=slots)


def construct_oneshot_prompt(prompt, exp_text, exp_gt, sentence, withkeys, dataset_info):
    """return the constructed prompt and stop sequence"""
    keys = get_KEY_by_task(dataset_info["dataset"],
                           dataset_info["labels_version"],
                           dataset_info["scenario"]) if withkeys else ""
    if prompt == "PROMPT_1_ONESHOT":
        return PROMPT_1_ONESHOT.format(
            keys=keys,
            oneshot_example=ONESHOT_EXAMPLE.format(
                exp_text=exp_text,
                exp_gt=exp_gt
            ),
            utterance=sentence
        )
    elif prompt == "PROMPT_2_ONESHOT":
        return PROMPT_2_ONESHOT.format(
            keys=keys,
            oneshot_example=ONESHOT_EXAMPLE.format(
                exp_text=exp_text,
                exp_gt=exp_gt
            ),
            utterance=sentence
        )
    elif prompt == "ROBOT_TRANSLATOR":
        return ROBOT_TRANSLATOR.format(keys=keys,
                                       oneshot_example=ONESHOT_ROBOT_TRANSLATOR.format(
                                        exp_text=exp_text,
                                        exp_gt=exp_gt),
                                       utterance=sentence)


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
