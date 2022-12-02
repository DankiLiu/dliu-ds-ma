from random import randint
from typing import List

from evaluation.evaluation_utils import get_std_gt, std_gpt3_example
from gpt3.gpt3_util import get_example_by_sim, load_examples, \
    construct_oneshot_example, get_examples_gpt3, get_oneshot_prompt
import openai
import os
import time

from util import append_to_json

API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    print("No API KEY found")
    exit()
openai.api_key = API_KEY

PROMPT_1 = "Extract the important information from this text and show them using key-value pairs.\n"
PROMPT_2 = "Extract the intention of the user from the input text, an example is given: \n"
PROMPT_3 = "Please summary the intention of the user in keyword form from the input text, an example is given: \n"


def GPT3_Completion(prompt, sentences):
    responses = []
    print(prompt)
    for i in sentences:
        print(i)
        new_prompt = prompt + '\n' + i
        response = openai.Completion.create(engine="text-davinci-002",
                                            prompt=new_prompt,
                                            temperature=0,
                                            max_tokens=256,
                                            top_p=1.0)
        responses.append(response)
    return responses


def zero_shot(prompt, sentences: List):
    GPT3_Completion(prompt, sentences)


def one_shot(prompt, sentences: List, oneshot_examples: List):
    # should be one example for each sentence
    assert len(sentences) == len(oneshot_examples)
    print(f"one-shot predicting for {len(sentences)} tests")
    results = []
    # gpt3 access limit 60 times per minute
    for i in range(len(sentences)):
        time.sleep(2.0)
        prompt_i = get_oneshot_prompt(prompt, oneshot_examples[i], sentences[i])
        try:
            response = openai.Completion.create(engine="text-davinci-002",
                                                prompt=prompt_i,
                                                temperature=0,
                                                max_tokens=256,
                                                top_p=1.0)
            results.append(response["choices"][0]["text"])
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError")
    return results


def one_shot_per_example(prompt, sentence, oneshot_example):
    prompt_i = get_oneshot_prompt(prompt, oneshot_example, sentence)
    try:
        response = openai.Completion.create(engine="text-davinci-002",
                                            prompt=prompt_i,
                                            temperature=0,
                                            max_tokens=256,
                                            top_p=1.0)
        print(f"response for \n{sentence}\n", response["choices"][0]["text"])
        return response["choices"][0]["text"]
    except openai.error.RateLimitError:
        print("rate limit error")


def gpt3jointslu(num, select=True, file_path="data/jointslu/gpt3/gpt3_output.json"):
    """num: number of input texts to be tested,
    select==Ture: choose examples that is similar to given input text"""
    # load test examples
    global oneshot_examples
    texts, labels = get_examples_gpt3("test", num=num)
    num_examples = len(texts)
    print(f"--- testing {len(texts)} examples ---")
    # construct texts and ground truths
    std_gts = []
    ts = [' '.join(t) for t in texts]
    for i in range(num_examples):
        std_gt = get_std_gt(ts[i], labels[i])
        std_gts.append(std_gt)
    # load examples if choose==True
    examples = load_examples()
    if select:
        # For each sentence, find an example by similarity
        print("%%choose example by similarity")
        examples = get_example_by_sim(ts, examples)
        oneshot_examples = construct_oneshot_example(examples)
    else:
        print("%%choose random examples")
        # choose random example for each text
        # todo: here should pick from selected examples or from all training data?
        idxs = [randint(0, len(examples) - 1) for _ in range(len(ts))]
        oneshot_examples = construct_oneshot_example([examples[i] for i in idxs])
    for i in range(num_examples):
        response = one_shot_per_example(PROMPT_2, ts[i], oneshot_examples[i])
        from datetime import date
        timestamp = str(date.today())
        exp_text, exp_gt = std_gpt3_example(examples[i])
        result = {
            "timestamp": timestamp,
            "prompt": {"name": "PROMPT_3", "text": PROMPT_3},
            "exp_text": exp_text,
            "exp_gt": exp_gt,
            "text": ts[i],
            "labels": labels[i],
            "prediction": response,
            "std_gt": std_gts[i]
        }
        append_to_json(file_path=file_path, data=[result])
        print(f"result is appended \n{result}")
        time.sleep(2.0)
