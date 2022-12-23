from random import randint
from typing import List

from evaluation.evaluation_utils import get_std_gt, std_gpt3_example
from gpt3.gpt3_util import get_example_by_sim, load_examples, \
    construct_oneshot_example, get_examples_gpt3, construct_oneshot_prompt
import openai
import os
import time

from util import append_to_json

API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    print("No API KEY found")
    exit()
openai.api_key = API_KEY


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


def one_shot(prompt, sentences: List, oneshot_examples: List, model_engine):
    # should be one example for each sentence
    assert len(sentences) == len(oneshot_examples)
    print(f"one-shot predicting for {len(sentences)} tests")
    results = []
    # gpt3 access limit 60 times per minute
    for i in range(len(sentences)):
        time.sleep(2.0)
        prompt_i = get_oneshot_prompt(prompt, oneshot_examples[i], sentences[i])
        try:
            response = openai.Completion.create(engine=model_engine,
                                                prompt=prompt_i,
                                                temperature=0,
                                                max_tokens=256,
                                                top_p=1.0)
            results.append(response["choices"][0]["text"])
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError")
    return results


def one_shot_per_example(prompt, sentence, exp_text, exp_gt, model_engine):
    # todo: different prompt has different solution
    prompt_i, stop_sequence = construct_oneshot_prompt(prompt, exp_text, exp_gt, sentence)
    print("prompt i for gpt3 one shot example: \n", prompt_i)
    try:
        response = openai.Completion.create(engine=model_engine,
                                            prompt=prompt_i,
                                            temperature=0,
                                            max_tokens=256,
                                            top_p=1.0,
                                            stop_seq=stop_sequence)
        print(f"response for \n{sentence}\n", response["choices"][0]["text"])
        return response["choices"][0]["text"]
    except openai.error.RateLimitError:
        print("rate limit error")


def gpt3jointslu(num, prompt, model_name, path, select):
    "data/jointslu/gpt3/gpt3_output.json"
    """num: number of input texts to be tested,
    model_version: the self-defined model version number, described in model_version.json
    select==Ture: choose examples that is similar to given input text"""
    # if file path exists, create a new file

    # load test examples
    global exp_texts, exp_labels
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
    if select or select == "True":
        # For each sentence, find an example by similarity
        print("%%choose example by similarity")
        examples = get_example_by_sim(ts, examples)
        exp_texts, exp_gts = construct_oneshot_example(examples)
    else:
        print("%%choose random examples")
        # choose random example for each text
        # todo: here should pick from selected examples or from all training data?
        idxs = [randint(0, len(examples) - 1) for _ in range(len(ts))]
        exp_texts, exp_gts = construct_oneshot_example([examples[i] for i in idxs])
    for i in range(num_examples):
        response = one_shot_per_example(prompt, ts[i], exp_texts[i], exp_gts[i], model_name)
        from datetime import date
        timestamp = str(date.today())
        exp_text, exp_gt = std_gpt3_example(examples[i])
        result = {
            "timestamp": timestamp,
            "prompt": prompt,
            "exp_text": exp_text,
            "exp_gt": exp_gt,
            "text": ts[i],
            "labels": labels[i],
            "prediction": response,
            "std_gt": std_gts[i]
        }
        append_to_json(file_path=path, data=[result])
        print(f"result is appended \n{result}")
        time.sleep(2.0)
