from random import randint
from typing import List

from evaluation.evaluation_utils import get_std_gt, std_gpt3_example
from gpt3.gpt3_util import get_example_by_sim, load_examples, \
    construct_oneshot_example, construct_oneshot_prompt, get_samples_gpt3, \
    construct_zeroshot_prompt, get_labels_ts_stdgts
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


def zero_shot(prompt, sentences: List, model_engine):
    GPT3_Completion(prompt, sentences)
    results = []
    # gpt3 access limit 60 times per minute
    for i in range(len(sentences)):
        time.sleep(2.0)
        prompt_i, stop_sequence = construct_zeroshot_prompt(prompt, sentences[i])
        try:
            response = openai.Completion.create(engine=model_engine,
                                                prompt=prompt_i,
                                                temperature=0,
                                                max_tokens=256,
                                                top_p=1.0,
                                                stop_sequence=stop_sequence)
            results.append(response["choices"][0]["text"])
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError")
    return results


def one_shot(prompt, sentences: List, exp_texts, exp_gts: List, model_engine):
    # should be one example for each sentence
    assert len(sentences) == len(exp_texts)
    print(f"one-shot predicting for {len(sentences)} tests")
    results = []
    # gpt3 access limit 60 times per minute
    for i in range(len(sentences)):
        time.sleep(2.0)
        prompt_i = construct_oneshot_prompt(prompt, exp_texts[i], exp_gts[i], sentences[i])
        try:
            response = openai.Completion.create(engine=model_engine,
                                                prompt=prompt_i,
                                                temperature=0,
                                                max_tokens=256,
                                                top_p=1.0)
            results.append(response["choices"][0]["text"])
        except openai.error.RateLimitError:
            print(openai.error.RateLimitError.user_message)
            print(openai.error.RateLimitError.args)
    return results


def one_shot_single(prompt, sentence, exp_text, exp_gt, model_engine):
    """return the gpt3 response given prompt name, the input sentence and one example, model engine
    shows which gpt3 model is used
    one input sentence is given to produce output"""
    # todo: different prompt has different solution
    prompt_i = construct_oneshot_prompt(prompt, exp_text, exp_gt, sentence)
    response = openai.Completion.create(engine=model_engine,
                                        prompt=prompt_i,
                                        temperature=0,
                                        max_tokens=256,
                                        top_p=1.0)
    return response["choices"][0]["text"]


def gpt3jointslu(num, prompt, model_name, testing_file, path, select, labels_version):
    """num: number of input texts to be tested,
    model_version: the self-defined model version number, described in model_version.json
    select==True: choose examples that is similar to given input text"""
    # load test examples
    labels, ts, std_gts = get_labels_ts_stdgts(testing_file, labels_version, num, "test")
    num_examples = len(ts)
    global exp_texts, exp_labels
    print(f"--- [gpt3] testing {len(ts)} examples ---")
    # construct texts and ground truths
    # load examples if choose==True
    examples = load_examples(dataset="jointslu",
                             labels_version=labels_version)
    if select or select == "True":
        # For each sentence, find an example by similarity
        print("%% choose example by similarity")
        examples = get_example_by_sim(ts, examples)
        exp_texts, exp_gts = construct_oneshot_example(examples)
    else:
        print("%% choose random examples")
        # choose random example for each text
        # todo: here should pick from selected examples or from all training data?
        idxs = [randint(0, len(examples) - 1) for _ in range(len(ts))]
        exp_texts, exp_gts = construct_oneshot_example([examples[i] for i in idxs])
    for i in range(num_examples):
        response = one_shot_single(prompt, ts[i], exp_texts[i], exp_gts[i], model_name)
        result = {
            "num": i,
            "prompt": prompt,
            "exp_text": exp_texts[i],
            "exp_gt": exp_gts[i],
            "text": ts[i],
            "labels": labels[i],
            "prediction": response,
            "std_gt": std_gts[i]
        }
        append_to_json(file_path=path, new_data=result)
        time.sleep(3.5)
