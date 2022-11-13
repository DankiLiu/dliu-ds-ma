from random import randint
from typing import List
from gpt3.gpt3_util import get_example_by_sim, load_examples, \
    construct_oneshot_example, get_examples_gpt3, label4phrase
import openai
import os
import json

from parse.evaluation import get_examples

API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    print("No API KEY found")
    exit()
openai.api_key = API_KEY

PROMPT_1 = "Extract the important information from this text and show them using key-value pairs.\n"
PROMPT_2 = "Extract the intention of the user from the input text, an example is given: \n"


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
    results = []
    for i in range(len(sentences)):
        prompt_i = prompt + oneshot_examples[i]
        sentence = "input: " + sentences[i]
        prompt_i = prompt_i + '\n' + sentence
        print("prompt:\n", prompt_i)

        response = openai.Completion.create(engine="text-davinci-002",
                                            prompt=prompt_i,
                                            temperature=0,
                                            max_tokens=256,
                                            top_p=1.0)
        results.append(response["choices"][0]["text"])
        print("result:\n", response["choices"][0]["text"])
    return results


def gpt3jointslu(num, choose=True):
    """num: number of input texts to be tested,
    choose==Ture: choose examples that is similar to given input text"""
    results = []
    # load test examples
    global oneshot_examples
    print(f"--- testing {num} examples ---")
    texts, labels = get_examples_gpt3("test", num=num)
    # construct texts and ground truths
    ts, lgts = [], []
    for i in range(num):
        t, lgt = label4phrase(texts[i], labels[i])
        ts.append(t)
        lgts.append(lgt)
    # load examples if choose==True
    examples = load_examples()
    if choose:
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
    responses = one_shot(PROMPT_2, ts, oneshot_examples)
    for i in range(num):
        result = {
            "text": ts[i],
            "labels": labels[i],
            "lgt": lgts[i],
            "response": responses[i]
        }
        print(f"%%%{i}th {result}")
        results.append(result)
    path = "../gpt3/results.json"
    file = open(path, 'r+')
    stored = json.load(file)
    to_store = stored + results
    file.seek(0)
    json.dump(to_store, file, indent=4)
    file.close()


if __name__ == '__main__':
    gpt3jointslu(50)
