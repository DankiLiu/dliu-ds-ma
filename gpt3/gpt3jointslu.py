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


def one_shot(prompt, sentences: List, lgt):
    examples = load_examples()
    # For each sentence, find an example by similarity
    examples = get_example_by_sim(sentences, examples)
    print(f"chosen example: \n{examples}")
    oneshot_examples = construct_oneshot_example(examples)
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
        result = {
            "text": sentence,
            "response": response["choices"][0]["text"],
            "gt": lgt[i]
        }
        print("result:\n", response)
        results.append(result)
    # store results in file for calculating the loss
    with open("results.json", 'a') as f:
        for result in results:
            json.dump(result, f, indent=4)
        f.close()


def gpt3jointslu():
    # generate the input sentences and its ground truth
    # get response given the sentences and a prompt
    # compare the responses with the ground truth
    pass


if __name__ == '__main__':
    texts, labels = get_examples_gpt3("test", num=1)
    t, lgt = label4phrase(texts[0], labels[0])
    print("text: ", t)
    print("gt: ", lgt)
    sentences = [' '.join(texts[i]) for i in range(len(texts))]
    one_shot(PROMPT_2, sentences, [lgt])

