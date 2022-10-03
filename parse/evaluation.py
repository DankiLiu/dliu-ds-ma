import util
from data.data_processing import read_jointslu_lines, jointslu_per_line
from parse.nltk_parser import dependency_parsing as dp
from parse.nltk_parser import name_entity_recognition as ner
from parse.nltk_parser import part_of_speech_parsing as pos

from sentence_transformers import SentenceTransformer
import torch


def sbert_ground_truth(candidates):
    pass


def phrases_from_dep_graph(words, dep_graph):
    phrases_dict = {}
    for i in range(len(words)):
        phrases_dict[i] = []
        # for loop start from 1 to sentence length
        dep_word = dep_graph.get_by_address(i+1)
        deps = dep_word['deps']
        dep_idxs = [ele - 1 for sublist in deps.values() for ele in sublist]
        for idx in dep_idxs:
            phrase = words[idx] + ' ' + words[i]
            phrases_dict[i].append(phrase)
    return phrases_dict


def parsing_evaluation(num=1):
    """Evaluate parsing with all labels"""
    if not util.server_is_running("http://localhost:9000/"):
        util.corenlp_server_start()
    dataset_path = "../data/sample.iob"
    # read dataset
    lines = read_jointslu_lines(dataset_path)
    # read labels
    labels_f = open("../data/jointslu_labels4parse.txt")
    all_labels = labels_f.readlines()
    # evaluation per example
    for n in range(num):
        sentence, words, labels = jointslu_per_line(lines[n])
        print(f"sentence {sentence}")
        dep_graph = dp(sentence)
        phrases = phrases_from_dep_graph(words, dep_graph)
        phrases_embs = {}
        # embedding phrases
        for i in range(len(words)):
            i_phrases = phrases.get(i)
            embs = sbert_embedding(i_phrases)
            phrases_embs[i] = embs
            if embs is not None:
                print(f"length embedding is: {len(embs)}")
        # embedding all labels
        labels_embs = sbert_embedding(all_labels)
        assert len(all_labels) == len(labels_embs)
        print(f"embedded all {len(labels_embs)} labels")
        # find label for each phrase with similarity
        cos = torch.nn.CosineSimilarity(dim=0)
        for ith in range(len(words)):
            ps_embs = phrases_embs.get(ith)
            if ps_embs is None:
                continue
            for p in ps_embs:
                sims = []
                for li in range(len(labels_embs)):
                    s = cos(torch.Tensor(p), torch.Tensor(labels_embs[li]))
                    sims.append(s)
                p_score = [i/sum(sims) for i in sims]
                index = p_score.index(max(p_score))
                print(phrases[ith])
                print(p_score)
                print("highest score> ", p_score[index])
                print(all_labels[index])


def sbert_embedding(phrases: list):
    # create sentence bert model
    if not phrases:
        return None
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode(phrases)
    return embs


def sbert_similarity(phrase, labels: list, model):
    """calculate similarities between the phrase and all labels"""



if __name__ == '__main__':
    parsing_evaluation()