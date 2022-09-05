from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')
candidates = ["want i", "fly to", "baltimore from", "dallas to", "trip round"]


def dep2candidates(sentence):
    """
    Parse the input snetence and return the list of candidates.
    input: sentence: the input sentence: str
    return: list of candidates: List(str)
    """
    pass


def read_labels_from_csv(file="../data/jointslu_labels.csv"):
    # open jointslu_labels.csv to read labels
    df = pd.read_csv(file,
                     usecols=['new_labels'])
    labels = df.values.tolist()
    label_set = set([i[0] for i in labels])
    label_list = list(label_set)
    return label_list


def sbert_dep_tree(candidates):
    """
    Find candidate-label pairs with candidates from a
    dependency tree and defined labels. sentence bert;
    similarity
    candidates: a list of candidate phrases
        example: ["want i", "fly to", "baltimore from",
        "dallas to", "trip round"]
    """
    # read label set
    label_list = read_labels_from_csv()
    candidates_embiddings = model.encode(candidates)
    labels_embiddings = model.encode(label_list)
    # calculate similarity
    cos_sim = util.cos_sim(candidates_embiddings,
                           labels_embiddings)
    # pair candidates and labels by similarity
    pairs = []
    for i in range(len(cos_sim) - 1):
        i_scores_tensor = cos_sim[i]
        i_scores = i_scores_tensor.tolist()
        max_score = max(i_scores)
        max_index = i_scores.index(max_score)
        pairs.append({
            "candidate": candidates[i],
            "labels": label_list[max_index]
        })
        print("candidate [", candidates[i], "] - [",
              label_list[max_index], "] label")

