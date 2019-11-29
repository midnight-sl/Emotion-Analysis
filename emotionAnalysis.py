import spacy
from typing import Tuple, Dict
from collections import defaultdict
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import numpy as np

nlp = spacy.load('en_core_web_md')

PATH_TO_FOUR_CLASSES_DATA = './NRC-Sentiment-Emotion-Lexicons/' \
                            'NRC-Affect-Intensity-Lexicon/NRC-AffectIntensity-Lexicon.csv'

PATH_TO_EIGHT_CLASSES_DATA = './NRC-Sentiment-Emotion-Lexicons/' \
                             'NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.csv'

POS_MULTIPLIER = {'ADJ': 0.9, 'ADV': 0.8, 'VERB': 0.75, 'NOUN': 0.7}

# reading first dataset
four_classes = pd.read_csv(PATH_TO_FOUR_CLASSES_DATA, names=['token', 'score', 'label'], sep=r"\s+")
# reading second dataset
eight_classes = pd.read_csv(PATH_TO_EIGHT_CLASSES_DATA, names=['token', 'label', 'score'], sep=r"\s+")

# get parts of speech


# clean text
def clean_text(text: str) -> str:
    doc = nlp(text)
    # removing stop words and punctuation
    tokens = ' '.join(
        [re.sub('\n', '', token.text.lower()) for token in doc if not token.is_stop and not token.is_punct])
    return tokens


def eval_text(text: str, dataset: pd.DataFrame, multiply: bool) -> Tuple:
    doc = nlp(text)
    emotional_class = defaultdict(list)
    result_dict = {}
    for token in doc:
        multiplier = 1
        if multiply:
            # part of speech
            pos = token.pos_
            try:
                multiplier = POS_MULTIPLIER[pos]
            except KeyError:
                continue

        sub_df = dataset.loc[dataset.token == token.text]
        if not sub_df.empty:
            for index, row in sub_df.iterrows():
                tmp = [row['token'], float(row['score'])*multiplier]
                if tmp not in emotional_class[row['label']]:
                    emotional_class[row['label']].append(tmp)
    # probabilities calc
    s = 0
    for k, _ in emotional_class.items():
        result_dict[k] = sum(map(lambda x: x[1], emotional_class[k])) / len(emotional_class[k])
        s += result_dict[k]
    for k, v in result_dict:
        result_dict[k] = v / s
    return emotional_class, result_dict


#  load text
filename = 'testText.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
text = clean_text(text)
result1 = eval_text(text, four_classes, multiply=False)[1]
result2 = eval_text(text, eight_classes, multiply=True)[1]
# removing 'positive' and 'negative' label
result2.pop('positive')
result2.pop('negative')


# normalization of each result
def scale_dict_result(dict_):
    values = dict_.values()
    min_ = min(values)
    max_ = max(values)
    return {key: ((v - min_) / (max_ - min_)) for (key, v) in dict_.items()}


result1 = scale_dict_result(dict_=result1)
result2 = scale_dict_result(dict_=result2)
a = 1
