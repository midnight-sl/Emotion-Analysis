import spacy
from typing import Tuple, Dict
from collections import defaultdict
import pandas as pd
import re

nlp = spacy.load('en_core_web_md')

PATH_TO_FOUR_CLASSES_DATA = './NRC-Sentiment-Emotion-Lexicons/' \
                            'NRC-Affect-Intensity-Lexicon/NRC-AffectIntensity-Lexicon.csv'

PATH_TO_EIGHT_CLASSES_DATA = './NRC-Sentiment-Emotion-Lexicons/' \
                             'NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.csv'

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


def eval_text(text: str, dataset: pd.DataFrame, multiplier=None, pos=None) -> Tuple:
    doc = nlp(text)
    emotional_class = defaultdict(list)
    result_dict = {}
    for token in doc:
        # part of speech
        pos = token.pos_
        sub_df = dataset.loc[dataset.token == token.text]
        if not sub_df.empty:
            for index, row in sub_df.iterrows():
                tmp = [row['token'], float(row['score'])]
                if tmp not in emotional_class[row['label']]:
                    emotional_class[row['label']].append(tmp)
    # probabilities calc
    for k, _ in emotional_class.items():
        result_dict[k] = sum(map(lambda x: x[1], emotional_class[k])) / len(emotional_class[k])

    return emotional_class, result_dict


#  load text
filename = 'testText.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
text = clean_text(text)
eval_text(text, four_classes)
a = 1
