import spacy 
nlp = spacy.load('en_core_web_sm')

import nltk
import csv
import sys
import pprint
from collections import defaultdict
#load text
filename = 'testText.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

#split into words by white space
words = text.split()
#convert to lower case
words = [word.lower() for word in words]

#remove punctuation from each word
import string
table = str.maketrans('','',string.punctuation)
stripped = [w.translate(table) for w in words]

# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]

# filter out stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
words = [w for w in words if not w in stop_words]
# print(words[:70])

import pandas as pd 
from pandas import DataFrame
data = pd.read_csv('Emotion-Affect-Intencity.csv',sep=";")
data.dropna(inplace= True)
data = data.reset_index(drop=True)
#data.set_index("term", inplace = True) 
#data_dict = data.to_dict()
#data_dict

print(words)

emotional_class = defaultdict(list)
for word in words:
	temp = data.loc[data['term'] == word]
	if not temp.empty:
		list_val = temp.values.tolist()[0]
		emotional_class[list_val[2]].append((list_val[0], list_val[1]))

# probabilities calc
result_dict = {}
for k, _ in emotional_class.items():
	result_dict[k] =  sum(map(lambda x: x[1], emotional_class[k])) / len(emotional_class[k])


pprint.pprint(result_dict)

