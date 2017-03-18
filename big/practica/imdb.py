import pyprind
import pandas as pd
import os

import sys

import codecs

pbar = pyprind.ProgBar(50000)
labels = {"pos", "neg"}
df = pd.DataFrame()

for s in ("test", "train"):
    for l in ("pos", "neg"):
        path = "/Users/ilyaperepelitsa/Downloads/aclImdb/%s/%s" % (s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r") as infile:
                infile = codecs.open(os.path.join(path, file), encoding='utf-8')
                # print(infile)
                txt = infile.read()
                # print(txt)
            df = df.append([[txt, l]], ignore_index = True)
            pbar.update()


df.columns = ["review", "sentiment"]
df.shape


import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv("/Users/ilyaperepelitsa/quant/MLpy/practica/movie_data.csv", index = False)


df = pd.read_csv("/Users/ilyaperepelitsa/quant/MLpy/practica/movie_data.csv")
df.head(3)





import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
# count = CountVectorizer(ngram_range = (2, 2))
docs = np.array(["The sun is shining",
                 "The weather is sweet",
                 "The sun is shining and the weather is sweet"])

bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())




from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision = 2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


df.loc[0, "review"][-50:]
import re
def preprocessor(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    return text


preprocessor(df.loc[0, "review"][-50:])
preprocessor("</a>This :) is :( a test :-)!")


df["review"] = df["review"].apply(preprocessor)




from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return[porter.stem(word) for word in text.split()]

tokenizer_porter("runners like running and thus they run")

from nltk.corpus import stopwords
stop = stopwords.words("english")
[w for w in tokenizer_porter("a runner likes running and runs a lot") if w not in stop]







###  FINALLY TRAINING SOME SHIT


X_train = df.loc[:25000, "review"].values
y_train = df.loc[:25000, "sentiment"].values
X_test = df.loc[25000:, "review"].values
y_test = df.loc[25000:, "sentiment"].values


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents = None, lowercase = False, preprocessor = None)
param_grid = [{"vect__ngram_range" : [(1, 1)],
               "vect__stop_words" : [stop, None],
               "vect_tokenizer" : [tokenizer, tokenizer_porter],
               "clf__penalty" : ["l1", "l2"],
               "clf__C" : [1.0, 10.0, 100.0]},
               ]
