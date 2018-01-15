import nltk
import gensim
import numpy as np
import pickle

x=[]
y=[]
m=1
model = gensim.models.Word2Vec.load('word2vec.bin');
import csv
with open('updatedconvo.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        if ((row[0]=='human') & (m==1)):
            x.append(row[1])
            m=0
        if ((row[0]=='robot') & (m==0)):
            y.append(row[1])
            m=1









tok_x = []
tok_y = []
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))

sentend = np.ones((300), dtype=np.float64)

vec_x = []
for sent in tok_x:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_x.append(sentvec)
#print (vec_x)

vec_y = []
for sent in tok_y:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_y.append(sentvec)

for tok_sent in vec_x:

    tok_sent[14:] = []

    tok_sent.append(sentend)

for tok_sent in vec_x:
    if len(tok_sent) < 15:
        for i in range(15 - len(tok_sent)):
            tok_sent.append(sentend)

for tok_sent in vec_y:
    tok_sent[14:] = []
    tok_sent.append(sentend)

for tok_sent in vec_y:
    if len(tok_sent) < 15:
        for i in range(15 - len(tok_sent)):
            tok_sent.append(sentend)

with open('conversation6.pickle', 'wb') as f:
    pickle.dump([vec_x, vec_y], f)
print (vec_x[0][0])
print (vec_y[0][0])