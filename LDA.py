import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import spacy 

# Define function for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return 

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    #nlp = spacy.load('en', disable=['parser', 'ner'])
    nlp = spacy.load('en_core_web_sm')
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
    
def sent_to_words(sentences):
  for sentence in sentences:
    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))            #deacc=True removes punctuations

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
df=pd.read_json('/jupyterhub_data/test9/sys_setup/newsgroups.json')

# Convert to list 
data = df.content.values.tolist()  
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]  
data = [re.sub('\s+', ' ', sent) for sent in data]  
data = [re.sub("\'", "", sent) for sent in data]  
data_words = list(sent_to_words(data))
# Remove Stop Words


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)
data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
id2word = corpora.Dictionary(data_lemmatized)  
texts = data_lemmatized  
corpus = [id2word.doc2bow(text) for text in texts]  

lda_model = LdaModel(corpus=corpus,
    id2word=id2word,
    num_topics=20, 
    alpha='asymmetric',      # 文档主题稀疏化
    eta=0.01,               # 主题词集中化（突出专业术语）
    passes=15,              # 增加迭代次数
    iterations=100,         # 单文档充分训练
    chunksize=5000,         # 大数据集分块
    random_state=42        # 可复现性
    )
lda_model.save("lda_model.gensim")  

# 导出主题-词分布（CSV）
with open("topic_words.csv", "w") as f:
    for topic in lda_model.print_topics():
        f.write(f"{topic[0]},{topic[1]}\n")
