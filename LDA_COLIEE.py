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
import spacy,os,argparse

def get_txt_files(directory):
    contents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    contents.append(f.read())
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")
    
    return contents

# Define function for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

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

data = get_txt_files('/jupyterhub_data/test9/COLIEE/2024/task1_train_files_2024')
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]  
data = [re.sub('\s+', ' ', sent) for sent in data]  
data = [re.sub("\'", "", sent) for sent in data]  
data_words = list(sent_to_words(data))

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
id2word = corpora.Dictionary(data_lemmatized)  
texts = data_lemmatized  
corpus = [id2word.doc2bow(text) for text in texts]  

lda_model = LdaModel(corpus=corpus,
    id2word=id2word,
    num_topics=10, 
    alpha='auto',      # 文档主题稀疏化
    eta='auto',               # 主题词集中化（突出专业术语）
    passes=15,              # 增加迭代次数
   # iterations=100,         # 单文档充分训练
   # chunksize=5000,         # 大数据集分块
    random_state=42        # 可复现性
    )
lda_model.save("lda_model.gensim")  

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# 生成可视化
vis_data = gensimvis.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis_data, 'lda_COLIEE_auto10_vis.html')  # 保存为HTML
# 导出主题-词分布（CSV）
with open("topic_words_2024train_eta.csv", "w") as f:
    for topic in lda_model.print_topics():
        f.write(f"{topic[0]},{topic[1]}\n")
