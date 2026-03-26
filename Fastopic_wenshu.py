import spacy
from fastopic import FASTopic
from topmost import Preprocess
from scipy import sparse
import jieba
import jieba.analyse
import jieba.posseg as pseg
import pandas as pd
import re
from collections import Counter, defaultdict
import concurrent.futures
from pathlib import Path

class MyPreprocess:
    def __init__(self):
        self.stopwords = set(['年月日时','年月日','二二','微信', '公众', '马克', '数据网','元','年','月','日','有限公司','公司','本案','原告','被告','审判','审判长','审判员','审理'])
        self.max_vocab_size =50000
        
    def fit_transform(self, token_lists):
        """处理大规模数据"""
        # 使用生成器方式统计词频，节省内存
        word_counter = Counter()
        total_tokens = 0
        for tokens in token_lists:
            word_counter.update(tokens)
            total_tokens += len(tokens)
        print(f"总token数量: {total_tokens}")
        print(f"原始词汇表大小: {len(word_counter)}")
        # 过滤和截取词汇表
        filtered_words = [(word, freq) for word, freq in word_counter.items() ]
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        if self.max_vocab_size:
            filtered_words = filtered_words[:self.max_vocab_size]
        # 构建词汇表
        self.vocab = {word: idx for idx, (word, freq) in enumerate(filtered_words)}
        self.id2word = {idx: word for idx, (word, freq) in enumerate(filtered_words)}
        vocab_list = [word for idx, (word, freq) in enumerate(filtered_words)]
        print(f"截取后词汇表大小: {len(self.vocab)}")
        for word, freq in filtered_words:
            print(word,freq)
        # 构建稀疏矩阵
        rows, cols, data = [], [], []
        
        for doc_idx, tokens in enumerate(token_lists):
            word_count = {}
            for token in tokens:
                if token in self.vocab:
                    word_count[token] = word_count.get(token, 0) + 1
            for token, count in word_count.items():
                rows.append(doc_idx)
                cols.append(self.vocab[token])
                data.append(count)
        matrix = sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(len(token_lists), len(self.vocab))
        )
        
        return matrix,vocab_list
    
    def preprocess(self, docs):

        processed_texts = []
        for text in docs:
            p_texts = []
            flag2 = True # 判断是否在停用表中或者是否包含停用表中的字词
            text = re.sub(r'[^\u4e00-\u9fa5]+', '', text)
            #words = jieba.lcut(text)
            #words = [word for word in words if len(word) > 1 and word not in self.stopwords]
            words = pseg.cut(text)
            filtered_words = []
            for word, flag in words:
                # 保留名词（n）、动词（v）等，过滤人名（nr）、地名（ns）、机构名（nt）
                if (flag not in ['nr', 'x']) and (word not in self.stopwords) and len(word) > 1:
                    for i in self.stopwords:
                        if word.__contains__(i):
                            flag2 = False
                    if flag2:
                        filtered_words.append(word)
            processed_texts.append(filtered_words)
        sparse_matrix,vocab = self.fit_transform(processed_texts)

        return {
            "train_bow": sparse_matrix, # sparse matrix
            "vocab": vocab # List[str]
        }
def read_file(file_path):
    """单个文件的读取函数"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]
def read_files_multithreaded(file_paths, max_workers=10):
    """多线程读取多个文件"""
    documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(read_file, fp): fp for fp in file_paths}
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                file_docs = future.result()
                documents.extend(file_docs)
            except Exception as e:
                print(f"Error reading file: {e}")
    
    return documents

# 执行预处理
#documents = []
# with open('../data/judgment2_2021.txt', 'r', encoding='utf-8') as f:
#     documents = [line.strip() for line in f.readlines()]
file_paths = ["../judgment2_2021.txt", "../judgment2_2019.txt", "../judgment2_2018.txt","../judgment2_2017.txt", "../judgment2_2016.txt", "../judgment2_2015.txt","../judgment2_2014.txt", "../judgment2_2013.txt"]  # 你的文件列表
documents = read_files_multithreaded(file_paths, max_workers=10)
preprocess = MyPreprocess()
print('---------1-------')
# 向量化
model = FASTopic(50, preprocess, doc_embed_model='/jupyterhub_data/test9/model/paraphrase-multilingual-MiniLM-L12-v2/', verbose=True, low_memory=True, low_memory_batch_size=36000) # 12000 15000MB 
print('---------2-------')
top_words, train_theta = model.fit_transform(documents,epochs=300 )
print('----------3------')
path = "./fastopic.zip"
model.save(path)
