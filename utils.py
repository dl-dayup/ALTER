import jieba
import jieba.analyse
import jieba.posseg as pseg
import re
from collections import defaultdict

def chinese_text_preprocess(texts, stopwords_path='stopwords.txt'):

    if stopwords_path:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
    else:
        stopwords = set(['有限公司','公司','年月日','本案','原告','被告','审判','审判长','审判员','审理'])

    processed_texts = []
    for text in texts:
        text = re.sub(r'[^\u4e00-\u9fa5]+', '', text)
        #words = jieba.lcut(text)
        #words = [word for word in words if len(word) > 1 and word not in stopwords]
        words = pseg.cut(text)
        filtered_words = []
        for word, flag in words:
        # 保留名词（n）、动词（v）等，过滤人名（nr）、地名（ns）、机构名（nt）
            if (flag not in ['nr', 'ns', 'nt', 'x']) and (word not in stopwords) and len(word) > 1:
                filtered_words.append(word)
        processed_texts.append(filtered_words)
    return processed_texts