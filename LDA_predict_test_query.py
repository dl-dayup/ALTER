import utils
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import numpy as np
import json,os,glob

documents = []
filepath='./data/LeCaRDV2_can_3p_all.txt'
with open(filepath, 'r', encoding='utf-8') as f:
    documents = [line.strip() for line in f.readlines()]
print('document  len',str(len(documents)))
processed_docs = utils.chinese_text_preprocess(documents)

dictionary = corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=3, no_above=0.5)  # 出现少于3次或超过50%文档的词
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
lda_model = LdaModel.load("./lda_model/lda_LeCaRDV2_3p_k30.gensim")


def get_topic(new_doc):
    processed_new_doc = utils.chinese_text_preprocess([new_doc])[0]
    new_bow = dictionary.doc2bow(processed_new_doc)
    # 获取主题分布
    topic_dist = lda_model.get_document_topics(new_bow)
    topic_dist = sorted(topic_dist, key = lambda x:x[0])
    # topic_show = lda_model.print_topics(-1, num_words=10)
    #
    # for topic_id, proc in topic_dist:
    #     print(f"Topic {topic_id}: {topic_show[topic_id], proc}")
    return topic_dist[10]

#new_doc = "违约方未按合同约定支付货款，需承担赔偿责任。"
#get_topic(new_doc)
import json
import csv
def jsonl_to_tsv(jsonl_file_path, tsv_file_path):
    try:
        # 打开JSONL文件和TSV文件
        with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file, \
                open(tsv_file_path, 'a', encoding='utf-8', newline='') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            for line in jsonl_file:
                json_obj = json.loads(line)
                id_value = json_obj.get('id', '')  # 如果没有id字段，则默认为空字符串
                query_value = json_obj.get('fact', '')  # 如果没有query字段，则默认为空字符串
                topics = get_topic(query_value)
                t = " ".join([t for (t, s) in topics if s > 0.1])
                # print(len(t.split(" ")))
                tt = t + " " + query_value
                tsv_writer.writerow([int(id_value), tt[:510]])
        print(f"转换完成，TSV文件已保存到 {tsv_file_path}")
    except FileNotFoundError:
        print(f"错误：文件 {jsonl_file_path} 未找到，请检查路径是否正确。")
    except json.JSONDecodeError:
        print(f"错误：JSON解析失败，请检查文件内容是否为有效的JSON格式。")
    except Exception as e:
        print(f"发生未知错误：{e}")

jsonl_file_path = '/jupyterhub_data/test9/LeCaRDv2/query/test_query.json'  # 替换为你的JSONL文件路径
tsv_file_path = '/jupyterhub_data/test9/LeCaRDv2/query/test_query_topicfact512.tsv'  # 替换为你想要保存的TSV文件路径
jsonl_to_tsv(jsonl_file_path, tsv_file_path)
