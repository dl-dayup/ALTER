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


def json_to_jsonl(json_file_path, jsonl_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
            if isinstance(data, list):  # 如果JSON文件是一个列表
                for item in data:
                    id_value = item.get('pid', '')  # 如果没有id字段，则默认为空字符串
                    query_value = item.get('fact', '')  # 如果没有query字段，则默认为空字符串
                    txt = item['fact']# + item['reason'] + item['result']
                    topics = get_topic(txt)
                    t = " ".join([t for (t, s) in topics if s > 0.1])
                    tt = t + " " + txt
                    new_item = { 'id': id_value, 'contents': tt}
                    jsonl_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            elif isinstance(data, dict):  # 如果JSON文件是一个字典
                id_value = data.get('pid', '')  # 如果没有id字段，则默认为空字符串
                query_value = data.get('fact', '')  # 如果没有query字段，则默认为空字符串
                txt = data['fact'] #+ data['reason'] + data['result']
                topics = get_topic(txt)
                t = " ".join([t for (t, s) in topics if s > 0.1])
                tt = t + " " + txt
                new_item = {'id': id_value, 'contents': tt}
                jsonl_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            else:
                print("错误：JSON文件格式不支持。仅支持列表或字典格式。")
        print(f"转换完成，JSONL文件已保存到 {jsonl_file_path}")
    except FileNotFoundError:
        print(f"错误：文件 {json_file_path} 未找到，请检查路径是否正确。")
    except json.JSONDecodeError:
        print(f"错误：JSON解析失败，请检查文件内容是否为有效的JSON格式。")
    except Exception as e:
        print(f"发生未知错误：{e}")

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for input_file in glob.glob(os.path.join(input_dir, '*.json')):
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        json_to_jsonl(input_file, output_file)

input_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/candidate_55192/'
# output_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/jsonl_candidate_55192/' #默认是qw
output_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/jsonl_topic_fact_candidate_55192/' #fact+reason+result
# output_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/jsonl_candidate_55192/' #仔细看下 qw是否是 query，
# output_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/jsonl_candidate_55192/' #默认是qw
process_directory(input_directory, output_directory)

'''
def transform_json2txt(input_dir, output_file):
    for input_file in glob.glob(os.path.join(input_dir, '*.json')):
        # 执行转换
        with open(input_file, 'r', encoding='utf-8') as infile:
            # 读取整个 JSON 文件
            tem = json.load(infile)
            if tem['reason']:
                txt = tem['fact'] + tem['reason'] + tem['result']
                topics = get_topic(new_doc)
                t=" ".join([t for (t,s) in topics if s>0.1])
                tt = t + " " + txt
                result={'id':id,'contents':tt}
                with open(output_file, 'a', encoding='utf-8') as outfile:
                    json.dump(result, output_file, ensure_ascii=False)
                    output_file.write("\n")  # 换行

input_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/candidate_55192/'
output_file = '/jupyterhub_data/test9/LDA_Topic/data/LeCaRDV2_can_3p_all.txt'
# output_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/new_format_candidate_55192/'
transform_json2txt(input_directory, output_file)
'''