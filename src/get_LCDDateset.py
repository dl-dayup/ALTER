import json
import os,random

def load_queries(query_file):
    """
    加载query文件，返回一个字典，键为query_id，值为query内容。
    支持逐行解析JSON文件。
    """
    query_dict = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                query_dict[item['id']] = item['fact']
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                continue
    return query_dict

def load_relevance(relevance_file):
    """
    加载relevance文件，返回一个字典，键为query_id，值为一个列表，包含doc_id和label。
    """
    relevance_dict = {}
    with open(relevance_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            query_id, _, doc_id, label = parts
            query_id = int(query_id)
            label = int(label)
            if query_id not in relevance_dict:
                relevance_dict[query_id] = []
            relevance_dict[query_id].append((doc_id, label))
    return relevance_dict

def load_relevancePN(relevance_file):
    """
    加载relevance文件，返回一个字典，键为query_id，值为一个列表，包含doc_id和label。
    """
    relevance_dict = {}
    with open(relevance_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            query_id, _, doc_id, label = parts
            query_id = int(query_id)
            label = int(label)
            if query_id not in relevance_dict:
                relevance_dict[query_id] = {'pos': [], 'neg': []}
            if label in [2, 3]:  # 正样本
                relevance_dict[query_id]['pos'].append(doc_id)
            elif label in [0, 1]:  # 负样本
                relevance_dict[query_id]['neg'].append(doc_id)
    return relevance_dict

def load_doc_content(doc_id, candidate_dir):
    """
    根据doc_id加载doc内容，从candidate目录下的文件中读取fact、reason和result字段。
    """
    doc_file = os.path.join(candidate_dir, f"{doc_id}.json")
    if not os.path.exists(doc_file):
        return None
    with open(doc_file, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)
    fact = doc_data.get('fact', '')
    reason = doc_data.get('reason', '')
    result = doc_data.get('result', '')
    sum = len(reason)+ len(result)
    if sum<512:
        content = fact[:512-sum] + reason + " " + result
    else:
        content = reason + " " + result
    return content.strip()

def process_data1q1pMn(query_file, relevance_file, candidate_dir, output_file, num_neg_samples=15):
    """
    处理数据并保存为JSON格式。
    """
    queries = load_queries(query_file)
    relevance = load_relevancePN(relevance_file)
    # print('queries len:',len(queries))
    # print('relevance len:',len(relevance),relevance.keys())
    processed_data = []
    for query_id, query_content in queries.items():

        if query_id not in relevance.keys():
            continue
        pos_docs = relevance[query_id]['pos']
        neg_docs = relevance[query_id]['neg']

        # 如果负样本不足15个，从candidate目录中随机选择
        if len(neg_docs) < num_neg_samples:
            M = len(pos_docs)
            all_docs = [f.split('.')[0] for f in os.listdir(candidate_dir) if f.endswith('.json')]
            additional_neg_docs = [doc for doc in all_docs if doc not in pos_docs and doc not in neg_docs]
            neg_docs.extend(random.sample(additional_neg_docs, min(M * num_neg_samples - len(neg_docs), len(additional_neg_docs))))

        # 生成数据
        for pos_doc in pos_docs:
            pos_doc_content = load_doc_content(pos_doc, candidate_dir)
            if pos_doc_content is None:
                continue
            neg_doc_contents = []
            for neg_doc in random.sample(neg_docs, num_neg_samples):
                neg_doc_content = load_doc_content(neg_doc, candidate_dir)
                if neg_doc_content is not None:
                    neg_doc_contents.append(neg_doc_content)
                if len(neg_doc_contents) == num_neg_samples:
                    break

            if len(neg_doc_contents) == num_neg_samples:
                processed_data.append({
                    'query': query_content,
                    'pos_doc': pos_doc_content,
                    'neg_docs': neg_doc_contents
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f"Processed data saved to {output_file}",len(processed_data))

def process_data1qMpMn(query_file, relevance_file, candidate_dir, output_file):
    """
    处理数据并保存为JSON格式。
    """
    queries = load_queries(query_file)
    relevance = load_relevance(relevance_file)

    processed_data = []

    for query_id, query_content in queries.items():
        if query_id not in relevance:
            continue
        pos_docs = []
        neg_docs = []

        for doc_id, label in relevance[query_id]:
            doc_content = load_doc_content(doc_id, candidate_dir)
            if doc_content is None:
                continue
            if label in [2, 3]:  # 正样本
                pos_docs.append(doc_content)
            elif label in [0, 1]:  # 负样本
                neg_docs.append(doc_content)

        if pos_docs and neg_docs:  # 确保有正样本和负样本
            processed_data.append({
                'query': query_content,
                'pos_docs': pos_docs,
                'neg_docs': neg_docs
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f"Processed data saved to {output_file}",len(processed_data))

# 示例调用
if __name__ == "__main__":
    query_file = "/jupyterhub_data/test9/LeCaRDv2/query/train_query.json"
    relevance_file = "/jupyterhub_data/test9/LeCaRDv2/label/relevence.trec"
    candidate_dir = "/jupyterhub_data/test9/LeCaRDv2/candidate/candidate_55192/"
    output_file = "../data/LCDV2_train_1q1pMn.json"
    process_data1q1pMn(query_file, relevance_file, candidate_dir, output_file)