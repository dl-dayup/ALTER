# 数据集格式转换，将数据转换成包含id和content的 bm25 构建索引需要这种格式
import json
import os
import glob
# 将 LeCaRDv2 中的 train 和 test 数据集转换成预测 Topic 后加入到
# 将 LeCaRDv2 中所有 candidate 转换成 query fact jsonl文件，每个 case 一个文件，供 dense index 使用。用来测试非 finetune 直接拼接 topic 的效果topic_fact见predict代码
def json_to_jsonl(json_file_path, jsonl_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
            if isinstance(data, list):  # 如果JSON文件是一个列表
                for item in data:
                    id_value = item.get('pid', '')  # 如果没有id字段，则默认为空字符串
                    query_value = item.get('fact', '')  # 如果没有query字段，则默认为空字符串
                    new_item = { 'id': id_value, 'contents': query_value}
                    jsonl_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            elif isinstance(data, dict):  # 如果JSON文件是一个字典
                id_value = data.get('pid', '')  # 如果没有id字段，则默认为空字符串
                query_value = data.get('fact', '')  # 如果没有query字段，则默认为空字符串
                new_item = { 'id': id_value, 'contents': query_value}
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
output_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/jsonl_fact_candidate_55192/' #fact 再次确认下是只取fact还是fact3p
# output_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/jsonl_candidate_55192/' #仔细看下 qw是否是 query，
# output_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/jsonl_candidate_55192/' #默认是qw
process_directory(input_directory, output_directory)

'''
#将 LeCaRDv2 中所有candidate 转换成 jsonl文件，一行一个case 包含 topic fact 等用于进行finetune BiEncoder
def transform_json2txt(input_dir, output_file):
    for input_file in glob.glob(os.path.join(input_dir, '*.json')):
        # 执行转换
        with open(input_file, 'r', encoding='utf-8') as infile:
            # 读取整个 JSON 文件
            tem = json.load(infile)
            if tem['reason']:
                tt = tem['fact']  #tem['query']
                result = {'id': tem['id'], 'contents': tt}
                with open(output_file, 'a', encoding='utf-8') as outfile:
                    json.dump(result, output_file, ensure_ascii=False)
                    output_file.write("\n")  # 换行

input_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/candidate_55192/'
output_file = '/jupyterhub_data/test9/LDA_Topic/data/LeCaRDV2_can_3p_all.txt'
# output_directory = '/jupyterhub_data/test9/LeCaRDv2/candidate/new_format_candidate_55192/'
transform_json2txt(input_directory, output_file)



def jsonl_to_tsv_tokenizer(jsonl_file_path, tsv_file_path):
    tokenizer = AutoTokenizer.from_pretrained('/jupyterhub_data/test9/BART_BASE/')
    try:
        # 打开JSONL文件和TSV文件
        with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file, \
             open(tsv_file_path, 'w', encoding='utf-8', newline='') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            for line in jsonl_file:
                json_obj = json.loads(line)
                id_value = json_obj.get('id', '')  # 如果没有id字段，则默认为空字符串
                query_value = json_obj.get('query', '')  # 如果没有query字段，则默认为空字符串
                tokens = tokenizer.tokenize(query_value[:510])
                tokenized_text = " ".join(tokens)
                tsv_writer.writerow([int(id_value), tokenized_text])   
        print(f"转换完成，TSV文件已保存到 {tsv_file_path}")
    except FileNotFoundError:
        print(f"错误：文件 {jsonl_file_path} 未找到，请检查路径是否正确。")
    except json.JSONDecodeError:
        print(f"错误：JSON解析失败，请检查文件内容是否为有效的JSON格式。")
    except Exception as e:
        print(f"发生未知错误：{e}")

jsonl_file_path = '/jupyterhub_data/test9/LeCaRDv2/query/test_query.json'  # 替换为你的JSONL文件路径
tsv_file_path = '/jupyterhub_data/test9/LeCaRDv2/query/test_query512_tokenizer.tsv'     # 替换为你想要保存的TSV文件路径
jsonl_to_tsv_tokenizer(jsonl_file_path, tsv_file_path)
'''