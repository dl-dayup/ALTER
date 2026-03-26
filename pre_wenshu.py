import os,csv,sys,json
all=right=filenum=0
output_file_path='./output.json'
#inputdir='./1985年裁判文书数据_马克数据网'
inputdir='./all'
#inputdir='./1999年裁判文书数据_马克数据网'
def split_legal_document(document):
    # 定义分割关键词
    keywords = ["经审理查明", "本院认为", "判决如下"]
    parts = []
    start = 0
    for keyword in keywords:
        # 查找关键词的位置
        index = document.find(keyword, start)
        if index != -1:
            # 如果找到关键词，提取前面的内容并添加到结果列表中
            if len(parts) == 0:
                parts.append(document[:index])  # 第一部分
            else:
                parts.append(document[start:index])  # 中间部分
            start = index  # 更新起始位置
        # else:
        #     # 如果未找到关键词，将剩余内容添加到结果列表中
        #     # parts.append(document[start:])
        #     break

    if len(parts) < 3:
        parts.append(document[start:])

    return parts
with open(output_file_path, "w", encoding="utf-8") as output_file:
    for filename in os.listdir(inputdir):
        filenum+=1
        csv_file_path = os.path.join(inputdir, filename)
        if os.path.isfile(csv_file_path):  # 确保是文件
            try:
                # 读取文件内容
                with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
                    # 创建 CSV 读取器对象
                    csv_reader = csv.reader(file)
                    headers = next(csv_reader, None)
                    # 逐行读取数据
                    for index, row in enumerate(csv_reader):
                        all+=1
                        # row 是一个列表，包含当前行的每个单元格的数据
                        parts = split_legal_document(row[-1:][0])
                        if len(parts)>1:
                            right+=1
                            category=row[12]
                            articles=row[13]
                            result = {
                            "file_name": filename+str(index),
                            "category": category,
                            "articles": articles,
                            "fact": parts[0].replace(' ','').replace('\t',''),
                            "interpretation": parts[1].replace(' ','').replace('\t','') if len(parts) > 1 else "",
                            "judgment": parts[2].replace(' ','').replace('\t','') if len(parts) > 2 else ""
                            }
    
                        # 将 JSON 对象写入文件，每行一个 JSON{"file_name": 1, "fact": "经审理查明：...", "interpretation": "本院认为...", "meta": {"relevant_articles": [313, 67], "accusation": ["拒不执行判决、裁定罪"]}, "articles": "第六十七条,第三百一十三条", "judgment": "判决如下：被告人刘孝娜犯拒不执行判决、裁定罪，判处有期徒刑一年零六个月。"}
    
                            json.dump(result, output_file, ensure_ascii=False)
                            output_file.write("\n")  # 换行
    
                print(f"处理完成：{filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错：{e}")
                
print(filenum, all, right)   
