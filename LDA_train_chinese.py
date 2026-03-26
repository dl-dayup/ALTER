import utils
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

documents = []
filepath='./data/LeCaRDV2_can_3p_all.txt'
with open(filepath, 'r', encoding='utf-8') as f:
    documents = [line.strip() for line in f.readlines()]
print('document  len',str(len(documents)))

# 执行预处理
processed_docs = utils.chinese_text_preprocess(documents)

# 2. 构建词袋模型
dictionary = corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=3, no_above=0.5)  # 出现少于3次或超过50%文档的词
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# 3. 训练LDA模型
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=30,               # 主题数（根据业务调整）
    alpha='auto',              # 文档-主题分布稀疏性
    eta='auto',                # 主题-词分布稀疏性
    passes=20,                 # 迭代次数
    random_state=42            # 随机种子
)
lda_model.save("./lda_model/lda_LeCaRDV2_3p_k30.gensim")

# 4. 评估模型 计算主题一致性（越高越好）
coherence_model = CoherenceModel(
    model=lda_model,
    texts=processed_docs,
    dictionary=dictionary,
    coherence='c_v'
)
coherence = coherence_model.get_coherence()
print(f"主题一致性: {coherence:.3f}")

# 5. 结果可视化
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, './vis/lda_LeCaRDV2_3p_k30.html')  # 保存为HTML

# 6. 查看主题关键词
for idx, topic in lda_model.print_topics(-1, num_words=10):
    print(f"Topic {idx}: {topic}")

