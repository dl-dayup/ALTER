import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import random,json

# 设置随机种子以保证结果可复现
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 定义数据集类
class LCDDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data[idx]['query']
        pos_doc = self.data[idx]['pos_doc']
        neg_docs = self.data[idx]['neg_docs']
        # 编码query
        query_encoding = self.tokenizer(query, return_tensors='pt', max_length=self.max_len, truncation=True, padding='max_length')
        query_input_ids = query_encoding['input_ids'].flatten()
        query_attention_mask = query_encoding['attention_mask'].flatten()

        # 编码正例doc
        pos_doc_encoding = self.tokenizer(pos_doc, return_tensors='pt', max_length=self.max_len, truncation=True,
                                          padding='max_length')
        pos_doc_input_ids = pos_doc_encoding['input_ids'].flatten()
        pos_doc_attention_mask = pos_doc_encoding['attention_mask'].flatten()

        # 编码负例doc
        #only one
        '''
        doc_encoding = self.tokenizer(neg_docs[0], return_tensors='pt', max_length=self.max_len, truncation=True,
                                          padding='max_length')
        neg_doc_input_ids = doc_encoding['input_ids'].flatten()
        neg_doc_attention_masks = doc_encoding['attention_mask'].flatten()
        '''

        neg_doc_input_ids = []
        neg_doc_attention_masks = []
        for doc in neg_docs:
            doc_encoding = self.tokenizer(doc, return_tensors='pt', max_length=self.max_len, truncation=True,
                                          padding='max_length')
            neg_doc_input_ids.append(doc_encoding['input_ids'].flatten())
            neg_doc_attention_masks.append(doc_encoding['attention_mask'].flatten())
        
        # 转换为Tensor
        neg_doc_input_ids = torch.stack(neg_doc_input_ids)
        neg_doc_attention_masks = torch.stack(neg_doc_attention_masks)
        
        return {
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'pos_doc_input_ids': pos_doc_input_ids,
            'pos_doc_attention_mask': pos_doc_attention_mask,
            'neg_doc_input_ids': neg_doc_input_ids,
            'neg_doc_attention_mask': neg_doc_attention_masks
        }

# 定义双塔模型 参数不共享
class DualEncoderModel(nn.Module):
    def __init__(self, model_name='/jupyterhub_data/test9/model/Lawformer'):
        super(DualEncoderModel, self).__init__()
        # Query Encoder
        self.query_encoder = BertModel.from_pretrained(model_name)
        self.query_dropout = nn.Dropout(0.1)
        self.query_fc = nn.Sequential(
            nn.Linear(self.query_encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128))

        # Document Encoder
        self.doc_encoder = BertModel.from_pretrained(model_name)
        self.doc_dropout = nn.Dropout(0.1)
        self.doc_fc = nn.Sequential(
            nn.Linear(self.doc_encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128))

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        # Query Encoding
        query_outputs = self.query_encoder(input_ids=query_input_ids, attention_mask=query_attention_mask)
        query_pooled_output = query_outputs.pooler_output
        query_pooled_output = self.query_dropout(query_pooled_output)
        query_embedding = self.query_fc(query_pooled_output)

        # Document Encoding
        doc_outputs = self.doc_encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask)
        doc_pooled_output = doc_outputs.pooler_output
        doc_pooled_output = self.doc_dropout(doc_pooled_output)
        doc_embedding = self.doc_fc(doc_pooled_output)

        return query_embedding, doc_embedding

# 定义InfoNCE损失
def info_nce_loss(query_embeddings, pos_doc_embeddings, neg_doc_embeddings, batch_size,num_docs,i, temperature=0.5):
    """
    计算InfoNCE损失
    """
    # 计算query和正样本doc的相似度
    pos_scores = torch.sum(query_embeddings * pos_doc_embeddings, dim=1, keepdim=True)  # [batch_size, 1]
    query_embeddings = query_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
    neg_doc_embeddings = neg_doc_embeddings.view(batch_size, num_docs, 128)

    neg_scores = torch.matmul(query_embeddings, neg_doc_embeddings.transpose(-1, -2)).squeeze(1)  # [batch_size, num_negatives]
    # 合并正样本和负样本的相似度
    scores = torch.cat([pos_scores, neg_scores], dim=1)  # [batch_size, 1 + num_negatives]
    # 计算InfoNCE损失
    labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)  # 正样本的索引为0
    crit = torch.nn.CrossEntropyLoss()
    loss = crit(scores / temperature, labels)
    if i % 100 == 0:
        print('pos_scores', pos_scores)
        print('neg_scores', neg_scores)
        print('labels', i, labels,loss)
    return loss


# 定义训练函数
def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    i = 990
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        i+=1
        # 获取query和doc的表示
        # 获取query和doc的表示
        query_embeddings, pos_doc_embeddings = model(batch['query_input_ids'], batch['query_attention_mask'],
                                                     batch['pos_doc_input_ids'], batch['pos_doc_attention_mask'])
        batch_size, num_docs, seq_length = batch['neg_doc_input_ids'].shape
        neg_doc_embeddings = []
        for i in range(batch['neg_doc_input_ids'].shape[1]):
            neg_doc_embeddings.append(
                model(batch['query_input_ids'], batch['query_attention_mask'], batch['neg_doc_input_ids'][:, i, :],
                      batch['neg_doc_attention_mask'][:, i, :])[1])
        neg_doc_embeddings = torch.stack(neg_doc_embeddings, dim=1)
        # 计算InfoNCE损失
        loss = info_nce_loss(query_embeddings, pos_doc_embeddings, neg_doc_embeddings,batch_size,num_docs, i)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # # 监控梯度
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad norm: {param.grad.norm().item()}")
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        # print(f" Loss: {loss:.4f}")
    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")

# 主函数
if __name__ == "__main__":
    # 数据集
   # data=[]
    with open('../data/LCDV2_train_1q1pMn.json', 'r', encoding='utf-8') as f:
            data = json.load(f)  # 加载整个 JSON 数组
    print('data:',len(data))
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('/jupyterhub_data/test9/model/Lawformer')
    model = DualEncoderModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 创建数据集和数据加载器
    dataset = LCDDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # 定义优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=5e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 3)

    # 训练模型
    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        train_model(model, dataloader, optimizer, scheduler, device)

    # 保存模型
    # model.save_pretrained('../model/sharebert/')
    # tokenizer.save_pretrained('../model/sharebert/')
    torch.save(model.state_dict(), '../model/NSLawformer_InfoNCE/pytorch_model.bin')

    print("Model saved.")