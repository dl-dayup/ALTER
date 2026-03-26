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
        pos_docs = self.data[idx]['pos_docs']
        neg_docs = self.data[idx]['neg_docs']
        # 编码query
        query_encoding = self.tokenizer(query, return_tensors='pt', max_length=self.max_len, truncation=True, padding='max_length')
        query_input_ids = query_encoding['input_ids'].flatten()
        query_attention_mask = query_encoding['attention_mask'].flatten()
        all_docs = pos_docs + neg_docs
        labels = [1.0] * len(pos_docs) + [0.0] * len(neg_docs)
        # 打乱顺序
        combined = list(zip(all_docs, labels))
        random.shuffle(combined)
        all_docs, labels = zip(*combined)

        # 编码所有文档
        doc_input_ids = []
        doc_attention_masks = []
        for doc in all_docs:
            doc_encoding = self.tokenizer(doc, return_tensors='pt', max_length=self.max_len, truncation=True, padding='max_length')
            doc_input_ids.append(doc_encoding['input_ids'].squeeze())
            doc_attention_masks.append(doc_encoding['attention_mask'].squeeze())
        # 转换为Tensor
        doc_input_ids = torch.stack(doc_input_ids)
        doc_attention_masks = torch.stack(doc_attention_masks)
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'doc_input_ids': doc_input_ids,
            'doc_attention_mask': doc_attention_masks,
            'labels': labels
        }

# 定义双塔模型
class DualEncoderModel(nn.Module):
    def __init__(self, model_name='/jupyterhub_data/test9/model/Lawformer'):
        super(DualEncoderModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        output = self.fc(pooled_output)
        return output

# 定义训练函数
def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        query_embeddings = model(batch['query_input_ids'], batch['query_attention_mask']) # Shape: [1,512,768]
        batch_size, num_docs, seq_length = batch['doc_input_ids'].shape #[1,30,512]
        doc_input_ids = batch['doc_input_ids'].view(batch_size * num_docs, seq_length) # Shape: [30,512,768]
        doc_attention_masks = batch['doc_attention_mask'].view(batch_size * num_docs, seq_length)
        doc_embeddings = model(doc_input_ids, doc_attention_masks)
        scores = torch.cosine_similarity(query_embeddings, doc_embeddings) # Shape: [1,30]
        # query_repr = query_embeddings.mean(dim=1)  # Shape: [1,512,768] -> [1, 768]
        # doc_repr = doc_embeddings.mean(dim=1)  # Shape: [30,512,768] -> [30, 768]
        #scores = torch.matmul(query_repr, doc_repr.T)  # Shape: [1,30]
        # print(scores)
        loss = torch.nn.functional.cross_entropy(scores.squeeze(0) , batch['labels'].squeeze(0) )
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")

# 主函数
if __name__ == "__main__":
    # 数据集
   # data=[]
    with open('../data/LCDV2_train.json', 'r', encoding='utf-8') as f:
            data = json.load(f)  # 加载整个 JSON 数组
    print('data:',len(data))
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('/jupyterhub_data/test9/model/Lawformer')
    model = DualEncoderModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 创建数据集和数据加载器
    dataset = LCDDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 定义优化器和学习率调度器
    optimizer = AdamW(model.fc.parameters(), lr=5e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 3)

    # 训练模型
    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        train_model(model, dataloader, optimizer, scheduler, device)

    # 保存模型
    # model.save_pretrained('../model/sharebert/')
    # tokenizer.save_pretrained('../model/sharebert/')
    torch.save(model.state_dict(), '../model/shareLawformer/pytorch_model.bin')

    print("Model saved.")