import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from CoAttentionBlock import *

class TopicAwareRetrievalModel(nn.Module):
    """主题感知的法律类案检索模型"""

    def __init__(self, config):
        super(TopicAwareRetrievalModel, self).__init__()
        self.config = config
        # 文本编码器
        self.text_encoder = nn.Linear(config.text_input_dim, config.hidden_dim)
        # 主题编码器
        self.topic_encoder = nn.Linear(config.topic_input_dim, config.hidden_dim)
        # Co-attention Blocks
        self.co_attention_blocks = nn.ModuleList([
            CoAttentionTransformerLayer(
                hidden_size=config.hidden_dim,
                num_attention_heads=config.num_heads,
                intermediate_size=config.intermediate_size
            ) for _ in range(config.num_co_attention_layers)
        ])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_input, topic_input, text_mask=None, topic_mask=None):
        """
        text_input: [batch_size, text_len, text_input_dim]
        topic_input: [batch_size, topic_num, topic_input_dim]
        """
        # 需要兼容predict时的场景，只有text_input，没有topic_input时，只计算text2topic 不计算topic2text
        # 编码文本和主题
        text_repr = self.text_encoder(text_input)  # [batch, text_len, hidden]
        if topic_input:
            topic_repr = self.topic_encoder(topic_input)  # [batch, topic_num, hidden]

            # 通过Co-attention Blocks
            for layer in self.co_attention_layers:
                text_states, topic_states = layer(text_repr, topic_repr, attention_mask)
            final_text_repr, final_topic_repr = text_states, topic_states

        # 取[CLS]标记作为文本的最终表示（假设第一个token是[CLS]）
        text_cls_repr = final_text_repr[:, 0, :]  # [batch_size, hidden_dim]

        # 分类任务的相关表示
        classification_text_repr = self.classification_proj(text_cls_repr)
        classification_topic_repr = self.classification_proj(final_topic_repr)

        return {
            'text_representation': text_cls_repr,  # 用于检索任务
            'classification_text_repr': classification_text_repr,  # 用于分类任务的文本表示
            'classification_topic_repr': classification_topic_repr,  # 用于分类任务的主题表示
            'final_text_repr': final_text_repr,
            'final_topic_repr': final_topic_repr
        }


class TopicAwareRetrievalLoss(nn.Module):
    """主题感知检索模型的损失函数"""

    def __init__(self, config):
        super(TopicAwareRetrievalLoss, self).__init__()
        self.config = config
        self.retrieval_loss_fn = CrossEntropyLoss()
        self.classification_loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, model_outputs, labels):
        """
        model_outputs: 模型的输出字典
        labels: 包含检索标签和分类标签的字典
        """
        # 检索任务损失
        retrieval_loss = self._compute_retrieval_loss(model_outputs, labels)

        # 分类任务损失
        classification_loss = self._compute_classification_loss(model_outputs, labels)

        # 总损失
        total_loss = self.config.alpha * classification_loss + (1 - self.config.alpha) * retrieval_loss

        return {
            'total_loss': total_loss,
            'retrieval_loss': retrieval_loss,
            'classification_loss': classification_loss
        }

    def _compute_retrieval_loss(self, model_outputs, labels):
        """计算检索任务损失：一个query，一个正样本，多个负样本"""
        query_repr = model_outputs['text_representation']  # [batch_size, hidden_dim]
        pos_doc_repr = labels['positive_doc_representation']  # [batch_size, hidden_dim]
        neg_docs_repr = labels['negative_docs_representation']  # [batch_size, num_negatives, hidden_dim]

        batch_size, hidden_dim = query_repr.shape
        num_negatives = neg_docs_repr.shape[1]

        # 计算query与正样本的相似度
        pos_similarity = F.cosine_similarity(query_repr, pos_doc_repr, dim=-1)  # [batch_size]

        # 计算query与所有负样本的相似度
        query_expanded = query_repr.unsqueeze(1).expand(-1, num_negatives, -1)  # [batch, num_neg, hidden]
        neg_similarities = F.cosine_similarity(
            query_expanded.reshape(-1, hidden_dim),
            neg_docs_repr.reshape(-1, hidden_dim),
            dim=-1
        ).reshape(batch_size, num_negatives)  # [batch_size, num_negatives]

        # 组合所有相似度得分
        all_similarities = torch.cat([
            pos_similarity.unsqueeze(1),  # [batch_size, 1]
            neg_similarities  # [batch_size, num_negatives]
        ], dim=1)  # [batch_size, 1 + num_negatives]

        # 目标标签：正样本始终在位置0
        target = torch.zeros(batch_size, dtype=torch.long, device=query_repr.device)

        return self.retrieval_loss_fn(all_similarities, target)

    def _compute_classification_loss(self, model_outputs, labels):
        """计算分类任务损失：KL散度"""
        text_repr = model_outputs['classification_text_repr']  # [batch_size, hidden_dim]
        topic_repr = model_outputs['classification_topic_repr']  # [batch_size, topic_num, hidden_dim]
        true_topic_distribution = labels['topic_distribution']  # [batch_size, topic_num]

        batch_size, topic_num, hidden_dim = topic_repr.shape

        # 计算文本表示与所有主题表示的相似度
        text_repr_expanded = text_repr.unsqueeze(1).expand(-1, topic_num, -1)  # [batch, topic_num, hidden]
        similarities = F.cosine_similarity(
            text_repr_expanded.reshape(-1, hidden_dim),
            topic_repr.reshape(-1, hidden_dim),
            dim=-1
        ).reshape(batch_size, topic_num)  # [batch_size, topic_num]

        # 将相似度转换为预测的主题分布（使用softmax）
        pred_topic_distribution = F.log_softmax(similarities, dim=-1)

        # 真实主题分布（LDA输出）
        true_topic_distribution = F.softmax(true_topic_distribution, dim=-1)

        # 计算KL散度损失
        return self.classification_loss_fn(pred_topic_distribution, true_topic_distribution)

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
        query_encoding = self.tokenizer(query, return_tensors='pt', max_length=self.max_len, truncation=True,
                                        padding='max_length')
        query_input_ids = query_encoding['input_ids'].flatten()
        query_attention_mask = query_encoding['attention_mask'].flatten()

        # 编码正例doc
        pos_doc_encoding = self.tokenizer(pos_doc, return_tensors='pt', max_length=self.max_len, truncation=True,
                                          padding='max_length')
        pos_doc_input_ids = pos_doc_encoding['input_ids'].flatten()
        pos_doc_attention_mask = pos_doc_encoding['attention_mask'].flatten()

        # 编码负例doc
        # only one
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

# 配置类
class ModelConfig:
    def __init__(self):
        self.text_input_dim = 768  # 假设的文本输入维度
        self.topic_input_dim = 100  # 假设的主题输入维度（LDA主题数）
        self.hidden_dim = 768
        self.num_co_attention_layers = 3
        self.hidden_dropout_prob = 0.1
        self.initializer_range = 0.02
        self.alpha = 0.8  # 分类任务损失的权重

# 使用示例
def main():
    config = ModelConfig()
    model = TopicAwareRetrievalModel(config)
    loss_fn = TopicAwareRetrievalLoss(config)

    batch_size, text_len, topic_num = 4, 128, 50
    text_input = torch.randn(batch_size, text_len, config.text_input_dim)
    topic_input = torch.randn(batch_size, topic_num, config.topic_input_dim)
    outputs = model(text_input, topic_input)

    # 准备标签
    labels = {
        'positive_doc_representation': torch.randn(batch_size, config.hidden_dim),
        'negative_docs_representation': torch.randn(batch_size, 5, config.hidden_dim),  # 5个负样本
        'topic_distribution': torch.randn(batch_size, topic_num)  # LDA主题分布
    }

    # 计算损失
    losses = loss_fn(outputs, labels)

    print("总损失:", losses['total_loss'].item())
    print("检索损失:", losses['retrieval_loss'].item())
    print("分类损失:", losses['classification_loss'].item())


if __name__ == "__main__":
    main()