import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import json
import numpy as np
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
from transformers import (
    HfArgumentParser,
    set_seed,
)


@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 30000
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    num_topics: int = 100  # 预定义话题数量
    max_seq_len: int = 512
    dropout: float = 0.1
    # 动态lambda参数
    alpha: float = 0.8  # 初始XMC权重
    beta: float = 0.1  # 最终XMC权重
    lambda_decay_epochs: int = 3000
    temperature: float = 0.07  # InfoNCE温度参数
    max_negs: int = 15

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    data_path: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

class TokenEmbedding(nn.Module):
    """词嵌入层"""

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embed(input_ids) + self.position_embed(positions)
        x = self.layer_norm(x)
        return self.dropout(x)


class CoAttentionBlock(nn.Module):
    """
    CoAttention模块：Q来自当前编码器，KV来自另一个编码器
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        # Q投影（来自自己）
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # KV投影（来自对方）
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, kv_source: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            query: [batch, seq_len_q, embed_dim] - 来自当前编码器
            kv_source: [batch, seq_len_kv, embed_dim] - 来自另一个编码器
            mask: [batch, seq_len_q, seq_len_kv] 或 None
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = kv_source.shape[1]

        # 计算Q（来自自己）
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算KV（来自对方）
        kv = self.kv_proj(kv_source)
        K, V = kv.chunk(2, dim=-1)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, heads, seq_q, seq_kv]

        if mask is not None:
            # 扩展mask到多头 [batch, 1, seq_q, seq_kv] -> [batch, heads, seq_q, seq_kv]
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [batch, heads, seq_q, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)

        output = self.out_proj(attn_output)
        output = self.layer_norm(query + output)  # 残差连接

        return output


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, embed_dim: int, ff_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.layer_norm(residual + x)


class TextEncoderLayer(nn.Module):
    """TextEncoder单层：使用CoAttention，KV来自TopicEncoder"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # 自注意力（可选，用于捕获文本内部依赖）
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(embed_dim)

        # CoAttention: Q来自Text, KV来自Topic
        self.co_attn = CoAttentionBlock(embed_dim, num_heads, dropout)

        # FFN
        self.ffn = FeedForward(embed_dim, dropout=dropout)

    def forward(self, text_features: torch.Tensor, topic_features: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None):
        """
        Args:
            text_features: [batch, text_seq, embed_dim]
            topic_features: [batch, topic_seq, embed_dim] - 来自TopicEncoder
            text_mask: [batch, text_seq] - padding mask (1表示有效，0表示padding)
        """
        # 1. 自注意力（文本内部）
        # PyTorch MHA使用key_padding_mask: True表示需要mask的位置
        key_padding_mask = None
        if text_mask is not None:
            # [batch, seq] -> [batch, seq], True表示padding位置需要被mask
            key_padding_mask = (text_mask == 0)

        residual = text_features
        text_features, _ = self.self_attn(
            text_features, text_features, text_features,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        text_features = self.self_attn_norm(residual + text_features)

        # 2. CoAttention: Q=text, KV=topic
        # 构建cross-attention mask: text序列中的每个位置可以关注所有topic位置
        batch_size, text_len = text_features.shape[:2]
        topic_len = topic_features.shape[1]
        # [batch, text_len, topic_len] - 全1，因为topic没有padding（所有样本共享固定话题集）
        co_mask = torch.ones(batch_size, text_len, topic_len, device=text_features.device)

        text_features = self.co_attn(text_features, topic_features, co_mask)

        # 3. FFN
        text_features = self.ffn(text_features)

        return text_features


class TopicEncoderLayer(nn.Module):
    """TopicEncoder单层：使用CoAttention，KV来自TextEncoder"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # CoAttention: Q来自Topic, KV来自Text
        self.co_attn = CoAttentionBlock(embed_dim, num_heads, dropout)

        # FFN
        self.ffn = FeedForward(embed_dim, dropout=dropout)

    def forward(self, topic_features: torch.Tensor, text_features: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None):
        """
        Args:
            topic_features: [batch, num_topics, embed_dim]
            text_features: [batch, text_seq, embed_dim] - 来自TextEncoder
            text_mask: [batch, text_seq] - 用于mask掉padding位置
        """
        batch_size, num_topics = topic_features.shape[:2]
        text_len = text_features.shape[1]

        # 构建cross-attention mask: topic可以关注所有非padding的text位置
        if text_mask is not None:
            # text_mask: [batch, text_len] -> [batch, num_topics, text_len]
            co_mask = text_mask.unsqueeze(1).expand(-1, num_topics, -1)
        else:
            co_mask = torch.ones(batch_size, num_topics, text_len, device=topic_features.device)

        # CoAttention: Q=topic, KV=text
        topic_features = self.co_attn(topic_features, text_features, co_mask)

        # FFN
        topic_features = self.ffn(topic_features)

        return topic_features


class TextEncoder(nn.Module):
    """文本编码器 - 用于检索任务"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocab_size, config.embed_dim,
                                        config.max_seq_len, config.dropout)

        self.layers = nn.ModuleList([
            TextEncoderLayer(config.embed_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])

        # 输出投影（用于检索的dense向量）
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, input_ids: torch.Tensor, topic_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            topic_features: [batch, num_topics, embed_dim] - 来自TopicEncoder
            attention_mask: [batch, seq_len] - 1表示有效token，0表示padding
        Returns:
            text_embedding: [batch, embed_dim] - 用于检索的句子向量
        """
        # 词嵌入
        x = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

        # 多层CoAttention
        for layer in self.layers:
            x = layer(x, topic_features, attention_mask)

        # Mean pooling over valid tokens
        if attention_mask is not None:
            # [batch, seq_len, 1]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (x * mask_expanded).sum(dim=1)  # [batch, embed_dim]
            count = mask_expanded.sum(dim=1).clamp(min=1)
            x = sum_embeddings / count
        else:
            x = x.mean(dim=1)

        # 最终投影和归一化（用于检索）
        x = self.output_proj(x)
        x = F.normalize(x, p=2, dim=-1)

        return x


class TopicEncoder(nn.Module):
    """话题编码器 - 用于XMC分类任务，同时提供话题特征给TextEncoder"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 可学习的全局话题嵌入 [num_topics, embed_dim]
        self.topic_embeddings = nn.Embedding(config.num_topics, config.embed_dim)

        # TopicEncoder层
        self.layers = nn.ModuleList([
            TopicEncoderLayer(config.embed_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])

        # XMC分类头：预测每个话题的概率
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, 1)  # 输出logit
        )

        # 初始化话题嵌入
        nn.init.xavier_uniform_(self.topic_embeddings.weight)

    def get_topic_features(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """获取全局话题特征 [batch_size, num_topics, embed_dim]"""
        topic_ids = torch.arange(self.config.num_topics, device=device)
        topic_emb = self.topic_embeddings(topic_ids)  # [num_topics, embed_dim]
        # 扩展到batch
        topic_emb = topic_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_topics, embed_dim]
        return topic_emb

    def forward(self, text_features: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_features: [batch, text_seq, embed_dim] - 来自TextEncoder中间层或输入嵌入
            text_mask: [batch, text_seq]
        Returns:
            topic_features: [batch, num_topics, embed_dim] - 用于TextEncoder的CoAttention
            topic_logits: [batch, num_topics] - XMC分类logits
        """
        batch_size = text_features.shape[0]
        device = text_features.device

        # 获取初始话题嵌入
        topic_features = self.get_topic_features(batch_size, device)

        # 通过CoAttention层与文本交互
        for layer in self.layers:
            topic_features = layer(topic_features, text_features, text_mask)

        # XMC分类：每个话题一个logit
        topic_logits = self.classifier(topic_features).squeeze(-1)  # [batch, num_topics]

        return topic_features, topic_logits


class JointModel(nn.Module):
    """
    联合训练模型：TextEncoder + TopicEncoder + CoAttention
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # TextEncoder用于检索
        self.text_encoder = TextEncoder(config)
        # TopicEncoder用于XMC
        self.topic_encoder = TopicEncoder(config)

    def forward_text_through_encoder(self, input_ids: torch.Tensor, topic_features: torch.Tensor,
                                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """单独前向TextEncoder（用于检索）"""
        return self.text_encoder(input_ids, topic_features, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        完整前向传播，用于XMC任务（需要Text和Topic交互）
        Returns:
            text_emb: [batch, embed_dim] - 用于检索的文本向量
            topic_logits: [batch, num_topics] - 用于XMC的话题分布logits
        """
        # 1. 获取文本初始嵌入（用于TopicEncoder的输入）
        text_init = self.shared_embedding(input_ids)  # [batch, seq_len, embed_dim]
        # 2. TopicEncoder：使用text_init作为KV，输出topic_features和topic_logits
        topic_features, topic_logits = self.topic_encoder(text_init, attention_mask)
        # 3. TextEncoder：使用topic_features作为KV，输出text_embedding
        text_emb = self.text_encoder(input_ids, topic_features, attention_mask)

        return text_emb, topic_logits


# ==================== Loss Functions ====================

class InfoNCELoss(nn.Module):
    """InfoNCE Loss for Retrieval"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb: torch.Tensor, pos_emb: torch.Tensor,
                neg_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: [batch, embed_dim]
            pos_emb: [batch, embed_dim] - 正例
            neg_emb: [batch, num_negs, embed_dim] - 负例
        """
        batch_size = query_emb.shape[0]

        # 正例相似度 [batch]
        pos_sim = torch.sum(query_emb * pos_emb, dim=-1) / self.temperature

        # 负例相似度 [batch, num_negs]
        neg_sim = torch.sum(query_emb.unsqueeze(1) * neg_emb, dim=-1) / self.temperature

        # 合并 [batch, 1 + num_negs]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        # 标签：正例永远是第0个
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class XMCBinaryKLLoss(nn.Module):
    """XMC多标签分类的KL散度损失"""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_topics] - 模型输出的logits
            targets: [batch, num_topics] - 目标话题分布（已归一化概率和为1或多标签概率）
            mask: [batch] - 哪些样本有效
        """
        # 对logits做softmax得到预测分布
        log_probs = F.log_softmax(logits, dim=-1)

        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (logP - logQ))
        # 这里targets是P，log_probs是logQ
        kl = targets * (torch.log(targets + 1e-10) - log_probs)
        loss = kl.sum(dim=-1)  # [batch]

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss


class JointLoss(nn.Module):
    """联合损失：检索 + XMC"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.retrieval_loss = InfoNCELoss(config.temperature)
        self.xmc_loss = XMCBinaryKLLoss()
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_lambda(self) -> float:
        """计算当前epoch的动态lambda值"""
        alpha = self.config.alpha
        beta = self.config.beta
        progress = min(1.0, self.current_epoch / self.config.lambda_decay_epochs)
        lambda_val = alpha - (alpha - beta) * progress
        return lambda_val

    def forward(self, query_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor,
                all_text_logits: torch.Tensor, all_text_targets: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            query_emb: [batch, embed_dim]
            pos_emb: [batch, embed_dim]
            neg_emb: [batch, num_negs, embed_dim]
            all_text_logits: [total_texts, num_topics] - query+pos+negs的XMC logits
            all_text_targets: [total_texts, num_topics] - 对应的话题分布
            valid_mask: [total_texts] - 哪些文本有效（排除padding）
        """
        # 检索损失
        loss_retrieval = self.retrieval_loss(query_emb, pos_emb, neg_emb)

        # XMC损失
        loss_xmc = self.xmc_loss(all_text_logits, all_text_targets, valid_mask)

        # 动态权重
        lambda_val = self.get_lambda()

        # 总损失
        total_loss = lambda_val * loss_xmc + (1 - lambda_val) * loss_retrieval

        loss_dict = {
            'total_loss': total_loss.item(),
            'loss_retrieval': loss_retrieval.item(),
            'loss_xmc': loss_xmc.item(),
            'lambda': lambda_val
        }

        return total_loss, loss_dict


# ==================== Dataset ====================

class RetrievalXMCDataset(Dataset):
    """
    检索+XMC联合训练数据集
    数据格式：
    {
        "query_text": str,
        "query_topic": [[weight1, topic_id1], [weight2, topic_id2], ...],
        "pos_doc_text": str,
        "pos_doc_topic": [...],
        "neg_doc_texts": [str, str, ...],
        "neg_doc_topics": [[...], [...], ...]
    }
    """

    def __init__(self, data_file: str, tokenizer, config: ModelConfig, max_negs: int = 5):
        self.config = config
        self.tokenizer = tokenizer
        self.max_negs = max_negs

        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def parse_topic(self, topic_list: List[List[float]], num_topics: int) -> torch.Tensor:
        """将话题列表转换为概率分布向量"""
        topic_dist = torch.zeros(num_topics)
        for weight, topic_id in topic_list:
            topic_dist[int(topic_id)] = float(weight)
        # 归一化（如果和不为0）
        if topic_dist.sum() > 0:
            topic_dist = topic_dist / topic_dist.sum()
        return topic_dist

    def tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:

        tokens = text.split()[:self.config.max_seq_len]
        token_ids = [hash(token) % self.config.vocab_size for token in tokens]

        # padding
        seq_len = len(token_ids)
        attention_mask = [1] * seq_len + [0] * (self.config.max_seq_len - seq_len)
        token_ids = token_ids + [0] * (self.config.max_seq_len - seq_len)

        return torch.tensor(token_ids[:self.config.max_seq_len]), \
            torch.tensor(attention_mask[:self.config.max_seq_len])

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # 限制负例数量
        neg_texts = item['neg_doc_texts'][:self.max_negs]
        neg_topics = item['neg_doc_topics'][:self.max_negs]

        # 如果负例不够，随机选择
        while len(neg_texts) < self.max_negs:
            neg_texts.append(neg_texts[-1] if neg_texts else "")
            neg_topics.append(neg_topics[-1] if neg_topics else [])

        # Tokenize所有文本
        query_ids, query_mask = self.tokenize(item['query_text'])
        pos_ids, pos_mask = self.tokenize(item['pos_doc_text'])

        neg_ids_list = []
        neg_mask_list = []
        for neg_text in neg_texts:
            neg_ids, neg_mask = self.tokenize(neg_text)
            neg_ids_list.append(neg_ids)
            neg_mask_list.append(neg_mask)

        # 解析话题分布
        query_topic = self.parse_topic(item['query_topic'], self.config.num_topics)
        pos_topic = self.parse_topic(item['pos_doc_topic'], self.config.num_topics)

        neg_topics_list = []
        for neg_topic in neg_topics:
            neg_topics_list.append(self.parse_topic(neg_topic, self.config.num_topics))

        return {
            'query_ids': query_ids,
            'query_mask': query_mask,
            'query_topic': query_topic,
            'pos_ids': pos_ids,
            'pos_mask': pos_mask,
            'pos_topic': pos_topic,
            'neg_ids': torch.stack(neg_ids_list),  # [max_negs, seq_len]
            'neg_mask': torch.stack(neg_mask_list),  # [max_negs, seq_len]
            'neg_topics': torch.stack(neg_topics_list),  # [max_negs, num_topics]
            'num_real_negs': len(item['neg_doc_texts'][:self.max_negs])  # 真实负例数
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Batch collate function"""
    return {
        'query_ids': torch.stack([b['query_ids'] for b in batch]),
        'query_mask': torch.stack([b['query_mask'] for b in batch]),
        'query_topics': torch.stack([b['query_topic'] for b in batch]),
        'pos_ids': torch.stack([b['pos_ids'] for b in batch]),
        'pos_mask': torch.stack([b['pos_mask'] for b in batch]),
        'pos_topics': torch.stack([b['pos_topic'] for b in batch]),
        'neg_ids': torch.stack([b['neg_ids'] for b in batch]),  # [batch, max_negs, seq_len]
        'neg_mask': torch.stack([b['neg_mask'] for b in batch]),  # [batch, max_negs, seq_len]
        'neg_topics': torch.stack([b['neg_topics'] for b in batch]),  # [batch, max_negs, num_topics]
        'num_real_negs': [b['num_real_negs'] for b in batch]
    }


# ==================== Training ====================

class Trainer:
    def __init__(self, model: JointModel, config: ModelConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = JointLoss(config)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.current_epoch = 0

    def train_epoch(self, dataloader: DataLoader) -> Dict:
        self.model.train()
        self.criterion.set_epoch(self.current_epoch)

        total_losses = {'total_loss': 0, 'loss_retrieval': 0, 'loss_xmc': 0}
        num_batches = 0

        for batch in dataloader:
            # 移动到设备
            query_ids = batch['query_ids'].to(self.device)
            query_mask = batch['query_mask'].to(self.device)
            query_topics = batch['query_topics'].to(self.device)

            pos_ids = batch['pos_ids'].to(self.device)
            pos_mask = batch['pos_mask'].to(self.device)
            pos_topics = batch['pos_topics'].to(self.device)

            neg_ids = batch['neg_ids'].to(self.device)  # [batch, num_negs, seq_len]
            neg_mask = batch['neg_mask'].to(self.device)
            neg_topics = batch['neg_topics'].to(self.device)  # [batch, num_negs, num_topics]

            batch_size, num_negs, seq_len = neg_ids.shape

            # ========== 前向传播 ==========

            # 1. Query编码
            query_emb, query_logits = self.model(query_ids, query_mask)

            # 2. Pos Doc编码
            pos_emb, pos_logits = self.model(pos_ids, pos_mask)

            # 3. Neg Docs编码 [batch*num_negs, ...]
            neg_ids_flat = neg_ids.view(-1, seq_len)
            neg_mask_flat = neg_mask.view(-1, seq_len)
            neg_emb_flat, neg_logits_flat = self.model(neg_ids_flat, neg_mask_flat)

            # 恢复形状 [batch, num_negs, embed_dim] 和 [batch, num_negs, num_topics]
            neg_emb = neg_emb_flat.view(batch_size, num_negs, -1)
            neg_logits = neg_logits_flat.view(batch_size, num_negs, self.config.num_topics)

            # ========== 准备XMC的输入 ==========
            # 合并所有文本的logits和targets: query + pos + all_negs
            # [batch, 1+1+num_negs, num_topics]
            all_logits = torch.cat([
                query_logits.unsqueeze(1),
                pos_logits.unsqueeze(1),
                neg_logits
            ], dim=1)

            all_targets = torch.cat([
                query_topics.unsqueeze(1),
                pos_topics.unsqueeze(1),
                neg_topics
            ], dim=1)

            # Flatten: [batch*(2+num_negs), num_topics]
            total_samples = batch_size * (2 + num_negs)
            all_logits_flat = all_logits.view(total_samples, self.config.num_topics)
            all_targets_flat = all_targets.view(total_samples, self.config.num_topics)

            # 构建valid mask（排除padding的负例）
            # 对于每个batch，前2+num_real_negs个是有效的
            valid_mask = torch.zeros(total_samples, device=self.device)
            for i, real_negs in enumerate(batch['num_real_negs']):
                valid_count = 2 + real_negs  # query + pos + real_negs
                start_idx = i * (2 + num_negs)
                valid_mask[start_idx:start_idx + valid_count] = 1

            # ========== 计算损失 ==========
            total_loss, loss_dict = self.criterion(
                query_emb, pos_emb, neg_emb,
                all_logits_flat, all_targets_flat, valid_mask
            )

            # ========== 反向传播 ==========
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 记录
            for k, v in loss_dict.items():
                total_losses[k] += v
            num_batches += 1

        # 平均
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_losses['epoch'] = self.current_epoch
        avg_losses['lambda'] = self.criterion.get_lambda()

        self.current_epoch += 1
        return avg_losses

    def save_checkpoint(self, path: str):
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']

def create_dummy_data(output_file: str, num_samples: int = 100, num_topics: int = 100):
    """创建示例训练数据"""
    import random

    with open(output_file, 'w') as f:
        for i in range(num_samples):
            # 随机生成1-3个话题
            num_query_topics = random.randint(1, 3)
            query_topic = []
            weights = [random.random() for _ in range(num_query_topics)]
            total = sum(weights)
            for j, w in enumerate(weights):
                topic_id = random.randint(0, num_topics - 1)
                query_topic.append([w / total, topic_id])

            num_pos_topics = random.randint(1, 3)
            pos_topic = []
            weights = [random.random() for _ in range(num_pos_topics)]
            total = sum(weights)
            for j, w in enumerate(weights):
                topic_id = random.randint(0, num_topics - 1)
                pos_topic.append([w / total, topic_id])

            # 3-5个负例
            num_negs = random.randint(3, 5)
            neg_texts = [f"negative document {k} for sample {i}" for k in range(num_negs)]
            neg_topics = []
            for _ in range(num_negs):
                num_t = random.randint(1, 3)
                weights = [random.random() for _ in range(num_t)]
                total = sum(weights)
                t_list = []
                for w in weights:
                    tid = random.randint(0, num_topics - 1)
                    t_list.append([w / total, tid])
                neg_topics.append(t_list)

            sample = {
                "query_text": f"query text for sample {i} about topic",
                "query_topic": query_topic,
                "pos_doc_text": f"positive document {i} relevant content",
                "pos_doc_topic": pos_topic,
                "neg_doc_texts": neg_texts,
                "neg_doc_topics": neg_topics
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():
    # 配置
    config = ModelConfig(
        vocab_size=10000,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        num_topics=50,
        max_seq_len=64,
        alpha=0.8,  # 初始XMC权重高
        beta=0.1,  # 后期检索权重高
        lambda_decay_epochs=2000,
        max_negs = 15
    )
    parser = HfArgumentParser(ModelArguments)
    model_args: ModelArguments= parser.parse_args_into_dataclasses()


    dataset = RetrievalXMCDataset(model_args.data_path, PreTrainedTokenizer, config, max_negs=config.max_negs)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 创建模型
    model = JointModel(config)

    # 查看参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, config, device)

    print("\nStarting training...")
    for epoch in range(config.lambda_decay_epochs):
        losses = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch}: total={losses['total_loss']:.4f}, "
              f"retrieval={losses['loss_retrieval']:.4f}, "
              f"xmc={losses['loss_xmc']:.4f}, "
              f"lambda={losses['lambda']:.4f}")

    print("\nTraining completed!")
    # 保存模型
    trainer.save_checkpoint(model_args.target_model_path)
    print("Model saved")


if __name__ == "__main__":

    main()
