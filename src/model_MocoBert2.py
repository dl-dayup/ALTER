import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.nn.functional import normalize
from transformers.modeling_outputs import MaskedLMOutput

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, region1, region2):
        # region1: [batch_size, seq_len, hidden_dim]
        # region2: [batch_size, seq_len, hidden_dim]
        region2 = region2.to(region1.device)

        self.query = self.query.to(region1.device)
        self.key = self.key.to(region1.device)
        self.value = self.value.to(region1.device)

        q = self.query(region2)
        k = self.key(region1)
        v = self.value(region1)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights, v)  # 加权求和

        return output + region2  # 残差连接

class MoCoBERT(nn.Module):
    def __init__(self, bert_model="/jupyterhub_data/test9/model/bert-base-chinese", feature_dim=128, queue_size=65536, momentum=0.999):
        super().__init__()
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化双塔BERT（共享初始权重）
        self.fact_encoder = BertModel.from_pretrained(bert_model)
        self.reason_encoder = BertModel.from_pretrained(bert_model)

        # 投影头（将BERT输出映射到对比学习空间）
        self.fact_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        self.reason_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

        # 冻结key_encoder的梯度（仅动量更新）
        for param in self.reason_encoder.parameters():
            param.requires_grad = False
        for param in self.reason_proj.parameters():
            param.requires_grad = False

        # 动态队列（存储负样本）
        self.register_buffer("queue", torch.randn(feature_dim, queue_size).to(self.device))
        self.queue = normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long).to(self.device))

    @torch.no_grad()
    def momentum_update(self):
        """动量更新key_encoder的参数"""
        for q_param, k_param in zip(
            self.fact_encoder.parameters(), self.reason_encoder.parameters()
        ):
            k_param.data = k_param.data * self.momentum + q_param.data * (1.0 - self.momentum)
        for q_proj, k_proj in zip(
            self.fact_proj.parameters(), self.reason_proj.parameters()
        ):
            k_proj.data = k_proj.data * self.momentum + q_proj.data * (1.0 - self.momentum)

    @torch.no_grad()
    def enqueue_dequeue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # 确保队列可被整除

        # 替换队列中的旧数据
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B):
        # 编码区域1（query）
        q_outputs = self.fact_encoder(input_ids_A, attention_mask_A)
        q_cls = q_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        q_features = q_outputs.last_hidden_state  # 完整序列表示

        # 编码区域2（key）
        with torch.no_grad():
            self.momentum_update()
            k_outputs = self.reason_encoder(input_ids_B, attention_mask_B)
            k_cls = k_outputs.last_hidden_state[:, 0, :]
            k_features = k_outputs.last_hidden_state

        # 加入Cross-Attention交互
        cross_attn = CrossAttentionLayer()
        q_enhanced = cross_attn(q_features, k_features)  # 用 Fact 增强 Reason
        q_enhanced_cls = q_enhanced[:, 0, :]  # 取增强后的[CLS]

        # 投影到对比学习空间
        q = normalize(self.fact_proj(q_cls), dim=1)
        k = normalize(self.reason_proj(k_cls), dim=1)

        # 后续MoCo流程（动态队列、损失计算等保持不变）
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / 0.07
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.enqueue_dequeue(k)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
    '''
    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B):
        """
        输入: region1和region2是同一文档的两个不同区域（如标题和正文）
        输出: InfoNCE对比损失
        """
        # 编码区域1（query）
        q = self.query_encoder(input_ids=input_ids_A, attention_mask=attention_mask_A).last_hidden_state[:, 0, :]  # [CLS] token
        q = normalize(self.query_proj(q), dim=1)  # 归一化到单位球面

        # 编码区域2（key，动量更新）
        with torch.no_grad():
            self.momentum_update()
            k = self.key_encoder(input_ids=input_ids_B, attention_mask=attention_mask_B).last_hidden_state[:, 0, :]
            k = normalize(self.key_proj(k), dim=1)

        # 加入Cross-Attention交互
        cross_attn = CrossAttentionLayer()
        q_enhanced = cross_attn(q_features, k_features)  # 用区域2增强区域1
        q_enhanced_cls = q_enhanced[:, 0, :]  # 取增强后的[CLS]

        # 计算正样本和负样本的相似度
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # 正样本对
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])  # 负样本队列
        logits = torch.cat([l_pos, l_neg], dim=1) / 0.07  # 温度系数τ=0.07

        # 对比损失（标签0表示正样本在logits的第0列）
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        loss = nn.CrossEntropyLoss()(logits, labels)

        # 更新队列
        self.enqueue_dequeue(k)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
    '''