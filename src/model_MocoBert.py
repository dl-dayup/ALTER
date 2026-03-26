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

    def forward(self, region1, region2, left2right=True):
        # region1: [batch_size, seq_len, hidden_dim]
        # region2: [batch_size, seq_len, hidden_dim]
        region2 = region2.to(region1.device)

        self.query = self.query.to(region1.device)
        self.key = self.key.to(region1.device)
        self.value = self.value.to(region1.device)

        if left2right:
            q = self.query(region2)
            k = self.key(region1)
            v = self.value(region1)
        else:
            q = self.query(region1)
            k = self.key(region2)
            v = self.value(region2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights, v)  # 加权求和
        if left2right:
            return output + region2  # 残差连接
        else:
            return output + region1

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
        # self.llm_teacher = AutoModel.
        # self.bert_teacher =
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
        """
        keys: [batch_size, feature_dim]
        """
        assert keys.shape[1] == self.feature_dim, \
            f"Keys feature_dim {keys.shape[1]} != queue feature_dim {self.feature_dim}"

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 检查队列剩余空间
        if ptr + batch_size > self.queue_size:
            # 分两段填充
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T

        # 更新指针
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
        k_enhanced = cross_attn(q_features, k_features)  # 默认q_features 为KV
        k_enhanced_cls = k_enhanced[:, 0, :]  # 取增强后的[CLS]

        # 投影到对比学习空间
        q = normalize(self.fact_proj(q_cls), dim=1)
        k = normalize(self.reason_proj(k_enhanced_cls), dim=1)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg1 = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        l_neg2 = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        
        logits = torch.cat([l_pos, l_neg], dim=1) / 0.07
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.enqueue_dequeue(k)

        # # 双教师蒸馏
        # with torch.no_grad():
        #     llm_emb = self.llm_teacher(fact_batch["input_ids"]).last_hidden_state[:, 0]
        #     bert_emb = self.bert_teacher(fact_batch["input_ids"]).last_hidden_state[:, 0]
        # loss_distill, _ = distiller(q, llm_emb, bert_emb)
        #
        # # 总损失
        # total_loss = 0.5 * loss_moco + 0.3 * loss_sub + 0.2 * loss_distill
        loss.backward()


        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
