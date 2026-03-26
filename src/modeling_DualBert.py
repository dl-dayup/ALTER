import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, encoder, K=65536, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        self.K = K  # 队列大小
        self.m = m  # 动量系数
        self.T = T  # 温度参数

        # 创建编码器和动量编码器
        self.encoder_q = encoder
        self.encoder_k = encoder

        # 初始化动量编码器的参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 创建队列
        self.register_buffer("queue", torch.randn(K, encoder.config.hidden_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # 队列大小必须是批次大小的整数倍

        # 替换队列中的旧键
        self.queue[ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.K  # 移动指针
        self.queue_ptr[0] = ptr

    def forward(self, query, key):
        # 查询和键的编码
        print('query, shape', query.shape)
        q = self.encoder_q(query)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()  # 更新动量编码器
            k = self.encoder_k(key)
            k = F.normalize(k, dim=1)

        # 计算对比损失
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = F.cross_entropy(logits, labels)

        # 更新队列
        self._dequeue_and_enqueue(k)

        return loss

# 在两个 Bert 的最外面加入了 Cross attention
class DualEncoderModel(nn.Module):
    def __init__(self, model_name="/jupyterhub_data/test9/model/bert-base-chinese"):
        super(DualEncoderModel, self).__init__()
        # 初始化两个BERT编码器
        self.encoder_A = BertModel.from_pretrained(model_name)
        self.encoder_B = BertModel.from_pretrained(model_name)
        self.moco_A = MoCo(self.encoder_A)
        self.moco_B = MoCo(self.encoder_B)

        # 初始化交叉注意力模块
        self.cross_attention_A = nn.MultiheadAttention(
            embed_dim=self.encoder_A.config.hidden_size,
            num_heads=8,
            dropout=0.1
        )
        self.cross_attention_B = nn.MultiheadAttention(
            embed_dim=self.encoder_B.config.hidden_size,
            num_heads=8,
            dropout=0.1
        )

        # 层归一化和Dropout
        self.norm_A = nn.LayerNorm(self.encoder_A.config.hidden_size)
        self.norm_B = nn.LayerNorm(self.encoder_B.config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B):
        # 编码器 A 的前向传播
        outputs_A = self.encoder_A(input_ids=input_ids_A, attention_mask=attention_mask_A)
        last_hidden_state_A = outputs_A.last_hidden_state

        # 编码器 B 的前向传播
        outputs_B = self.encoder_B(input_ids=input_ids_B, attention_mask=attention_mask_B)
        last_hidden_state_B = outputs_B.last_hidden_state

        # 交叉注意力：编码器 A 使用编码器 B 的输出作为上下文
        cross_attn_output_A, _ = self.cross_attention_A(
            query=last_hidden_state_A.permute(1, 0, 2),
            key=last_hidden_state_B.permute(1, 0, 2),
            value=last_hidden_state_B.permute(1, 0, 2)
        )
        cross_attn_output_A = cross_attn_output_A.permute(1, 0, 2)
        cross_attn_output_A = self.norm_A(last_hidden_state_A + self.dropout(cross_attn_output_A))

        # 交叉注意力：编码器 B 使用编码器 A 的输出作为上下文
        cross_attn_output_B, _ = self.cross_attention_B(
            query=last_hidden_state_B.permute(1, 0, 2),
            key=last_hidden_state_A.permute(1, 0, 2),
            value=last_hidden_state_A.permute(1, 0, 2)
        )
        cross_attn_output_B = cross_attn_output_B.permute(1, 0, 2)
        cross_attn_output_B = self.norm_B(last_hidden_state_B + self.dropout(cross_attn_output_B))

        # MoCo 对比学习损失
        loss_A = self.moco_A(last_hidden_state_A, last_hidden_state_B)
        loss_B = self.moco_B(last_hidden_state_B, last_hidden_state_A)
        return loss_A + loss_B