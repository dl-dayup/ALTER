import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


class CoAttentionBlock(nn.Module):
    """
    层间协同注意力模块
    在每一层Transformer中执行文本<->主题的双向注意力
    """

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0.1):
        super(CoAttentionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # Text-to-Topic 注意力
        self.text2topic_query = nn.Linear(hidden_size, hidden_size)
        self.text2topic_key = nn.Linear(hidden_size, hidden_size)
        self.text2topic_value = nn.Linear(hidden_size, hidden_size)

        # Topic-to-Text 注意力
        self.topic2text_query = nn.Linear(hidden_size, hidden_size)
        self.topic2text_key = nn.Linear(hidden_size, hidden_size)
        self.topic2text_value = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.text2topic_output = nn.Linear(hidden_size, hidden_size)
        self.topic2text_output = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """将隐藏状态转换为注意力分数所需的形状"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, text_states, topic_states, attention_mask=None):
        """
        Args:
            text_states: [batch_size, text_seq_len, hidden_size] 文本表示
            topic_states: [batch_size, 1, hidden_size] 主题表示（我们将其视为长度为1的序列）
            attention_mask: [batch_size, text_seq_len] 文本注意力掩码
        Returns:
            updated_text_states: [batch_size, text_seq_len, hidden_size] 主题感知的文本表示
            updated_topic_states: [batch_size, 1, hidden_size] 文本引导的主题表示
        """
        batch_size, text_seq_len, _ = text_states.size()

        # === 1. Text-to-Topic Attention ===
        # 文本作为Query，主题作为Key/Value
        text_query_layer = self.transpose_for_scores(self.text2topic_query(text_states))  # [B, H, L, d_h]
        topic_key_layer = self.transpose_for_scores(self.text2topic_key(topic_states))  # [B, H, 1, d_h]
        topic_value_layer = self.transpose_for_scores(self.text2topic_value(topic_states))  # [B, H, 1, d_h]

        # 计算注意力分数 [B, H, L, 1]
        text2topic_attention_scores = torch.matmul(text_query_layer, topic_key_layer.transpose(-1, -2))
        text2topic_attention_scores = text2topic_attention_scores / torch.sqrt(
            torch.tensor(self.attention_head_size, dtype=torch.float32)
        )

        # 应用注意力权重 [B, H, L, d_h]
        text2topic_attention_probs = F.softmax(text2topic_attention_scores, dim=-1)
        text2topic_attention_probs = self.dropout(text2topic_attention_probs)

        text2topic_context_layer = torch.matmul(text2topic_attention_probs, topic_value_layer)
        text2topic_context_layer = text2topic_context_layer.permute(0, 2, 1, 3).contiguous()
        new_text_context_shape = text2topic_context_layer.size()[:-2] + (self.hidden_size,)
        text2topic_context_layer = text2topic_context_layer.view(*new_text_context_shape)

        # 投影输出
        updated_text_states = self.text2topic_output(text2topic_context_layer)

        # === 2. Topic-to-Text Attention ===
        # 主题作为Query，文本作为Key/Value
        topic_query_layer = self.transpose_for_scores(self.topic2text_query(topic_states))  # [B, H, 1, d_h]
        text_key_layer = self.transpose_for_scores(self.topic2text_key(text_states))  # [B, H, L, d_h]
        text_value_layer = self.transpose_for_scores(self.topic2text_value(text_states))  # [B, H, L, d_h]

        # 计算注意力分数 [B, H, 1, L]
        topic2text_attention_scores = torch.matmul(topic_query_layer, text_key_layer.transpose(-1, -2))
        topic2text_attention_scores = topic2text_attention_scores / torch.sqrt(
            torch.tensor(self.attention_head_size, dtype=torch.float32)
        )

        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            # 扩展注意力掩码到注意力头维度 [B, 1, 1, L]
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            topic2text_attention_scores = topic2text_attention_scores + extended_attention_mask

        # 应用注意力权重 [B, H, 1, d_h]
        topic2text_attention_probs = F.softmax(topic2text_attention_scores, dim=-1)
        topic2text_attention_probs = self.dropout(topic2text_attention_probs)

        topic2text_context_layer = torch.matmul(topic2text_attention_probs, text_value_layer)
        topic2text_context_layer = topic2text_context_layer.permute(0, 2, 1, 3).contiguous()
        new_topic_context_shape = topic2text_context_layer.size()[:-2] + (self.hidden_size,)
        topic2text_context_layer = topic2text_context_layer.view(*new_topic_context_shape)

        # 投影输出
        updated_topic_states = self.topic2text_output(topic2text_context_layer)

        return updated_text_states, updated_topic_states


class CoAttentionTransformerLayer(nn.Module):
    """完整的Co-attention Transformer层，包含自注意力、Co-attention和前馈网络"""

    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1):
        super(CoAttentionTransformerLayer, self).__init__()

        # 文本自注意力
        self.text_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_probs_dropout_prob,
            batch_first=True
        )

        # 主题自注意力（虽然主题只有一个向量，但为了统一接口保留）
        self.topic_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_probs_dropout_prob,
            batch_first=True
        )

        # Co-attention模块
        self.co_attention = CoAttentionBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

        # 前馈网络
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(hidden_dropout_prob)
        )

        self.topic_ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size // 4),  # 主题FFN可以小一些
            nn.GELU(),
            nn.Linear(intermediate_size // 4, hidden_size),
            nn.Dropout(hidden_dropout_prob)
        )

        # 层归一化
        self.text_norm1 = nn.LayerNorm(hidden_size)
        self.text_norm2 = nn.LayerNorm(hidden_size)
        self.text_norm3 = nn.LayerNorm(hidden_size)

        self.topic_norm1 = nn.LayerNorm(hidden_size)
        self.topic_norm2 = nn.LayerNorm(hidden_size)
        self.topic_norm3 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, text_states, topic_states, text_attention_mask=None):
        """
        Args:
            text_states: [batch_size, seq_len, hidden_size]
            topic_states: [batch_size, 1, hidden_size]
            text_attention_mask: [batch_size, seq_len]
        """
        # === 文本路径 ===
        # 1. 文本自注意力
        text_self_attn_output, _ = self.text_self_attention(
            text_states, text_states, text_states,
            key_padding_mask=~text_attention_mask if text_attention_mask is not None else None
        )
        text_states = self.text_norm1(text_states + self.dropout(text_self_attn_output))

        # 2. Co-attention (Text-to-Topic)
        co_attn_text, co_attn_topic = self.co_attention(
            text_states, topic_states, text_attention_mask
        )
        text_states = self.text_norm2(text_states + self.dropout(co_attn_text))

        # 3. 文本前馈网络
        text_ffn_output = self.text_ffn(text_states)
        text_states = self.text_norm3(text_states + self.dropout(text_ffn_output))

        # === 主题路径 ===
        # 1. 主题自注意力（实际上主要是残差连接和归一化）
        topic_self_attn_output, _ = self.topic_self_attention(
            topic_states, topic_states, topic_states
        )
        topic_states = self.topic_norm1(topic_states + self.dropout(topic_self_attn_output))

        # 2. Co-attention (Topic-to-Text) 的结果已经在上面计算了
        topic_states = self.topic_norm2(topic_states + self.dropout(co_attn_topic))

        # 3. 主题前馈网络
        topic_ffn_output = self.topic_ffn(topic_states)
        topic_states = self.topic_norm3(topic_states + self.dropout(topic_ffn_output))

        return text_states, topic_states


# 使用示例
if __name__ == "__main__":
    # 配置参数
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    num_heads = 12
    intermediate_size = 3072
    num_layers = 3

    # 创建模拟输入
    text_input = torch.randn(batch_size, seq_len, hidden_size)  # 文本嵌入
    topic_input = torch.randn(batch_size, 1, hidden_size)  # 主题嵌入（投影后）
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)  # 注意力掩码

    # 创建Co-attention层堆叠
    co_attention_layers = nn.ModuleList([
        CoAttentionTransformerLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size
        ) for _ in range(num_layers)
    ])

    # 前向传播
    text_states = text_input
    topic_states = topic_input

    for layer in co_attention_layers:
        text_states, topic_states = layer(text_states, topic_states, attention_mask)

    print("输出文本状态形状:", text_states.shape)  # [2, 128, 768]
    print("输出主题状态形状:", topic_states.shape)  # [2, 1, 768]

    # 提取 [CLS] token 用于检索
    cls_representation = text_states[:, 0, :]  # 取第一个token ([CLS])
    print("[CLS] 表示形状:", cls_representation.shape)  # [2, 768]