# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# from main import device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context

class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        B, N, T, C = input_Q.shape

        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output

class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        B, N, T, C = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output


class SATT(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion):
        super(SATT, self).__init__()
        self.adj = adj
        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)
        self.norm5 = nn.LayerNorm(embed_size)
        self.norm6 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.feed_forward1 = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

        self.ff1 = nn.Linear(embed_size, embed_size)

        self.E = nn.Sequential(
            nn.Linear(adj.shape[0], embed_size * 12),
            nn.ReLU(),
            nn.Linear(embed_size * 12, embed_size * 12)
        )
        self.Att = SMultiHeadAttention(embed_size, heads)
    def forward(self, value, key, query):
        B, N, T, C = query.shape
        query1 = query
        X_G = torch.Tensor(B, N, 0, C).to(device)
        for t in range(query.shape[2]):
            x = query[:, :, t, :]
            A = self.adj.to(device)
            D = (A.sum(-1) ** -0.5)
            D[torch.isinf(D)] = 0.
            D = torch.diag_embed(D)
            A = torch.matmul(torch.matmul(D, A), D)
            x = torch.relu(self.ff1(torch.matmul(A, x)))
            # x = torch.softmax(self.ff2(torch.matmul(A, x)), dim=-1)
            x = x.unsqueeze(2)
            X_G = torch.cat((X_G, x), dim=2)

        X_E = self.E(self.adj.to(device)).reshape(N, T, C).unsqueeze(0)
        query = self.norm5(query + X_E)
        attention = self.attention(query, query, query)  # (B, N, T, C)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))
        X_G = self.norm6(query1+X_G)
        Attention1 = self.Att(X_G,X_G,X_G)
        y = self.dropout(self.norm3(Attention1 + X_G))
        forward1 = self.feed_forward1(y)
        X_G = self.dropout(self.norm4(y + forward1))
        g = torch.sigmoid(self.fs(U_S) + self.fg(X_G))
        out = g * U_S + (1 - g) * X_G
        return out


class TATT(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TATT, self).__init__()
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, t):
        attention = self.attention(query, query, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out



class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(Attention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out
class STBlock(nn.Module):
    def __init__(
            self,
            adj,
            in_channels,
            embed_size,
            T_dim,
            output_T_dim,
            heads,
            forward_expansion,
            dropout=0
    ):
        super(STBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.conv4 = nn.Conv2d(in_channels, embed_size, 1)
        self.conv5 = nn.Conv2d(in_channels, embed_size, 1)
        self.S = SATT(embed_size, heads, adj,dropout, forward_expansion)
        self.T_G = TATT(embed_size, heads, dropout, forward_expansion)
        self.T_L = TATT(embed_size, heads, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)
        self.Att = Attention(d_model=embed_size * T_dim, d_k=embed_size * 2, d_v=embed_size * 2, h=heads, dropout=0)
        # 缩小时间维度
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
        self.global_temporal_conv_gate1 = nn.Conv2d(embed_size, T_dim, (T_dim, 1))
        self.global_temporal_conv_gate2 = nn.Conv2d(embed_size, T_dim, (T_dim, 1))
        self.local_tempooral_conv_gate1 = nn.Conv2d(embed_size, T_dim, (T_dim // 3, 1))
        self.local_tempooral_conv_gate2 = nn.Conv2d(embed_size, T_dim, (T_dim // 3, 1))
        self.pool = nn.MaxPool2d((T_dim - T_dim // 3 + 1, 1))
        self.fs = nn.Linear(embed_size * T_dim, embed_size * T_dim)
        self.fg = nn.Linear(embed_size * T_dim, embed_size * T_dim)

    def forward(self, x):
        input_Transformer = x
        B, N, T, H = input_Transformer.shape
        # **********************************************************************************************
        # 提取空间信息
        output_S = self.S(input_Transformer, input_Transformer, input_Transformer)
        # 残差+层归一化
        output_S = self.norm1(output_S + input_Transformer)
        # **********************************************************************************************
        # 提取全局时间信息
        out_G = (torch.tanh(self.global_temporal_conv_gate1(input_Transformer.permute(0, 3, 2, 1)))
                       * torch.sigmoid(self.global_temporal_conv_gate2(input_Transformer.permute(0, 3, 2, 1)))).permute(
            0, 2, 3, 1)

        out_G = self.conv4(out_G)
        out_G = out_G.permute(0, 2, 3, 1)
        out_G = self.norm3(out_G + input_Transformer)
        out_G = self.T_G(out_G, out_G, out_G, 4).reshape(-1, N, T * H)

        # 提取局部时间信息
        out_L = self.pool(torch.tanh(self.local_tempooral_conv_gate1(input_Transformer.permute(0, 3, 2, 1)))
                          * torch.sigmoid(
            self.local_tempooral_conv_gate2(input_Transformer.permute(0, 3, 2, 1)))).permute(0, 2, 3, 1)
        out_L = self.conv5(out_L)
        out_L = out_L.permute(0, 2, 3, 1)
        out_L = self.norm4(out_L + input_Transformer)
        out_L = self.T_L(out_L, out_L, out_L, 4).reshape(-1, N, T * H)


        g = torch.sigmoid(self.fs(out_L) + self.fg(out_G))
        output_T = g * out_L + (1 - g) * out_G
        output_T = self.norm2(output_T.reshape(B, N, T, H) + input_Transformer)
        # 融合时间+空间
        # out = self.Att(output_T.reshape(-1, 207, 768), output_S.reshape(-1, 207, 768),
        #                output_S.reshape(-1, 207, 768)).reshape(-1, 207, 12, 64)
        out = self.Att(output_S.reshape(-1, N, T * H), output_T.reshape(-1, N, T * H),
                       output_T.reshape(-1, N, T * H)).reshape(-1, N, T, H)
        # **********************************************************************************************

        return out





class DSTTFN(nn.Module):
    def __init__(
            self,
            adj,
            in_channels,
            embed_size,
            T_dim,
            output_T_dim,
            heads,
            forward_expansion,
            dropout=0
    ):
        super(DSTTFN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
        self.ST1 = STBlock(adj,in_channels,embed_size,T_dim,output_T_dim,heads,forward_expansion,dropout=dropout)
        self.ST2 = STBlock(adj, in_channels, embed_size, T_dim, output_T_dim, heads,forward_expansion, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)
        input_Transformer = self.conv1(x)
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)
        out = self.ST1(input_Transformer)
        out = self.norm1(out+input_Transformer)
        out = self.ST2(out)

        #####################################
        out = out.permute(0, 2, 1, 3)
        out = self.relu(self.conv2(out))
        out = out.permute(0, 3, 2, 1)
        out = self.conv3(out)
        out = out.squeeze(1)
        return out.permute(0, 2, 1)
