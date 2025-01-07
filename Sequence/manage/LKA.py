import torch
import torch.nn as nn
class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, 3, stride=1, padding=1, bias=True, groups=dim)
    def forward(self, x):
        # x = x.transpose(1, 2)  # 转置以适应 Conv1d (B, C, L)
        x = self.dwconv(x)
        return x  # 转置回 (B, L, C)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 使用1x1卷积处理输入特征
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.dwconv = DWConvLKA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.fc1(x.transpose(1, 2))  # 转置以适应 Conv1d (B, C, L)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)  # 使用 DWConvLKA 处理
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.transpose(1, 2)  # 转置回 (B, L, C)

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv1d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv1d(dim, dim, 1)
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        # print(attn.size())
        return u * attn

class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv1d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv1d(d_model, d_model, 1)
    def forward(self, x):
        shortcut = x.clone()
        # print(x.size())
        x = self.proj_1(x.transpose(1, 2))  # 转置以适应 Conv1d (B, C, L)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x.transpose(1, 2) + shortcut  # 转置回 (B, L, C)
        return x

class LKABlock(nn.Module):
    def __init__(self,dim,mlp_ratio=4.,drop=0.4,act_layer=nn.GELU,linear=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # 归一化层
        self.attn = SpatialAttention(dim)  # 空间注意力模块
        self.dropout = nn.Dropout(drop)  # 使用 Dropout
        self.norm2 = nn.LayerNorm(dim)  # 另一个归一化层
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):        
        y = self.norm1(x)  # (B, L, C)
        # y=y.permute(0, 2, 1)
        y = self.attn(y)   # 应用注意力
        y = self.layer_scale_1 * y
        y = self.dropout(y)  # 应用 Dropout
        x = x + y  # 残差连接(B, L, C)
        # x=x.permute(0, 2, 1)
        y = self.norm2(x)  # (B, L, C)
        # y=y.permute(0, 2, 1)
        y = self.mlp(y)    # 应用 MLP
        y = self.layer_scale_2 * y
        y = self.dropout(y)  # 应用 Dropout
        x = x + y  # 残差连接
        return x  # (B, C,L) 