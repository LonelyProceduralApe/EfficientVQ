import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .VQAttention import VQEmbedding
from utils.operate import list_sum
from .layer import *


class DAGBlock(nn.Module):
    def __init__(
            self,
            inputs: dict[str, nn.Module],  # 输入模块字典，键为输入名，值为处理输入的模块
            merge: str,  # 合并方式
            post_input: nn.Module or None,  # 对合并后的输入进行处理的模块
            middle: nn.Module,  # 中间模块，对处理后的输入进一步处理
            outputs: dict[str, nn.Module],  # 输出模块字典，键为输出名，值为生成输出的模块
    ):
        super().__init__()

        self.input_keys = list(inputs.keys())  # 获取输入键的列表
        self.input_ops = nn.ModuleList(list(inputs.values()))  # 将输入模块转换为 ModuleList
        self.merge = merge  # 合并方式
        self.post_input = post_input  # 对合并后的输入处理的模块

        self.middle = middle  # 中间模块

        self.output_keys = list(outputs.keys())  # 获取输出键的列表
        self.output_ops = nn.ModuleList(list(outputs.values()))  # 将输出模块转换为 ModuleList

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 对每个输入使用对应的模块进行处理
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]

        # 根据 merge 参数将处理后的特征合并
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError

        # 如果定义了 post_input 模块，则对合并后的特征进行处理
        if self.post_input is not None:
            feat = self.post_input(feat)
        # 使用中间模块进一步处理特征
        feat = self.middle(feat)
        # 使用输出模块生成最终的输出
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)

        return feature_dict


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module or None,            # 主要分支，执行核心操作的模块
        shortcut: nn.Module or None,        # 捷径分支，通常是恒等映射或卷积层
        pre_norm: nn.Module or None = None, # 预归一化模块
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm            # 预归一化模块
        self.main = main                    # 主要分支模块
        self.shortcut = shortcut            # 捷径分支模块

    # 定义处理主要分支的方法
    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        # 如果没有预归一化模块，则直接通过主要分支
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果没有主要分支，结果即为输入
        if self.main is None:
            res = x
        # 如果没有捷径分支，结果即为主要分支的输出
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:  # 如果有捷径分支，将主要分支和捷径分支的结果相加
            res = self.forward_main(x) + self.shortcut(x)

        return res


class EfficientVQBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,  # 输入通道数
            heads_ratio: float = 1.0,  # 多头注意力的头数比例
            dim=32,  # 每个注意力头的维度
            expand_ratio: float = 4,  # MBConv块的扩展比例
            norm="bn2d",  # 归一化层的类型
            act_func="hswish",  # 激活函数的类型
    ):
        super(EfficientVQBlock, self).__init__()

        # 上下文模块，包含一个轻量级多头自注意力（LiteMLA）和一个恒等层
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
            ),
            IdentityLayer(),
        )

        # 局部模块，包含一个MBConv层（用于移动网络）和一个恒等层
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class LiteMLA(nn.Module):
    def __init__(
        self,
        in_channels: int,          # 输入通道数
        out_channels: int,         # 输出通道数
        heads: int or None = None, # 头数，如果为None，则根据下面的计算确定
        heads_ratio: float = 1.0,  # 头数比例
        dim=8,                     # 每个头的维度
        use_bias=False,            # 是否使用偏置
        norm=(None, "bn2d"),       # 归一化层的类型
        act_func=(None, None),     # 激活函数类型
        scales: tuple[int, ...] = (5,), # 多尺度处理的尺度大小
        eps=1.0e-15,               # 避免除以零的小数
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        # 计算总的维度
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim

        # 处理构造函数的输入参数
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim

        # 查询、键、值（qkv）卷积层
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )

        # 聚合多尺度特征的模块
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )

        # 核函数，用于处理查询和键
        # self.kernel_func = build_act(kernel_func, inplace=False)
        self.vqemb = VQEmbedding(D=self.dim)

        # 投影层，将多尺度特征投影到输出通道
        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor):
        # 生成多尺度的查询、键和值
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        # 应用ReLU线性注意力并通过投影层
        out = self.vqemb(multi_scale_qkv)
        out = self.proj(out)

        return out

    @staticmethod
    def configure_litemla(model: nn.Module, **kwargs) -> None:
        eps = kwargs.get("eps", None)
        for m in model.modules():
            if isinstance(m, LiteMLA):
                if eps is not None:
                    m.eps = eps

