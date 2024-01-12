import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VQEmbedding(nn.Module):
    '''
    K: the length of the codebook
    D: the latent dimension of the codebook
    '''

    def __init__(self, K=512, D=16, beta=0.25, eps=1.0e-15):
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta
        self.eps = eps
        self.embedding = nn.Embedding(K, D)

    def forward(self, qkv):
        B, _, H, W = list(qkv.size())

        # 转换为浮点精度以避免数值问题
        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        # 重塑和分割查询、键和值
        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.D,
                H * W,
            ),
        )                                       # bs, head, 3*dim, h*w
        qkv = torch.transpose(qkv, -1, -2)      # bs, head, h*w, 3*dim
        q, k, v = (
            qkv[..., 0 : self.D],
            qkv[..., self.D : 2 * self.D],
            qkv[..., 2 * self.D :],
        )                                           # bs, head, h*w, dim

        k_res=k.reshape(-1, 16)                     # ((bs head h*w), dim)
        indices = vq(k_res, self.embedding.weight)  # (bs head h*w)

        one_hot = F.one_hot(indices, num_classes=self.embedding.weight.shape[0]).to(k.dtype)      # (bs head h*w), 512

        q_res=q.reshape(-1, 16)                                         # ((bs head h*w), dim)
        v_res=v.reshape(-1, 16)
        QC_t = torch.matmul(q_res, self.embedding.weight.t())           # ((bs head h*w), 512)
        O_tV = torch.matmul(one_hot.t(), v_res)                         # (512, dim)

        numerator=torch.matmul(torch.exp(QC_t), O_tV)                               # ((bs head h*w), dim)
        denominator=torch.matmul(torch.exp(QC_t), one_hot.t()).to('cuda:0')         # ((bs head h*w), (bs head h*w))
        ones_matrix = torch.ones(denominator.shape[0], 1).to('cuda:0')              # ((bs head h*w), 1)
        denominator = torch.matmul(denominator, ones_matrix).squeeze()
        denominator = denominator + self.eps
        result = numerator / denominator.unsqueeze(1)

        result=torch.reshape(result,(B, -1, H, W))
        return result


def vq(input, codebook):
    '''
    input: (b, d)
    codeebook: (K, c)
    return: (b,)
    '''
    input = input.unsqueeze(1)  # (b, 1, d)
    codebook = codebook.unsqueeze(0) # (1, l, d)
    distances = ((input - codebook) ** 2).sum(-1)   # (b, l, d) --> (b, l)
    _, indices = distances.min(-1)
    return indices.detach()
