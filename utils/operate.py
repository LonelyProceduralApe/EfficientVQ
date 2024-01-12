import torch
import torch.nn.functional as F
from torchpack import distributed


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def resize(
    x: torch.Tensor,                    # 输入的张量
    size: any or None = None,           # 目标大小
    scale_factor: list[float] or None = None,  # 缩放因子
    mode: str = "bicubic",              # 插值模式
    align_corners: bool or None = False, # 对齐角落的选项
) -> torch.Tensor:
    # 如果模式是 'bilinear' 或 'bicubic'，使用 torch.nn.functional.interpolate
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,                        # 目标大小
            scale_factor=scale_factor,        # 缩放因子
            mode=mode,                        # 插值模式
            align_corners=align_corners,      # 是否对齐角落
        )
    # 如果模式是 'nearest' 或 'area'，也使用 interpolate 但不需要 align_corners
    elif mode in {"nearest", "area"}:
        return F.interpolate(
            x,
            size=size,                   # 目标大小
            scale_factor=scale_factor,   # 缩放因子
            mode=mode                    # 插值模式
        )
    # 如果提供了未实现的模式，抛出异常
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, is_distributed=True):
        self.is_distributed = is_distributed
        self.sum = 0
        self.count = 0

    def _sync(self, val: torch.Tensor or int or float) -> torch.Tensor or int or float:
        return sync_tensor(val, reduce="sum") if self.is_distributed else val

    def update(self, val: torch.Tensor or int or float, delta_n=1):
        self.count += self._sync(delta_n)
        self.sum += self._sync(val * delta_n)

    def get_count(self) -> torch.Tensor or int or float:
        return self.count.item() if isinstance(self.count, torch.Tensor) and self.count.numel() == 1 else self.count

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg


def sync_tensor(tensor: torch.Tensor or float, reduce="mean") -> torch.Tensor or list[torch.Tensor]:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(distributed.size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list


def list_sum(x: list) -> any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: list) -> any:
    return list_sum(x) / len(x)
