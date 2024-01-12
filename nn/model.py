import torch
import torch.nn as nn

from .backbone import EfficientVQBackbone, SegHead

class EfficientVQ(nn.Module):
    def __init__(self, backbone: EfficientVQBackbone, head: SegHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)

        return x["segout"]

def efficientVQ(dataset: str, **kwargs) -> EfficientVQ:
    from .backbone import EfficientVQBackbone

    backbone = EfficientVQBackbone()

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[16, 8, 4],
            head_stride=4,
            head_width=32,
            head_depth=1,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=19,
        )
    else:
        raise NotImplementedError
    model = EfficientVQ(backbone, head)
    return model

