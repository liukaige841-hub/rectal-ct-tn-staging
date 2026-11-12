"""最终 T/N 分类器：ResNet152 与 layer1 后双分支。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BranchHead(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.body(x).flatten(1))


class DualBranchResNet(nn.Module):
    """2048 维主干特征与两个 128 维分支特征拼接为 2304 维。"""

    def __init__(self, backbone: nn.Module, num_classes: int = 2, branch_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.stem_pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.rest = nn.Sequential(backbone.layer2, backbone.layer3, backbone.layer4)
        self.avgpool = backbone.avgpool
        feature_dim = backbone.fc.in_features
        layer1_channels = backbone.layer1[-1].conv3.out_channels
        self.maxpool_branch = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=2, padding=2))
        self.maxpool_head = BranchHead(layer1_channels, branch_dim)
        self.crop_head = BranchHead(layer1_channels, branch_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 2 * branch_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.layer1(self.stem_pool(self.stem(x)))
        pooled = self.maxpool_head(self.maxpool_branch(shallow))
        height, width = shallow.shape[-2:]
        crop = shallow[..., height // 4:height // 4 + height // 2,
                       width // 4:width // 4 + width // 2]
        crop = F.interpolate(crop, size=(height // 2, width // 2), mode="bilinear", align_corners=False)
        centered = self.crop_head(crop)
        main = self.avgpool(self.rest(shallow)).flatten(1)
        return self.classifier(torch.cat([main, pooled, centered], dim=1))


def build_classifier(config: dict, pretrained: bool | None = None) -> nn.Module:
    model_config = config["model"]
    if str(model_config.get("backbone", "resnet152")).lower() != "resnet152":
        raise ValueError("公开包仅支持最终分类结构 resnet152")
    if not bool(model_config.get("dual_branch", True)):
        raise ValueError("公开包仅支持最终分类结构 dual_branch=true")
    if int(config.get("data", {}).get("in_channels", 3)) != 3:
        raise ValueError("最终分类器要求三通道 2.5D 输入")
    use_pretrained = bool(model_config.get("pretrained", True)) if pretrained is None else pretrained
    weights = models.ResNet152_Weights.IMAGENET1K_V2 if use_pretrained else None
    backbone = models.resnet152(weights=weights)
    return DualBranchResNet(backbone, num_classes=int(config.get("num_classes", 2)))


def load_classifier_weights(model: nn.Module, path, device: torch.device) -> None:
    from src.inference_utils import load_state_dict

    model.load_state_dict(load_state_dict(path, device, preferred_keys=("state_dict", "model")), strict=True)
