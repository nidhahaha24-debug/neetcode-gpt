import torch
import torch.nn
from torchtyping import TensorType


class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        t = torch.as_tensor(to_reshape, dtype=torch.float32) if not isinstance(to_reshape, torch.Tensor) else to_reshape.float()
        out = torch.reshape(t, (-1, 2))
        return torch.round(out * 10000) / 10000

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        t = torch.as_tensor(to_avg, dtype=torch.float32) if not isinstance(to_avg, torch.Tensor) else to_avg.float()
        out = torch.mean(t, dim=0)
        return torch.round(out * 10000) / 10000

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        a = torch.as_tensor(cat_one, dtype=torch.float32) if not isinstance(cat_one, torch.Tensor) else cat_one.float()
        b = torch.as_tensor(cat_two, dtype=torch.float32) if not isinstance(cat_two, torch.Tensor) else cat_two.float()
        out = torch.cat((a, b), dim=1)
        return torch.round(out * 10000) / 10000

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        p = torch.as_tensor(prediction, dtype=torch.float32) if not isinstance(prediction, torch.Tensor) else prediction.float()
        t = torch.as_tensor(target, dtype=torch.float32) if not isinstance(target, torch.Tensor) else target.float()
        out = torch.nn.functional.mse_loss(p, t)
        return torch.round(out * 10000) / 10000