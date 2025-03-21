import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .utils import extract

###############################
# 训练阶段：差异域 + 注意力引导
###############################

class GaussianDiffusionTrainer_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        # 构造beta序列及相关预计算系数（扩散过程系数针对差异域 d_0）
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        输入 x_0 形状：[B, 2, H, W]，其中：
            - 第1通道 ct：目标图像
            - 第2通道 cbct：条件图像
        """
        # 随机采样时间步 t
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # 分离 ct 与 cbct，并扩充 channel 维度
        ct = x_0[:, 0, :, :].unsqueeze(1)
        cbct = x_0[:, 1, :, :].unsqueeze(1)

        # 差异域：d_0 = ct - cbct
        d_0 = ct - cbct
        noise = torch.randn_like(d_0)

        # 根据扩散公式构造噪声差异 d_t
        d_t = (extract(self.sqrt_alphas_bar, t, d_0.shape) * d_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, d_0.shape) * noise)

        # 将条件图像拼接回去
        model_input = torch.cat((d_t, cbct), dim=1)
        predicted = self.model(model_input, t)
        predicted_noise = predicted[0]
        loss = F.mse_loss(predicted_noise, noise, reduction='sum')
        return loss