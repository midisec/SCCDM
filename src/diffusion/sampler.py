import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .utils import extract

class GaussianDiffusionSampler_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, psi=1.0, s=0.1):
        """
        参数:
          - psi：注意力图阈值，用于生成二值 mask
          - s：噪声修正的缩放因子
        """
        super().__init__()
        self.model = model
        self.T = T
        self.psi = psi  # attention mask 阈值
        self.s = s      # 噪声修正缩放因子

        # 构造beta序列及相关扩散系数（针对差异域 d_t）
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance',
                             self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def predict_d0_from_d_t(self, d_t, t, eps):
        """
        根据公式预测 d_0
            d_0 = (d_t - sqrt(1 - alphas_cumprod) * eps) / sqrt(alphas_cumprod)
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, d_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, d_t.shape)
        d_0 = (d_t - sqrt_one_minus_alphas_cumprod_t * eps) / sqrt_alphas_cumprod_t
        return d_0

    def forward(self, x_T):
        """
        x_T 形状：[B, 2, H, W]，其中条件图 cbct 放在第二通道，
        初始差异 d_T 用随机噪声初始化，采样过程中逐步恢复 d_0，
        最后重构 ct = cbct + d_0。
        """
        with torch.no_grad():
            batch_size = x_T.size(0)
            device = x_T.device

            # 提取条件图像 cbct（第二通道）
            cbct = x_T[:, 1:2, :, :]
            # 初始化差异 d_T 为随机噪声（与 cbct 尺寸相同）
            d_t = torch.randn_like(cbct)

            for time_step in reversed(range(self.T)):
                t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)

                # 将当前差异 d_t 与条件图像拼接为模型输入
                model_input = torch.cat((d_t, cbct), dim=1)
                # 模型输出：预测噪声 eps 和注意力图 attention_map
                eps, attention_map = self.model(model_input, t)

                # --- 利用注意力图构造 mask ---
                # 假设 attention_map 形状为 [B, N, _]，N 可 reshape 为 HxW
                B, N, _ = attention_map.shape
                H = W = int(N ** 0.5)
                attention_map_sum = attention_map.sum(dim=1)  # 汇聚查询维度
                attention_map_sum = attention_map_sum.view(B, 1, H, W)
                # 根据阈值 psi 得到二值 mask M_t
                M_t = (attention_map_sum > self.psi).float()
                M_t = F.interpolate(M_t, size=d_t.shape[2:], mode='nearest')

                # --- 利用预测噪声还原出估计的 d_0 ---
                sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, d_t.shape)
                sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, d_t.shape)
                d_hat_0 = (d_t - sqrt_one_minus_alphas_cumprod_t * eps) / sqrt_alphas_cumprod_t

                # --- 对 d_hat_0 应用高斯模糊 ---
                d_tilde_0 = self.gaussian_blur(d_hat_0)
                # 重构模糊后的 d_tilde_t
                d_tilde_t = sqrt_alphas_cumprod_t * d_tilde_0 + sqrt_one_minus_alphas_cumprod_t * eps

                # --- 利用注意力 mask 融合原始 d_t 与模糊后的 d_tilde_t ---
                d_bar_t = (1 - M_t) * d_t + M_t * d_tilde_t

                # --- 使用修正后的 d_bar_t 进行二次噪声预测 ---
                refined_input = torch.cat((d_bar_t, cbct), dim=1)
                eps_bar, _ = self.model(refined_input, t)
                # 根据参数 s 调整噪声预测
                eps_tilde = eps_bar + (1 + self.s) * (eps - eps_bar)

                # --- 计算反向扩散步骤的均值与方差，更新 d_t ---
                alphas_t = extract(self.alphas, t, d_t.shape)
                sqrt_alpha_t = torch.sqrt(alphas_t)
                betas_t = extract(self.betas, t, d_t.shape)
                mean = (1 / sqrt_alpha_t) * (d_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * eps_tilde)
                var = extract(self.posterior_variance, t, d_t.shape)

                if time_step > 0:
                    noise = torch.randn_like(d_t)
                    d_t = mean + torch.sqrt(var) * noise
                else:
                    d_t = mean

            # 采样结束后，根据 d_0 重构 ct：ct = cbct + d_0
            ct = cbct + d_t
            x_0 = torch.cat((ct, cbct), dim=1)
            x_0 = torch.clamp(x_0, -1., 1.)
        return x_0

    def gaussian_blur(self, x):
        """对输入 x 应用固定核大小和 sigma 的高斯模糊"""
        kernel_size = 5
        sigma = 1.0
        # torchvision.transforms.functional 中的 gaussian_blur 接受 (kernel_size, sigma)
        return TF.gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)