import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(generated_image, target_image):
    ssim_value = ssim(generated_image, target_image, 
                      data_range=target_image.max() - target_image.min())
    psnr_value = psnr(target_image, generated_image, 
                      data_range=target_image.max() - target_image.min())
    return ssim_value, psnr_value