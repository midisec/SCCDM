import os
import time
import datetime
import torch
from torch.utils.data import DataLoader
from config import Config
from src.models.unet import UNet
from src.diffusion.trainer import GaussianDiffusionTrainer_cond
from src.diffusion.sampler import GaussianDiffusionSampler_cond
from src.data.dataset import FootDataset, FootDataset2
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger

def main():
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(Config.log_file)
    os.makedirs(Config.save_dir, exist_ok=True)

    # Data loaders
    train_dataloader = DataLoader(
        FootDataset2(Config.dataset_path, image_size=Config.image_size),
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=8,
    )

    eval_dataloader = DataLoader(
        FootDataset2(Config.eval_dataset_path, image_size=Config.image_size),
        batch_size=Config.batch_size,
        shuffle=False,
    )

    # Model setup
    net_model = UNet(
        Config.T, Config.ch, Config.ch_mult, Config.attn,
        Config.num_res_blocks, Config.dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        net_model.parameters(),
        lr=Config.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )

    trainer = GaussianDiffusionTrainer_cond(
        net_model, Config.beta_1, Config.beta_T, Config.T
    ).to(device)
    
    sampler = GaussianDiffusionSampler_cond(
        net_model, Config.beta_1, Config.beta_T, Config.T
    , Config.psi, Config.s).to(device)

    # Training loop
    best_psnr = float('-inf')
    best_ssim = float('-inf')
    prev_time = time.time()

    for epoch in range(Config.n_epochs):
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            condition = batch['condition'].to(device)
            target = batch['target'].to(device)
            x_0 = torch.cat((target, condition), 1)
            loss = trainer(x_0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), Config.grad_clip
            )
            optimizer.step()

        # Logging
        time_duration = datetime.timedelta(seconds=(time.time() - prev_time))
        epoch_left = Config.n_epochs - epoch
        time_left = datetime.timedelta(seconds=epoch_left * (time.time() - prev_time))
        prev_time = time.time()

        logger.info(
            f"[Epoch {epoch}/{Config.n_epochs}] "
            f"[ETA: {time_left}] "
            f"[EpochDuration: {time_duration}] "
            f"[MSELoss: {loss.item()}]"
        )

        # Evaluation phase
        if epoch > 1900:
            ssim_values = []
            psnr_values = []

            with torch.no_grad():
                for eval_batch in eval_dataloader:
                    random_noise = torch.randn_like(eval_batch['target'].to(device))
                    x_T = torch.cat((random_noise, eval_batch['condition'].to(device)), 1)
                    generated_images = sampler(x_T)

                    for i in range(len(eval_batch['target'])):
                        generated_image = generated_images[i, 0].cpu().numpy()
                        target_image = eval_batch['target'][i, 0].cpu().numpy()
                        ssim_value, psnr_value = calculate_metrics(generated_image, target_image)
                        ssim_values.append(ssim_value)
                        psnr_values.append(psnr_value)

            avg_ssim = sum(ssim_values) / len(ssim_values)
            avg_psnr = sum(psnr_values) / len(psnr_values)

            logger.info(
                f"[Epoch {epoch}] "
                f"[Average SSIM: {avg_ssim:.4f}] "
                f"[Average PSNR: {avg_psnr:.4f}]"
            )

            # Save best model
            if avg_psnr > best_psnr or avg_ssim > best_ssim:
                best_psnr = max(best_psnr, avg_psnr)
                best_ssim = max(best_ssim, avg_ssim)
                torch.save(
                    net_model.state_dict(),
                    os.path.join(Config.save_dir, f'best_model_epoch_{epoch}.pt')
                )
                logger.info(
                    f"Model saved at epoch {epoch} "
                    f"with SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}"
                )

            # Regular checkpoint saving
            if epoch % 20 == 0:
                torch.save(
                    net_model.state_dict(),
                    os.path.join(Config.save_dir, f'ckpt_{epoch}.pt')
                )

if __name__ == "__main__":
    main()