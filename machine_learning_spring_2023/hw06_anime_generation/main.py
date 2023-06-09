import argparse
import math
import os
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import torch
import torchvision
from accelerate import Accelerator
from dataset import MyDataset
from ema_pytorch import EMA
from model import (GaussianDiffusion, UNet, cosine_beta_schedule, exists,
                   linear_beta_schedule)
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from utils import count_parameters

torch.backends.cudnn.benchmark = True
torch.manual_seed(4096)

if torch.cuda.is_available():
    torch.cuda.manual_seed(4096)

# Reference:
# DDPM https://arxiv.org/pdf/2006.11239.pdf
path = '../../../dataset/crypko'
IMG_SIZE = 64             # Size of images, do not change this if you do not know why you need to change
batch_size = 128
train_num_steps = 200000  # total training steps, DDPM uses 800k steps for CIFAR-10
lr = 1e-3                 # DDPM uses 2e-4 (not sweep)
grad_steps = 1            # gradient accumulation steps, the equivalent batch size for updating equals to batch_size * grad_steps = 16 * 1
ema_decay = 0.995         # exponential moving average decay, DDPM uses 0.9999

channels = 64             # Numbers of channels of the first layer of CNN
dim_mults = (1, 2, 4, 8)

timesteps = 1000          # Number of steps (adding noise)
beta_schedule_fn = cosine_beta_schedule



class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        split_batches = True,
        inception_block_idx = 2048
    ):
        def has_int_squareroot(num):
            return (math.sqrt(num) ** 2) == num

        def cycle(dl):
            while True:
                for data in dl:
                    yield data

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'

        super().__init__()

        # Record hparams
        self.hparams = {
            'batch_size': train_batch_size,
            'learning_rate': train_lr,
            'iterations': train_num_steps,
            'ema_update_every': ema_update_every,
            'ema_decay': ema_decay,
        }

        # See https://huggingface.co/docs/accelerate/usage_guides/tracking to learn how to log
        # with accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'no',
            log_with='tensorboard',
            project_dir='runs/'
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = MyDataset(folder, self.image_size)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @staticmethod
    def num_to_groups(num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups

        if remainder > 0:
            arr.append(remainder)

        return arr

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, ckpt: str):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(ckpt, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        accelerator.init_trackers(str(datetime.now()), config={})

        device = accelerator.device

        pbar = tqdm(
            initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process,
            ncols=0, desc='Training'
        )

        while self.step < self.train_num_steps:
            total_loss = 0.

            # ------------------------ Discriminator ------------------------- #
            # TODO: Implement data distribution classifier

            for _ in range(self.gradient_accumulate_every):
                # ------------ Generator part I - Diffusion Loss ------------- #
                data = next(self.dl).to(device)

                with self.accelerator.autocast():
                    loss = self.model(data)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                self.accelerator.backward(loss)

                # ---------- Generator part II - Adversarial Loss ------------ #
                # TODO: Sample image from normal distribution, tune noise predictor by the
                #       guidance of discriminator.

            accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

            # Log for improving performance
            pbar.set_postfix(loss=total_loss)
            accelerator.log({'train_loss': total_loss}, step=self.step)

            accelerator.wait_for_everyone()

            self.opt.step()
            self.opt.zero_grad()

            accelerator.wait_for_everyone()

            self.step += 1
            if accelerator.is_main_process:
                # replace with inference()
                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()

                    with torch.no_grad():
                        milestone = self.step // self.save_and_sample_every
                        batches = self.num_to_groups(self.num_samples, self.batch_size)
                        all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                    all_images = torch.cat(all_images_list, dim = 0)
                    utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                    self.save(milestone)

            pbar.update(1)

    @torch.no_grad()
    def inference(self, num=1000, n_iter=5, loop=1, output_path='./submission', sample_fn=None):
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for i in range(n_iter):
            batches = self.num_to_groups(num // n_iter, 200)
            all_images = list(map(
                lambda n: self.ema.ema_model.sample(batch_size=n, sample_fn=sample_fn), batches
            ))[0]

            for j in range(all_images.size(0)):
                save_image(all_images[j], f'{output_path}/{i * 200 + j + 1}.jpg')

def main():
    parser = argparse.ArgumentParser(description='Diffusion model')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--diffuser', type=str, choices=['ddpm', 'ddim'], default='ddpm')
    parser.add_argument('--load-ckpt', type=str)

    args = parser.parse_args()

    model = UNet(dim=channels, dim_mults=dim_mults)
    diffusion = GaussianDiffusion(
        model, image_size=IMG_SIZE, timesteps=timesteps, beta_schedule_fn=beta_schedule_fn
    )

    num_params = count_parameters(model)
    print(f'{num_params=}')

    trainer = Trainer(
        diffusion,
        path,
        train_batch_size = batch_size,
        train_lr = lr,
        train_num_steps = train_num_steps,
        gradient_accumulate_every = grad_steps,
        ema_decay = ema_decay,
        save_and_sample_every = 1000
    )

    # Hardcode some parameters to debug model.
    if args.debug:
        trainer = Trainer(
            diffusion, path, train_batch_size=16, train_lr=lr, train_num_steps=1,
            gradient_accumulate_every=1, ema_decay=0.995,
            save_and_sample_every=float('inf'),
        )

    if args.load_ckpt:
        trainer.load(args.load_ckpt)

    if args.train:
        trainer.train()

    if args.generate:
        simpler = {
            'ddpm': trainer.ema.ema_model.p_ddpm_sample_loop,
            'ddim': trainer.ema.ema_model.p_ddim_sample_loop
        }
        simple_fn = simpler[args.diffuser]
        trainer.inference(sample_fn=simple_fn)

    # Demonstrate model sampling procedure (Report problem 1)
    # imgs = trainer.ema.ema_model.sample(5, True)
    # imgs = imgs[:, ::50]
    # imgs = imgs.reshape(-1, *imgs.shape[2:])
    # save_image(make_grid(imgs, nrow=21), 'diffusion_demo.png')

if __name__ == "__main__":
    main()
