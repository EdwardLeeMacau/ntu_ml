import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from accelerate import Accelerator
from torch.optim import Adam
import torchvision
import yaml
from torchvision import transforms as T, utils
from tqdm.auto import tqdm
from ema_pytorch import EMA
import os
from dataset import MyDataset
from model import UNet, GaussianDiffusion, exists
from torch.utils.tensorboard import SummaryWriter

with open('params.yaml') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

torch.backends.cudnn.benchmark = True
torch.manual_seed(4096)

if torch.cuda.is_available():
    torch.cuda.manual_seed(4096)

# model = Unet(64)
path = '../../../dataset/crypko'
IMG_SIZE = 64             # Size of images, do not change this if you do not know why you need to change
batch_size = 256
train_num_steps = 10000   # total training steps
lr = 1e-3
grad_steps = 1            # gradient accumulation steps, the equivalent batch size for updating equals to batch_size * grad_steps = 16 * 1
ema_decay = 0.995         # exponential moving average decay

channels = 16             # Numbers of channels of the first layer of CNN
dim_mults = (1, 2, 4)     # The model size will be (channels, 2 * channels, 4 * channels, 4 * channels, 2 * channels, channels)

timesteps = 1000          # Number of steps (adding noise)
beta_schedule = 'linear'



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

    def load(self, ckpt):
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
        accelerator.init_trackers("diffusion", config={})

        device = accelerator.device

        pbar = tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process, ncols=0)
        while self.step < self.train_num_steps:
            total_loss = 0.

            for _ in range(self.gradient_accumulate_every):
                data = next(self.dl).to(device)

                with self.accelerator.autocast():
                    loss = self.model(data)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                self.accelerator.backward(loss)

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
    def inference(self, num=1000, n_iter=5, output_path='./submission'):
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for i in range(n_iter):
            batches = self.num_to_groups(num // n_iter, 200)
            all_images = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))[0]
            for j in range(all_images.size(0)):
                torchvision.utils.save_image(all_images[j], f'{output_path}/{i * 200 + j + 1}.jpg')

def train():
    model = UNet(dim=channels, dim_mults=dim_mults)
    diffusion = GaussianDiffusion(model, image_size = IMG_SIZE, timesteps = timesteps, beta_schedule = beta_schedule)

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

    trainer.train()
    trainer.inference()

if __name__ == "__main__":
    train()
