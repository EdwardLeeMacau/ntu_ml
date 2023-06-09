import matplotlib.pyplot as plt
import numpy as np
import torch
from model import (ConditionalEntropy, DomainClassifier, KLDivWithLogitsLoss,
                   Model, Reconstructor, VirtualAdversarialLoss, WeightEMA)
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from utils import create_optimizer, create_scheduler, cycle


def adaptive_lambda(curr: int, total: int, k: float = 10):
    """ adaptive lambda function, return lambda in range [0, 1] given x """
    x = curr / total

    # sigmoid function with shifting and scaling
    lambda_ = (2 / (1 + np.exp(-k*x))) - 1
    return lambda_

class DomainAdaptationTrainer:
    """ Class wrapped domain transfer algorithm. """
    def __init__(self, config: dict, dataset: dict, **kwargs) -> None:
        super().__init__()

        # hyper-parameters
        self.config = config
        self.epochs = config.get('epochs', None)
        self.num_iter = config.get('iterations', None)
        self.batch_size = config["batch-size"]
        self.checkpointing = config["checkpointing"]

        # model instance initialization
        self.model = Model()
        self.reconstructor = None
        self.domain_classifier = DomainClassifier()

        # store dataset to self
        self.dataset = dataset

    @torch.no_grad()
    def visualize_cross_class(self, fname: str = 'tsne-class.png'):
        """ visualize embedded layer using t-SNE. """

        # construct dataloader
        x, y = next(iter(DataLoader(
            self.dataset["source"], batch_size=5000, shuffle=True, drop_last=True
        )))

        # target_x, _ = next(iter(DataLoader(
        #     self.dataset["target"], batch_size=5000, shuffle=False, drop_last=True
        # )))

        # move data to gpu
        device = next(self.model.parameters()).device

        # extract feature
        x = x.to(device)
        z = self.model.feature_extractor(x)
        z = z.cpu().numpy()

        # apply t-SNE
        z = TSNE(
            n_components=2, init='random', random_state=5, verbose=1
        ).fit_transform(z)

        # normalize
        z = (z - z.min(0)) / (z.max(0) - z.min(0))

        # plot by matplotlib
        plt.figure(figsize=(10, 10))
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap=plt.get_cmap('tab10'), s=1)
        plt.legend()
        plt.savefig(fname)
        plt.clf()

        # log to stdout
        print(f"Save tsne plot to {fname}")

    @torch.no_grad()
    def visualize_cross_domain(self, fname: str = 'tsne-domain.png'):
        """ visualize embedded layer using t-SNE. """

        # construct dataloader
        source_x, _ = next(iter(DataLoader(
            self.dataset["source"], batch_size=5000, shuffle=False, drop_last=True
        )))

        target_x, _ = next(iter(DataLoader(
            self.dataset["target"], batch_size=5000, shuffle=False, drop_last=True
        )))

        y = torch.zeros(10000)
        y[:5000] = 1

        # move data to gpu
        device = next(self.model.parameters()).device

        # extract feature
        x = torch.cat([source_x, target_x], dim=0)
        x = x.to(device)
        z = self.model.feature_extractor(x)
        z = z.cpu().numpy()

        # apply t-SNE
        z = TSNE(
            n_components=2, init='random', random_state=5, verbose=1
        ).fit_transform(z)

        # normalize
        z = (z - z.min(0)) / (z.max(0) - z.min(0))

        # plot by matplotlib
        plt.figure(figsize=(10, 10))
        plt.scatter(z[:5000, 0], z[:5000, 1], c='r', s=1)
        plt.scatter(z[5000:, 0], z[5000:, 1], c='b', s=1)
        plt.savefig(fname)
        plt.clf()

        # log to stdout
        print(f"Save tsne plot to {fname}")

    def parse_config(config: dict):
        """ Parse config and store to member field. """
        raise NotImplementedError

    def fit(self) -> Model:
        """ Function to be inherited. """
        raise NotImplementedError

class PretrainingTrainer(DomainAdaptationTrainer):
    """ Pretrain model with self-supervised learning. """
    def __init__(self, config: dict, dataset: dict, **kwargs) -> None:
        super().__init__(config, dataset)

    def fit(self) -> Model:
        # check cuda availability
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # create model for reconstruction
        reconstructor = Reconstructor()
        extractor = self.model.feature_extractor

        # create optimizer and scheduler
        params = list(reconstructor.parameters()) + list(extractor.parameters())
        optimizer = create_optimizer(params, **self.config["optimizer"])
        scheduler = create_scheduler(optimizer, **self.config["scheduler"])

        # move model to device
        reconstructor.to(device)
        extractor.to(device)

        # create writer
        writer = SummaryWriter()

        # create dataloader from dataset, use target dataset as unsupervised dataset here
        dataloader = DataLoader(
            self.dataset["target"], batch_size=self.batch_size, pin_memory=True, num_workers=4,
            shuffle=True, drop_last=True
        )

        # compute number of iterations for each epoch
        num_iter = self.config['iterations']['pretrain']

        # create loss function
        criterion = nn.MSELoss()

        # create training loop
        dataloader = cycle(dataloader)
        pbar = trange(num_iter, ncols=0, desc='Pretraining')
        for i in pbar:
            # prepare data
            (x, _) = next(dataloader)

            # move data to device
            x = x.to(device)

            # forward
            z = extractor(x)
            x_hat = reconstructor(z)

            # compute loss
            loss = criterion(x_hat, x)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # log to tensorboard
            metrics = { "loss": loss.item(), }
            writer.add_scalars("pretrain", metrics, i)

            # log to progress bar
            postfix = { "loss": f'{loss.item():.4f}', }
            pbar.set_postfix(postfix)

        # move model to cpu, and deprecate the reconstructor.
        reconstructor.cpu()
        extractor.cpu()

        # save model
        torch.save(reconstructor.state_dict(), 'reconstructor.pth')
        torch.save(extractor.state_dict(), 'extractor.pth')

        # store reconstructor to self
        self.reconstructor = reconstructor
        self.model.feature_extractor = extractor

        return self.model

class DomainAdversarialTrainer(DomainAdaptationTrainer):
    """ Implementation of DANN algorithm. """
    def __init__(self, config: dict, dataset: dict, **kwargs) -> None:
        super().__init__(config, dataset)

    def fit(self) -> Model:
        # check cuda availability
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # create optimizer and scheduler
        params = list(self.model.parameters()) + list(self.domain_classifier.parameters())
        optimizer = create_optimizer(params, **self.config["optimizer"])
        scheduler = create_scheduler(optimizer, **self.config["scheduler"])

        # create writer
        writer = SummaryWriter()

        # move model to device
        self.model.to(device)
        self.domain_classifier.to(device)

        # create dataloader from dataset
        src_dl = DataLoader(
            self.dataset["source"], batch_size=self.batch_size, pin_memory=True, num_workers=8,
            shuffle=True, drop_last=True
        )

        tgt_dl = DataLoader(
            self.dataset["target"], batch_size=self.batch_size, pin_memory=True, num_workers=8,
            shuffle=True, drop_last=True
        )

        dataloader = zip(src_dl, tgt_dl)

        # compute number of iterations for each epoch
        num_iter = min((len(x) for x in (src_dl, tgt_dl))) * self.epochs

        # create loss function
        class_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCEWithLogitsLoss()

        # progress to store a tsne plot
        progress = [0.1, 0.5, 0.99]
        progress = [int(x * num_iter) for x in progress]

        # create training loop
        pbar = trange(num_iter, ncols=0, desc='Dann Training')
        for i in pbar:
            # prepare data
            try:
                (src_x, src_y), (tgt_x, _) = next(dataloader)
            except StopIteration:
                dataloader = zip(src_dl, tgt_dl)
                (src_x, src_y), (tgt_x, _) = next(dataloader)

            # utility variable
            n = self.batch_size

            # hyper-parameters
            lamda_D = adaptive_lambda(i, num_iter)

            # move data to device
            src_x = src_x.to(device)
            src_y = src_y.to(device)
            tgt_x = tgt_x.to(device)

            # domain: 1 => source, 0 => target
            x = torch.cat([src_x, tgt_x], dim=0)
            domain_label = torch.zeros([2 * n, 1]).to(device)
            domain_label[:n] = 1

            # feature extraction
            z = self.model.feature_extractor(x)

            # domain classification, and image label prediction
            # note that the lamda is applied in forward function of domain classifier.
            domain_logits = self.domain_classifier(z, lamda_D)
            class_logits = self.model.label_predictor(z[:self.batch_size])

            # compute prediction loss and domain classification loss
            loss_F = class_criterion(class_logits, src_y)
            loss_D = domain_criterion(domain_logits, domain_label)

            # notes the model applied gradient reversal layer.
            # the optimize should tune the discriminator to minimize the loss_D, and tune
            # the feature extractor to maximize the loss_D.
            loss = loss_F + loss_D

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # compute accuracy (in source domain)
            source_acc = torch.sum(torch.argmax(class_logits, dim=1) == src_y).item()
            source_acc /= self.batch_size

            # log to tensorboard
            metrics = {
                "loss_F": loss_F.item(), "loss_D": loss_D.item(), "acc": source_acc,
            }
            writer.add_scalars("train", metrics, i)

            # log to progress bar
            postfix = {
                "loss_F": f'{loss_F.item():.4f}',
                "loss_D": f'{loss_D.item():.4f}',
                "acc": f'{source_acc:.2%}',
            }
            pbar.set_postfix(postfix)

        return self.model

class VirtualAdversarialDomainAdaptationTrainer(DomainAdaptationTrainer):
    """ Implementation of VADA algorithm. """
    def __init__(self, config: dict, dataset: dict, **kwargs) -> None:
        super().__init__(config, dataset)

    def fit(self) -> Model:
        # check cuda availability
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # create optimizer and scheduler
        params = list(self.model.parameters()) + list(self.domain_classifier.parameters())
        optimizer = create_optimizer(params, **self.config["optimizer"])
        scheduler = create_scheduler(optimizer, **self.config["scheduler"])

        # create writer
        writer = SummaryWriter()

        # move model to device
        self.model.to(device)
        self.domain_classifier.to(device)

        # hyper-parameters
        # lamda_D = self.config["lambda"]
        lamda_C = 0 # self.config["lambda"]
        lamda_V = 0 # self.config["lambda"]

        # create dataloader from dataset
        src_dl = DataLoader(
            self.dataset["source"], batch_size=self.batch_size, pin_memory=True, num_workers=8,
            shuffle=True, drop_last=True
        )

        tgt_dl = DataLoader(
            self.dataset["target"], batch_size=self.batch_size, pin_memory=True, num_workers=8,
            shuffle=True, drop_last=True
        )

        dataloader = zip(src_dl, tgt_dl)

        # compute number of iterations for each epoch
        num_iter = min((len(x) for x in (src_dl, tgt_dl))) * self.epochs

        # create loss function
        class_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCEWithLogitsLoss()
        conditional_criterion = ConditionalEntropy()
        virtual_adversarial_criterion = VirtualAdversarialLoss(
            model=self.model, radius=1
        )

        # create training loop
        pbar = trange(num_iter, ncols=0, desc='Training')
        for i in pbar:
            # prepare data
            try:
                (src_x, src_y), (tgt_x, _) = next(dataloader)
            except StopIteration:
                dataloader = zip(src_dl, tgt_dl)
                (src_x, src_y), (tgt_x, _) = next(dataloader)

            # utility variable
            # n: batch size for source domain
            n = self.batch_size

            # determine weight of domain adversarial loss
            lamda_D = adaptive_lambda(i, num_iter)

            # move data to device
            src_x = src_x.to(device)
            src_y = src_y.to(device)
            tgt_x = tgt_x.to(device)

            # domain: 1 => source, 0 => target
            x = torch.cat([src_x, tgt_x], dim=0)
            domain_label = torch.zeros([self.batch_size * 2, 1]).to(device)
            domain_label[:self.batch_size] = 1

            # feature extraction
            z = self.model.feature_extractor(x)

            # domain classification, and image label prediction
            # note that the lamda is applied in forward function of domain classifier.
            domain_logits = self.domain_classifier(z, lamda_D)
            class_logits = self.model.label_predictor(z)

            # compute prediction loss and domain classification loss
            # notes the model applied gradient reversal layer.
            # the optimize should tune the discriminator to minimize the loss_D, and tune
            # the feature extractor to maximize the loss_D.
            loss_F = class_criterion(class_logits[:n], src_y)
            loss_D = domain_criterion(domain_logits, domain_label)

            # compute conditional entropy loss and virtual adversarial loss
            loss_C = 0 # conditional_criterion(class_logits[n:])
            loss_V = 0 # virtual_adversarial_criterion(x, class_logits)

            loss = loss_F + loss_D + lamda_C * loss_C + lamda_V * loss_V

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # compute accuracy (in source domain)
            source_acc = torch.sum(torch.argmax(class_logits[:n], dim=1) == src_y).item()
            source_acc /= self.batch_size

            # log to tensorboard
            metrics = {
                "loss_F": loss_F.item(), "loss_D": loss_D.item(),
                # "loss_C": loss_C.item(), "loss_V": loss_V.item(),
                "acc": source_acc,
            }
            writer.add_scalars("train", metrics, i)
            writer.add_scalars("hparams", { 'lambda_D': lamda_D }, i)

            # log to progress bar
            postfix = {
                "loss_F": f'{loss_F.item():.4f}',
                "loss_D": f'{loss_D.item():.4f}',
                # "loss_C": f'{loss_C.item():.4f}',
                # "loss_V": f'{loss_V.item():.4f}',
                "acc": f'{source_acc:.2%}',
            }
            pbar.set_postfix(postfix)

            # store checkpoint

        return self.model

class DecisionBoundaryIterativeRefinementTrainer(DomainAdaptationTrainer):
    """ Implementation of DIRT-T algorithm. """
    def __init__(self, config: dict, dataset: dict, model: nn.Module, teacher: nn.Module, **kwargs) -> None:
        super().__init__(config, dataset)

        self.model = model
        self.teacher = teacher
        self.teacher.eval()

    def parse_config(config: dict):
        return super().parse_config()

    def fit(self) -> Model:
        # check cuda availability
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # put models to device
        self.model.to(device)
        self.teacher.to(device)

        # create optimizer and scheduler
        optimizer = create_optimizer(self.model.parameters(), **self.config["optimizer"])
        scheduler = create_scheduler(optimizer, **self.config["scheduler"])

        # create ema optimizer
        ema = WeightEMA(
            list(self.teacher.parameters()), list(self.model.parameters()), alpha=0.999
        )

        # create writer
        writer = SummaryWriter()

        kl_div = KLDivWithLogitsLoss()
        conditional_criterion = ConditionalEntropy()
        virtual_adversarial_criterion = VirtualAdversarialLoss(model=self.model, radius=1)

        # create dataloader from dataset
        dataloader = DataLoader(
            self.dataset["target"], batch_size=self.batch_size, pin_memory=True, num_workers=8,
            shuffle=True, drop_last=True
        )

        # compute number of iterations for each epoch
        num_iter = self.epochs * len(dataloader) if self.epochs is not None else self.num_iter
        num_ckpt = self.checkpointing * len(dataloader)

        # prepare data
        dataloader = cycle(dataloader)

        # create training loop
        pbar = trange(num_iter, ncols=0, desc='DIRT-T training')
        for i in pbar:
            # prepare data
            (x, _) = next(dataloader)

            # move data to device
            x = x.to(device)

            # feature extraction for both teacher and student model
            z_t = self.teacher.feature_extractor(x)
            z = self.model.feature_extractor(x)

            # prediction for both teacher and student model
            logits_t = self.teacher.label_predictor(z_t)
            logits = self.model.label_predictor(z)

            # compute conditional entropy loss and virtual adversarial loss
            loss_C = conditional_criterion(logits)
            loss_V = torch.tensor(0) # virtual_adversarial_criterion(x, logits)

            # compute loss to teacher model
            loss_T = kl_div(logits, logits_t)

            # backward propagation
            loss = loss_C + loss_V + loss_T

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            ema.step()

            # log to tensorboard
            metrics = {
                "loss_C": loss_C.item(), "loss_V": loss_V.item(), "loss_T": loss_T.item(),
            }
            writer.add_scalars("train", metrics, i)

            # log embedding to tensorboard
            # if i % 1000 == 0:
            #     writer.add_embedding(z_t, global_step=i, label_img=x)
            #     writer.add_embedding(z, global_step=i, label_img=x)

            # log to progress bar
            postfix = {
                "loss_C": f'{loss_C.item():.4f}',
                "loss_V": f'{loss_V.item():.4f}',
                "loss_T": f'{loss_T.item():.4f}',
            }
            pbar.set_postfix(postfix)

            # store checkpoint
            if i % num_ckpt == 0:
                torch.save(self.model.state_dict(), f'model-{i}.pth')

        return self.model
