'''
Modified based on original by adding funtion of saving best weights
'''

import logging
import os
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import datetime

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        # Initialize SummaryWriter with log_dir=None to prevent event file generation
        self.writer = SummaryWriter(log_dir=None)  # Disable TensorBoard event files

        # Set up logging to a file in the output directory
        log_dir = './output/SimCLR'  # Define the log directory separately
        os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
        logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.DEBUG)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        # Initialize best accuracy tracker
        self.best_acc = 0.0

    def info_nce_loss(self, features):
        # (unchanged)
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # Save config file to the log directory
        log_dir = './output/SimCLR'  # Define the log directory separately
        save_config_file(log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            print(f'\nEpoch {epoch_counter}/{self.args.epochs}')
            running_loss = 0.0
            running_top1 = 0.0
            running_top5 = 0.0
            num_batches = 0

            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    # Log metrics to the training log instead of TensorBoard
                    logging.info(
                        f"Iteration {n_iter}\tLoss: {loss.item()}\tTop1 accuracy: {top1[0]}\tTop5 accuracy: {top5[0]}\tLearning rate: {self.scheduler.get_lr()[0]}")

                running_loss += loss.item()
                running_top1 += top1[0]
                running_top5 += top5[0]
                num_batches += 1
                n_iter += 1

            # Calculate average metrics for the epoch
            avg_loss = running_loss / num_batches
            avg_top1 = running_top1 / num_batches
            avg_top5 = running_top5 / num_batches

            # Warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            # Log epoch-level metrics
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {avg_loss}\tTop1 accuracy: {avg_top1}\tTop5 accuracy: {avg_top5}")

            # Check if the current top-1 accuracy is the best so far
            if avg_top1 > self.best_acc:
                self.best_acc = avg_top1
                logging.info(f"New best top-1 accuracy: {avg_top1}. Saving best.pth.tar...")

                # Save the best model checkpoint
                best_checkpoint_name = os.path.join(log_dir, 'simclr_checkpoint_best.pth.tar')
                save_checkpoint({
                    'epoch': epoch_counter + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_acc': self.best_acc,
                }, is_best=True, filename=best_checkpoint_name)

        logging.info("Training has finished.")

        # Save the final model checkpoint
        current_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        final_checkpoint_name = f'checkpoint_epoch{self.args.epochs:04d}_{current_time}.pth.tar'
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(log_dir, final_checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {log_dir}.")

        # Optionally, log the final best accuracy
        logging.info(f"Best top-1 accuracy achieved: {self.best_acc}")