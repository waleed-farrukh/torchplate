import logging
import os
import timeit

import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    """
    Trainer class. This class is used to define all the training parameters and processes for training the network.
    """

    def __init__(self, net, criterion, optimizer, train_loader, valid_loader, model_output_path, lr=1e-04,
                 num_epochs=50, scheduler=None, summary_writer=SummaryWriter(),
                 device="cpu"):
        """

        :param net: your model to train (nn.Module)
        :param criterion: loss function
        :param optimizer: optimizer class
        :param train_loader: train data loader
        :param valid_loader: train data loader
        :param model_output_path: output path for the trained model
        :param lr: learning rate
        :param num_epochs: maximum number of epochs
        :param scheduler: schedular class
        :param summary_writer: SummaryWriter() to output model structure
        :param device: device to train on e.g. "cpu" or "cuda"
        """
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.input_height = self.train_loader.dataset.input_height
        self.input_width = self.train_loader.dataset.input_width
        self.device = device
        self.num_epochs = num_epochs
        self.model_output_path = model_output_path
        self.tensorboard_writer = summary_writer

    def train(self):

        self.dropout_zero_me_prob = 0
        if hasattr(self.net, 'dropout'):
            self.dropout_zero_me_prob = self.net.dropout.p

        if not os.path.exists(os.path.dirname(self.model_output_path)):
            logging.info(
                f"Output directory does not exist. Creating directory {os.path.dirname(self.model_output_path)}")
            os.makedirs(os.path.dirname(self.model_output_path))

        model_path = os.path.join(self.model_output_path,
                                  f"model_{self.num_epochs}epochs"
                                  f"_lr_{self.lr}"
                                  f"_{self.input_width}x{self.input_height}"
                                  f"_dropout_{self.dropout_zero_me_prob}"
                                  f".pth")
        self.best_model_path = model_path

        train_len = len(self.train_loader.batch_sampler)

        logging.info("Ready to start training!")
        tic = timeit.default_timer()
        best_validation_loss = 10000000

        for epoch in range(self.num_epochs):
            logging.info("Training")
            sample_size = 20  # Log loss after every 40 data samples TODO: compute from train_len
            self.net.train(True)  # Set model to training mode

            if self.scheduler:
                self.scheduler.step()

            running_loss = 0.0
            av_loss = 0.0

            for i, data in enumerate(self.train_loader):
                labels = data['gt'].to(self.device)
                label_mask = data['image_mask'].to(self.device).squeeze_(1)
                inputs = data['image'].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, label_mask)

                loss.backward()

                self.optimizer.step()

                av_loss += loss.item()
                running_loss += loss.item()
                if i % sample_size == (sample_size - 1):
                    # give feedback on command line
                    logging.info("epoch:{}, step {}, loss: {}".format(epoch + 1, i + 1, running_loss / sample_size))
                    running_loss = 0.0

            # output training loss
            av_train_loss = av_loss / train_len
            logging.info(f"epoch:{epoch + 1}, average loss: {av_train_loss}")
            self.tensorboard_writer.add_scalar('training/average_train_loss', av_train_loss, epoch + 1)

            best_validation_loss, av_valid_loss = self.validation(epoch, best_validation_loss, model_path)

            logging.info(f"epoch:{epoch + 1}, average validation loss: {av_valid_loss}")
            self.tensorboard_writer.add_scalar('training/average_valid_loss', av_valid_loss, epoch + 1)
            self.tensorboard_writer.add_scalars('training/average_losses',
                                                {'train': av_train_loss, 'valid': av_valid_loss}, epoch + 1)

        toc = timeit.default_timer()
        logging.info(f"Finished training in {toc - tic}s")

        self.tensorboard_writer.close()

        return self.best_model_path

    def validation(self, current_epoch, best_loss, model_path=None):
        logging.info("Validation")
        valid_len = len(self.valid_loader.batch_sampler)

        self.net.train(False)  # Set model to evaluation mode

        av_loss = 0.0

        for i, data in enumerate(self.valid_loader):
            labels = data['gt'].to(self.device)
            inputs = data['image'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.net(inputs)

            loss = self.criterion(outputs, labels)

            av_loss += loss.item()

        # output loss
        av_valid_loss = av_loss / valid_len

        # save best model if applicable
        if av_valid_loss < best_loss and model_path and self.best_model_path:
            best_loss = av_valid_loss
            self.best_model_path = model_path
            logging.info(
                "Better model found. Saving. Epoch {}, Path {}".format(current_epoch + 1, self.best_model_path))
            logging.info("Saving best model at epoch {} as {}".format(current_epoch + 1, self.best_model_path))
            torch.save(self.net, self.best_model_path)

        return best_loss, av_valid_loss
