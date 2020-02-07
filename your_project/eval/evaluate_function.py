import logging
import os

import torch

from your_project.data_loader.data_loader import get_train_valid_test_loader
from your_project.datasets.your_dataset_1 import YourDataset1
from your_project.loss.loss_functions import set_criterion
from your_project.transforms.transforms import configure_transforms


def evaluate_model(config):
    device = torch.device("cpu")

    report_output_path = config["report"]["report_output_path"]
    if not os.path.exists(os.path.dirname(report_output_path)):
        logging.info(
            f"Output directory does not exist. Creating directory {os.path.dirname(report_output_path)}")
        os.makedirs(os.path.dirname(report_output_path))

    model_config = config["model"]
    model_name = model_config["model_name"]
    net = torch.load(model_name, map_location=lambda storage, loc: storage)
    net.eval()

    ## instantiate the dataset
    params_transforms = config["transforms"]
    transform = configure_transforms(params_transforms)

    params_dataset = config["dataset"]
    dataset = YourDataset1(**params_dataset, transform=transform)

    params_data_loader = config["data_loader"]
    train_loader, valid_loader, test_loader = get_train_valid_test_loader(dataset=dataset, **params_data_loader)
    test_len = len(test_loader.batch_sampler)
    logging.info(f"Loaded Test set of {test_len} images")

    params_loss = config["loss"]
    loss_type = params_loss["loss_type"]
    del params_loss["loss_type"]
    params_loss["device"] = device
    criterion = set_criterion(loss_type=loss_type, loss_args=params_loss)

    logging.info(f"Evaluating model {model_name}")
    logging.info(f"Results will be written to {report_output_path}")

    logging.info("Ready to start evaluating!")


    for i, data in enumerate(test_loader):
        filename = data['filename']
        img = data['image']
        gt = data['gt']
        your_outputs = net(img)
        loss = criterion(your_outputs, gt)

        #You can do your customized evaluation here
