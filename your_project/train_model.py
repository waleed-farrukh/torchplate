import logging

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from your_project.data_loader.data_loader import get_train_valid_test_loader
from your_project.datasets.your_dataset_1 import YourDataset1
from your_project.loss.loss_functions import set_criterion
from your_project.models.your_basic_model import YourBasicModel
from your_project.trainer.trainer import Trainer
from your_project.transforms.transforms import configure_transforms


def train_model(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensorboard_writer = SummaryWriter()

    ## instantiate the dataset
    params_dataset = config["dataset"]

    params_transforms = config["transforms"]
    transform = configure_transforms(params_transforms)
    dataset = YourDataset1(**params_dataset, transform=transform)

    ## instantiate the data_loader
    params_data_loader = config["data_loader"]
    train_loader, valid_loader, test_loader = get_train_valid_test_loader(dataset=dataset, **params_data_loader)

    ## instantiate the network
    params_net = config["model"]
    net = YourBasicModel(input_height=dataset.input_height, input_width=dataset.input_width, **params_net)
    net.apply(net.init_weights)
    net.to(device)
    summary(net, input_size=(3, dataset.input_height, dataset.input_width), batch_size=train_loader.batch_size,
            device=device.type)

    ## instantiate the loss
    params_loss = config["loss"]
    assert params_loss["loss_type"], "No Loss type set. Choose a loss function type from the documentation"
    loss_type = params_loss["loss_type"]
    del params_loss["loss_type"]
    criterion = set_criterion(loss_type=loss_type, loss_args=params_loss)

    ## instantiate the Trainer
    # instantiate the optimizer
    params_trainer = config["trainer"]

    optimizer = optim.Adam(net.parameters(), lr=params_trainer['lr'], weight_decay=params_trainer['weight_decay'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    trainer = Trainer(net=net, criterion=criterion, optimizer=optimizer, train_loader=train_loader, scheduler=scheduler, \
                      valid_loader=valid_loader, device=device, summary_writer=tensorboard_writer, **params_trainer)

    for key, value in config.items():
        tensorboard_writer.add_text(key, str(value), 0)
        logging.info("param key:{}, value:{}".format(key, str(value)))

    saved_model_path = trainer.train()

    ## check if saved model works
    # sanity check: load saved model
    test_device = torch.device("cpu")
    logging.info("Loading full model from {}".format(saved_model_path))
    net_test = torch.load(saved_model_path)
    net_test.to(test_device)
    net_test.eval()

    # quick sanity check for sample input
    logging.info("Sanity check")
    data_iter = iter(test_loader)
    sample = data_iter.next()
    test_labels = sample['gt']
    test_images = sample['image']

    test_outputs = net_test(test_images.to(test_device))

    logging.info("Sanity check labels: {}".format(test_labels))
    logging.info("Sanity check outputs: {}".format(test_outputs))

    # trace trained model
    traced_model_file = saved_model_path.replace(".pth", "_traced_cpu.pth")
    logging.info("Tracing model and saving result to " + traced_model_file)

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, dataset.input_height, dataset.input_width).to(test_device)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(net_test, example)
    traced_script_module.save(traced_model_file)
