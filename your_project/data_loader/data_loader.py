import logging

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def split(dataset, train=0.7, valid=0.2, test=0.1, shuffle=True, random_seed=0):
    """
    Function to split Dataset into train, test & valid DataLoader.
    :param dataset  (torch.utils.data.Dataset()): Dataset class YourDataset1
    :param train    (float): <1.0
    :param valid    (float): <1.0
    :param test     (float): <1.0
    :param shuffle  (bool): whether the output samplers will be shuffled or not.
    :param random_seed  (bool): for setting np.random.seed() for shuffling (if shuffle==True)
    :return: train, valid and test torch.utils.dataloader.DataLoader
    """
    assert int(train + valid + test) == 1, "Train, valid and test ratios do not add up to 1."
    num_data = len(dataset)
    indices = list(range(num_data))
    split_train = int(np.floor(train * num_data))
    split_valid = split_train + int(np.floor(valid * num_data))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx, test_idx = indices[:split_train], indices[split_train:split_valid], indices[split_valid:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    return train_sampler, valid_sampler, test_sampler


def get_train_valid_test_loader(dataset,
                                batch_size,
                                random_seed=0,
                                valid_size=0.1,
                                test_size=0.1,
                                shuffle=True,
                                num_workers=1,
                                pin_memory=True):
    """
    Utility function for loading and returning train, valid and test
    multi-process iterators over our dataset. If using CUDA, num_workers
    should be set to 1 and pin_memory to True.
    :param dataset: Dataset() class.
    :param batch_size: how many samples per batch to load.
    :param random_seed: fix seed for reproducibility.
    :param valid_size: percentage split of the training set used for
     the validation set. Should be a float in the range [0, 1].
    :param test_size: percentage split of the training set used for
     the test set. Should be a float in the range [0, 1]. If you
     have a separate test set in another directory, make this 0.
    :param shuffle: whether to shuffle the train/validation indices
    :param num_workers: number of subprocesses to use when loading the dataset
    :param pin_memory: hether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    :return: Tuple of (train_loader, valid_loader, test_loader) iterators
    """

    train_sampler, valid_sampler, test_sampler = split(dataset=dataset, train=1.0 - valid_size - test_size, \
                                                       valid=valid_size, test=test_size, shuffle=shuffle, \
                                                       random_seed=random_seed)

    logging.info('Loading data sets: %d training images, %d validation images, %d test images' %
                 (len(train_sampler), len(valid_sampler), len(test_sampler)))

    train_loader = DataLoader(dataset,
                              batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = DataLoader(dataset,
                              batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(dataset,
                             batch_size=batch_size, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader
