import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class YourDataset1(Dataset):
    """
    Dataset class for your data samples
    """

    def __init__(self, root_dir, transform=None):
        """
        :
            image_dir (string): Directory containing images used for evaluation.
            gt_dir  (string): Directory containing groundtruth annotations.
            :param root_dir: Directory containing the data.
            :param transform:
        """
        self.root_dir = root_dir
        self.transform = transform
        if self.transform:
            self._get_transformed_shape()
        self.doc_dirs_list = self._get_list_of_dirs(self.root_dir)

    def __len__(self):
        return len(self.doc_dirs_list)

    @staticmethod
    def _get_list_of_dirs(root_dir):
        doc_dirs = []
        for subdir, dirs, files in os.walk(root_dir):
            image_path = os.path.join(subdir, "file.jpg")
            gt_path = os.path.join(subdir, "gt.json")
            if os.path.exists(image_path) and os.path.exists(gt_path):
                doc_dirs.append(subdir)
        return doc_dirs

    def _get_transformed_shape(self):
        rand_image = np.zeros((500, 500, 3))
        mock_gt = torch.tensor(0.)
        mock_sample = {"image": rand_image,
                       "gt": mock_gt}
        transformed_mock_sample = self.transform(mock_sample)
        self.input_height = transformed_mock_sample["image"].shape[1]
        self.input_width = transformed_mock_sample["image"].shape[2]

    def __getitem__(self, idx):
        doc_dir = self.doc_dirs_list[idx]

        img_file = os.path.join(doc_dir, "file.jpg")
        gt_file = os.path.join(doc_dir, "gt.json")

        # read image
        image = cv2.imread(img_file)
        assert image is not None, f"Could not load image '{img_file}'"

        # read annotations
        with open(gt_file, 'r') as f:
            gt = json.load(f)[0]
        sample = {'filename': doc_dir,
                  'image': image,
                  'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample
