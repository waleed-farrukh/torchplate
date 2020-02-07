import random

import cv2
import numpy as np
import torch
import torchvision


def get_transform(transform_type: str, params):
    transform_class = {"Resize": Resize,
                       "RandomOrientation": RandomOrientation,
                       "NormalizeImage": NormalizeImage,
                       "RandomBGRtoGray": RandomBGRtoGray}[transform_type]
    return transform_class(**params)


def configure_transforms(config):
    transforms_list = []
    for i in range(0, len(config)):
        transforms_list.append(get_transform(**config[str(i)]))
    transforms_list.append(ToTensor())

    transform = torchvision.transforms.Compose(transforms_list)
    return transform


class Resize(object):
    """Resize images to a given size"""

    def __init__(self, output_size):
        """
        :param output_size: int or tuple of ints defining the desired output size
         if a tuple is given it's the resulting height and width
         if a single int is given it is both, the resulting height and width"""

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif len(output_size) == 2:
            self.output_size = (output_size, output_size)
        else:
            raise ValueError(
                f"expected 'output_size' to be int or tuple of it, got: {output_size}"
            )

    def __call__(self, sample):
        # resize image
        image = sample['image']
        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_NEAREST)
        sample['image'] = image

        # change gt corresponding to resizing if needed
        # gt = sample['gt']
        # ...
        # sample['gt'] = gt

        return sample


class RandomOrientation(object):
    """Randomly rotate the image in a sample by 0°,90°,180°, or 270° adapting the orientation."""

    def rot90(self, img, rotflag):
        """Rotate the image as specified by rotflag.
        :param img: input image
        :param rotflag: 0=None, 1=90°CW, 2=180°, 3=270°CW
        :return: rotated image
        """

        width = img.shape[1]
        height = img.shape[0]

        if rotflag == 1:
            img = cv2.transpose(img)
            img = cv2.flip(img, 1)  # transpose+flip(1)=CW
        elif rotflag == 3:
            img = cv2.transpose(img)
            img = cv2.flip(img, 0)  # transpose+flip(0)=CCW
        elif rotflag == 2:
            img = cv2.flip(img, -1)  # transpose+flip(-1)=180
        elif rotflag != 0:  # if not 0,1,2,3
            raise Exception("Unknown rotation flag({})".format(rotflag))
        return img

    def __call__(self, sample):
        image = sample['image']
        factor = np.random.randint(0, 4)
        image = self.rot90(image, factor)
        sample['image'] = image

        return sample


class NormalizeImage(object):
    """Normalize image to have zero mean"""

    def __init__(self, mean, stddev):
        if isinstance(mean, float) or isinstance(mean, int):
            mean = [mean, mean, mean]
        if isinstance(stddev, float) or isinstance(stddev, int):
            stddev = [stddev, stddev, stddev]

        self.mean = mean
        self.stddev = stddev

    def __call__(self, sample):
        image = sample['image']
        sample['image'] = (image - self.mean) / self.stddev

        return sample


class RandomBGRtoGray(object):
    def __call__(self, sample):
        if bool(random.getrandbits(1)):
            image = sample["image"]
            assert image.shape[2] == 3
            image = image.astype('uint8')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = image.astype('float64')
            sample["image"] = image

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        sample['image'] = image
        sample['gt'] = torch.tensor(sample['gt'], dtype=torch.float)

        return sample
