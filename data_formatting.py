from typing import List, Dict

from torchvision.models.detection import faster_rcnn
import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
from PIL import Image
import glob

from data_processing import SVHN_TEST_CSV, SVHN_TRAIN_CSV, SVHN_TEST_DIR, SVHN_TRAIN_DIR
# TODO: Try torch.utils.data.DataLoader


def construct_images_training_dict(file_path: str):
    """
    For TorchVision's [Faster R-CNN object detection model]
    (https://pytorch.org/docs/stable/torchvision/models.html#faster-r-cnn),
    the input structure for the model is expected to be a list of tensors, one for each image, and each of the shape
    [C, H, W], where C = the number of color channels, H = image height, and W = image width.

    For training, the model also expects a list of dictionaries alongside the list of tensors. Each dictionary should
    have
    two keys:
    - 'boxes' (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format,
                                   with values between 0 and H and 0 and W

    - 'labels' (Int64Tensor[N]): the class label for each ground-truth box
    """
    csv = pd.read_csv(file_path)
    to_tensor = transforms.ToTensor()
    file_names = csv["name"].drop_duplicates()

    image_tensors: List[torch.Tensor] = []
    image_dicts: List[Dict[str, torch.Tensor]] = []

    for file_name in file_names:
        image_path = SVHN_TRAIN_DIR + "/" + file_name
        image = Image.open(image_path)
        image_tensor = to_tensor.__call__(image)
        image.close()

        rows = csv[csv["name"] == file_name]
        n = len(rows)

        boxes = torch.zeros((n, 4), dtype=torch.float)
        labels = torch.zeros(n, dtype=torch.int64)

        for i in range(n):
            row = rows.iloc[i]
            box = torch.tensor([row["left"], row["top"], row["left"] + row["width"], row["top"] + row["height"]])
            label = row["label"]

            boxes[i] = box
            labels[i] = label

        image_dict = {
            "boxes": boxes,
            "labels": labels
        }

        image_tensors.append(image_tensor)
        image_dicts.append(image_dict)

    return image_tensors, image_dicts


if __name__ == "__main__":
    images, image_boxes = construct_images_training_dict(SVHN_TRAIN_CSV)
    print("Hello world!")
