from torchvision.models import detection
import torch
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from data_processing import SVHN_TEST_CSV, SVHN_TRAIN_CSV, SVHN_TEST_DIR, SVHN_TRAIN_DIR
from data_formatting import construct_images_training_dict

import itertools

if __name__ == "__main__":
    train_images, train_image_boxes = construct_images_training_dict(SVHN_TRAIN_CSV, SVHN_TRAIN_DIR)

    model = detection.fasterrcnn_resnet50_fpn(num_classes=10)

    shuffle = ShuffleSplit(n_splits=10, train_size=0.5, random_state=1)
    for train_ind, test_ind in shuffle.split(train_images, train_image_boxes):
        X_train = [train_images[ind] for ind in train_ind]
        X_test = [train_images[ind] for ind in test_ind]
        y_train = [train_image_boxes[ind] for ind in train_ind]
        y_test = [train_image_boxes[ind] for ind in test_ind]

        model.train(mode=True)

        loss = model.forward(images=X_train, targets=y_train)

        model.eval()

        predictions = model.forward(images=X_test)

        for pred, actual in zip(predictions, y_test):
            total_pred = len(pred["boxes"])
            total_actual = len(actual["boxes"])

            pred_inds = list(range(total_pred))
            actual_inds = list(range(total_actual))

            for pred_ind, actual_ind in itertools.product(pred_inds, actual_inds):
                pred_box = pred["boxes"][pred_ind]
                pred_label = pred["labels"][pred_ind]
                actual_box = actual["boxes"][actual_ind]
                actual_label = actual["labels"][actual_ind]

    print("Hello world!")
