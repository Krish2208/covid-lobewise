import segmentation_models as sm
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import keras as keras
import cv2
import os


class Dataset:
    """Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['lobe-1', 'lobe-2', 'lobe-3', 'lobe-4', 'lobe-5', 'unlabelled']

    def __init__(
            self,
            image,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = '1'
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(
            cls.lower()) for cls in classes]
        # print(image)
        self.images = image
        self.mask = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        # self.mask = np.zeros((512,512,3))
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        mask = self.mask
        image = self.images
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Orignal Shapes: ", image.shape, mask.shape)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        print("Shapes after extracting masks: ", image.shape, mask.shape)
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            #print("Shape at augmentation: ",image.shape, mask.shape)
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            #print("Shape at end of augmentation: ",image.shape, mask.shape)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
