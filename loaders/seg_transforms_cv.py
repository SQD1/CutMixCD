"""
Transformations that use an OpenCV transformation pipeline.

`SegCVTransformRandomCropScaleHung` follows the scheme used in Hung et al. [1] and Mittal et al. [2]

[1] 'Adversarial Learning for Semi-Supervised Semantic Segmentation' by Hung et al.
https://arxiv.org/abs/1802.07934

[2] 'Semi-Supervised Semantic Segmentation with High- and Low-level Consistency' by Mittal et al.
https://arxiv.org/abs/1908.05724
"""

import math

import cv2
import numpy as np
from skimage import img_as_float
from PIL import Image

# from datapipe import affine
from loaders.seg_transforms import SegTransform

# PyTorch data loaders use multi-processing.
# OpenCV uses threads that are not replicated when the process is forked,
# causing OpenCV functions to lock up, so we have to tell OpenCV not to use threads
cv2.setNumThreads(0)



class SegCVTransformTVT (SegTransform):
    """Apply a torchvision transform

    tvt_xform - the torchvision transform to apply
    apply_single - apply to single samples
    apply_pair0 - when transforming a pair of samples, apply to sample0
    apply_pair1 - when transforming a pair of samples, apply to sample1
    """
    def __init__(self, transform, apply_single=False, apply_pair0=False, apply_pair1=True):
        self.tvt_xform = transform
        self.apply_single = apply_single
        self.apply_pair0 = apply_pair0
        self.apply_pair1 = apply_pair1

    def _apply_to_image_array(self, img_arr):
        if img_arr.shape[2] == 4:
            alpha_channel = img_arr[:, :, 3:4]
        else:
            alpha_channel = None
        img_pil = Image.fromarray(img_arr[:, :, :3])
        img_pil = self.tvt_xform(img_pil)
        img_arr_rgb = np.array(img_pil)
        if alpha_channel is not None:
            return np.append(img_arr_rgb, alpha_channel, axis=2)
        else:
            return img_arr_rgb

    def transform_single(self, sample):
        if self.apply_single:
            sample = sample.copy()
            sample['image_arr'] = self._apply_to_image_array(sample['image_arr'])

        return sample

    def transform_pair(self, sample0, sample1):
        if self.apply_pair0:
            sample0 = sample0.copy()
            sample0['image_arr'] = self._apply_to_image_array(sample0['image_arr'])

        if self.apply_pair1:
            sample1 = sample1.copy()
            sample1['image_arr'] = self._apply_to_image_array(sample1['image_arr'])

        return (sample0, sample1)


class SegCVTransformNormalizeToTensor (SegTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform_single(self, sample):
        sample = sample.copy()

        # Convert image to float
        image = img_as_float(sample['image_arr'])

        if image.shape[2] == 4:
            # Has alpha channel introduced by padding
            # Split the image into RGB/alpha
            alpha_channel = image[:, :, 3:4]
            image = image[:, :, :3]

            # Account for the alpha during standardisation
            if self.mean is not None and self.std is not None:
                image = (image - (self.mean[None, None, :] * alpha_channel)) / self.std[None, None, :]
        else:
            # Standardisation
            if self.mean is not None and self.std is not None:
                image = (image - self.mean[None, None, :]) / self.std[None, None, :]

        # Convert to NCHW tensors
        assert image.shape[2] == 3
        sample['image'] = image.transpose(2, 0, 1).astype(np.float32)
        del sample['image_arr']
        if 'labels_arr' in sample:
            sample['labels'] = sample['labels_arr'][None, ...].astype(np.int64)
            del sample['labels_arr']
        if 'mask_arr' in sample:
            sample['mask'] = img_as_float(sample['mask_arr'])[None, ...].astype(np.float32)
            del sample['mask_arr']

        return sample

    def transform_pair(self, sample0, sample1):
        sample0 = sample0.copy()
        sample1 = sample1.copy()

        # Convert image to float
        image0 = img_as_float(sample0['image_arr'])
        image1 = img_as_float(sample1['image_arr'])

        if image0.shape[2] == 4:
            # Has alpha channel introduced by padding
            # Split the image into RGB/alpha
            alpha_channel0 = image0[:, :, 3:4]
            image0 = image0[:, :, :3]
            alpha_channel1 = image1[:, :, 3:4]
            image1 = image1[:, :, :3]

            # Account for the alpha during standardisation
            if self.mean is not None and self.std is not None:
                image0 = (image0 - (self.mean[None, None, :] * alpha_channel0)) / self.std[None, None, :]
                image1 = (image1 - (self.mean[None, None, :] * alpha_channel1)) / self.std[None, None, :]
        else:
            # Standardisation
            if self.mean is not None and self.std is not None:
                image0 = (image0 - self.mean[None, None, :]) / self.std[None, None, :]
                image1 = (image1 - self.mean[None, None, :]) / self.std[None, None, :]

        # Convert to NCHW tensors
        if image0.shape[2] != 3:
            raise ValueError('image0 should have 3 channels, not {}'.format(image0.shape[2]))
        if image1.shape[2] != 3:
            raise ValueError('image1 should have 3 channels, not {}'.format(image1.shape[2]))
        assert image1.shape[2] == 3
        sample0['image'] = image0.transpose(2, 0, 1).astype(np.float32)
        sample1['image'] = image1.transpose(2, 0, 1).astype(np.float32)
        del sample0['image_arr']
        del sample1['image_arr']
        if 'mask_arr' in sample0:
            sample0['mask'] = img_as_float(sample0['mask_arr'])[None, ...].astype(np.float32)
            sample1['mask'] = img_as_float(sample1['mask_arr'])[None, ...].astype(np.float32)
            del sample0['mask_arr']
            del sample1['mask_arr']

        if 'labels_arr' in sample0:
            sample0['labels'] = sample0['labels_arr'][None, ...].astype(np.int64)
            sample1['labels'] = sample1['labels_arr'][None, ...].astype(np.int64)
            del sample0['labels_arr']
            del sample1['labels_arr']

        return (sample0, sample1)