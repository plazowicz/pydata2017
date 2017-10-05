import cv2
import numpy as np

from utils import img_ops


class CelebDsTransformer(object):

    def __init__(self, crop_size, out_size):
        self.crop_size = crop_size
        self.out_size = out_size

    def get_img(self, img_path):
        img = cv2.imread(img_path)
        crop = img_ops.center_crop(img, self.crop_size, self.crop_size) if self.crop_size is not None else img
        resized_img = cv2.resize(crop, (self.out_size, self.out_size)) if self.out_size is not None else crop
        return img_ops.bound_image_values(resized_img)

    def inverse_transform(self, gen_img):
        img_0_1_bounded = (gen_img + 1.)/2.
        img_0_255_bounded = (img_0_1_bounded * 255.).astype(np.uint8)
        return img_0_255_bounded

    def img_size(self):
        return self.out_size

