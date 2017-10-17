import cv2

from utils import img_ops
from utils.logger import log


@log
class CelebDataset(object):

    def __init__(self, crop_size, out_size, celeb_files_list, batch_size):
        self.crop_size = crop_size
        self.out_size = out_size
        self.celeb_files_list = celeb_files_list
        self.batch_size = batch_size

        self.logger.info("Celeb faces dataset size = %d" % len(self.celeb_files_list))

    def generate_train_mb(self):
        for i in xrange(len(self.celeb_files_list), self.batch_size):
            batch_files_list = self.celeb_files_list[i: i + self.batch_size]
            if len(batch_files_list) < self.batch_size:
                batch_files_list += self.celeb_files_list[0: self.batch_size - len(batch_files_list)]
            images_batch = [self.get_img(p) for p in batch_files_list]
            yield images_batch

    def get_img(self, img_path):
        img = cv2.imread(img_path)
        crop = img_ops.center_crop(img, self.crop_size, self.crop_size) if self.crop_size is not None else img
        resized_img = cv2.resize(crop, (self.out_size, self.out_size)) if self.out_size is not None else crop
        return img_ops.bound_image_values(resized_img)

    def img_size(self):
        return self.out_size

    def batch_size(self):
        return self.batch_size


