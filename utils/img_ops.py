import numpy as np


def center_crop(img_arr, crop_h, crop_w):
    h, w = img_arr.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))

    return img_arr[j:j + crop_h, i:i + crop_w, :]


# Pixel from [0,255] to [-1,1]
def bound_image_values(img_arr):
    return img_arr / 127.5 - 1


def unbound_image_values(gen_img):
    img_0_1_bounded = (gen_img + 1.) / 2.
    img_0_255_bounded = (img_0_1_bounded * 255.).astype(np.uint8)
    return img_0_255_bounded


def merge_images_to_grid(images, grid_size):
    h, w = images.shape[1], images.shape[2]
    big_img = np.zeros((h * grid_size[0], w * grid_size[1], 3), dtype=np.uint8)
    for idx, image in enumerate(images):  # idx=0,1,2,...,63
        i = idx % grid_size[1]  # column number
        j = idx // grid_size[1]  # row number
        big_img[j * h:j * h + h, i * w:i * w + w, :] = image
    return big_img
