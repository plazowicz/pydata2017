import cv2


def center_crop(img_arr, crop_h, crop_w):
    h, w = img_arr.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))

    return img_arr[j:j + crop_h, i:i + crop_w, :]


# Pixel from [0,255] to [-1,1]
def bound_image_values(img_arr):
    return img_arr/127.5 - 1

