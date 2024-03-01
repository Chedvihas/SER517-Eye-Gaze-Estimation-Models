import numpy as np

def crop_repeating_edge(image, rect):
    crop_x, crop_y, crop_w, crop_h = rect

    output = np.zeros((crop_h, crop_w, image.shape[2]), dtype=np.uint8)

    left_padding = max(0, 1 - crop_x)
    top_padding = max(0, 1 - crop_y)
    right_padding = max((crop_x + crop_w - 1) - image.shape[1], 0)
    bottom_padding = max((crop_y + crop_h - 1) - image.shape[0], 0)