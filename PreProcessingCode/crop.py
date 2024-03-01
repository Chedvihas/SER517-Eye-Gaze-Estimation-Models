import numpy as np

def crop_repeating_edge(image, rect):
    crop_x, crop_y, crop_w, crop_h = rect

    output = np.zeros((crop_h, crop_w, image.shape[2]), dtype=np.uint8)

    left_padding = max(0, 1 - crop_x)
    top_padding = max(0, 1 - crop_y)
    right_padding = max((crop_x + crop_w - 1) - image.shape[1], 0)
    bottom_padding = max((crop_y + crop_h - 1) - image.shape[0], 0)
    
    content_out_pixels_y = slice(0 + top_padding, crop_h - bottom_padding)
    content_out_pixels_x = slice(0 + left_padding, crop_w - right_padding)
    content_in_pixels_y = slice(crop_y + top_padding, crop_y + crop_h - bottom_padding)
    content_in_pixels_x = slice(crop_x + left_padding, crop_x + crop_w - right_padding)

    output[content_out_pixels_y, content_out_pixels_x, :] = \
        image[content_in_pixels_y, content_in_pixels_x, :]
    
    if len(content_out_pixels_x) == 0 or len(content_out_pixels_y) == 0:
        print('No out pixels in x or y direction.')
        output = np.nan
        return output
    
    output[:top_padding, content_out_pixels_x, :] = \
        np.tile(output[content_out_pixels_y.start, content_out_pixels_x, :],
                (top_padding, 1, 1))
    output[-bottom_padding:, content_out_pixels_x, :] = \
        np.tile(output[content_out_pixels_y.stop - 1, content_out_pixels_x, :],
                (bottom_padding, 1, 1))

    output[:, :left_padding, :] = \
        np.tile(output[:, content_out_pixels_x.start, :],
                (1, left_padding, 1))
    output[:, -right_padding:, :] = \
        np.tile(output[:, content_out_pixels_x.stop - 1, :],
                (1, right_padding, 1))

    return output
        
        
import imageio
import os

frame = imageio.imread("/Volumes/Extreme SSD/517-small-data-model/data/00002/frames/00000.jpg")
print(frame, frame.shape)
import matplotlib.pyplot as plt

plt.imshow(frame)
plt.axis('off')  
plt.show()