import os

import cv2
import numpy as np


def crop_images(path) -> None:
    for file in os.listdir(path):
        file_path = os.path.join(path, file)

        try:
            input_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            if input_image is None:
                print(f"Error reading image: {file_path}")
                continue

            height = input_image.shape[0]
            width = input_image.shape[1]

            if len(input_image.shape) == 2:
                gray_input_image = input_image.copy()
            else:
                gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

            upper_threshold, thresh_input_image = cv2.threshold(
                gray_input_image,
                thresh=0,
                maxval=255,
                type=cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            lower_threshold = 0.5 * upper_threshold

            canny = cv2.Canny(input_image, lower_threshold, upper_threshold)
            pts = np.argwhere(canny > 0)

            y1, x1 = pts.min(axis=0)
            y2, x2 = pts.max(axis=0)

            border_size = 20
            y1, x1 = max(0, y1 - border_size), max(0, x1 - border_size)
            y2, x2 = min(height, y2 + border_size), min(width, x2 + border_size)

            output_image = input_image[y1:y2, x1:x2]

            cv2.imwrite(file_path, output_image)
        except Exception as e:
            print(f"Error processing image: {file_path}")
            print(e)
