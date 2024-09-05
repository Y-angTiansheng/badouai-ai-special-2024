import cv2
import numpy as np
def grayscale(image):
    if len(image.shape) == 3:
        gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        gray_image = gray_image.astype(np.uint8)
        return gray_image
    else:
        raise ValueError("输入图像非法")
def binarize(image, threshold = 128):
    gray_image = grayscale(image)
    return (gray_image > threshold).astype(np.uint8) * 255
image = cv2.imread('text001.jpg')
gray_image = grayscale(image)
binary_image = binarize(image)
cv2.imshow('Original Image', image)
cv2.imshow('Custom Gray Image', gray_image)
cv2.imshow('Binarized Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('text001_grayscale.jpg', gray_image)
cv2.imwrite('text001_binarized.jpg', gray_image)
