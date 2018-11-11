import cv2

def rgb2dataset(rgb_data):
    gray_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2GRAY)
    cropped = gray_data[16:240, 16:240]
    resized = cv2.resize(cropped, (84, 84))
    downsampled = resized / 255.0
    return downsampled