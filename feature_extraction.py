import numpy as np
from PIL import Image

IMAGE_SIZE = (128, 128)
HIST_BINS = 8

FEATURE_NAMES = (
    ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B"]
    + [f"hist_R_{i}" for i in range(HIST_BINS)]
    + [f"hist_G_{i}" for i in range(HIST_BINS)]
    + [f"hist_B_{i}" for i in range(HIST_BINS)]
    + ["edge_mean", "edge_std"]
)


def load_image_array(image_path, size=IMAGE_SIZE):

    image = Image.open(image_path).convert("RGB").resize(size)
    return np.array(image, dtype=np.float32)  # shape: (height, width, 3)


def color_stats(img_array):
    means = img_array.mean(axis=(0, 1))   # mean of R, G, B separately
    stds = img_array.std(axis=(0, 1))     # std of R, G, B separately
    return means, stds


def color_histogram(img_array, bins=HIST_BINS):
    histograms = []
    for channel in range(3):  # R, G, B
        pixel_values = img_array[:, :, channel].flatten()
        hist, _ = np.histogram(pixel_values, bins=bins, range=(0, 256))
        hist = hist / hist.sum()  # normalize so histograms are comparable regardless of image size
        histograms.append(hist)
    return np.concatenate(histograms)  # single 1D array of length bins*3


def texture_features(img_array):
    grayscale = (
        0.2989 * img_array[:, :, 0]
        + 0.5870 * img_array[:, :, 1]
        + 0.1140 * img_array[:, :, 2]
    )

    grad_y, grad_x = np.gradient(grayscale)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return gradient_magnitude.mean(), gradient_magnitude.std()


def extract_features(image_path):
    img_array = load_image_array(image_path)

    means, stds = color_stats(img_array)
    hist = color_histogram(img_array)
    edge_mean, edge_std = texture_features(img_array)

    values = list(means) + list(stds) + list(hist) + [edge_mean, edge_std]

    return dict(zip(FEATURE_NAMES, values))
