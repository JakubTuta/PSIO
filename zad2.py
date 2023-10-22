import matplotlib.pyplot as plt
import numpy as np
import skimage


def zad1():
    image = skimage.data.camera()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    histogram, _ = np.histogram(image, bins=256)
    bars, _ = np.histogram(image, bins=8)
    cum = np.cumsum(histogram)

    axes[0][0].imshow(image, cmap="gray")
    axes[0][0].axis("off")

    axes[0][1].plot(histogram)

    axes[1][0].bar(range(len(bars)), bars)

    axes[1][1].plot(cum)

    plt.show()


def zad2():
    image = skimage.data.camera()
    height, width = image.shape

    x_start, x_end = 50, 350
    x_center = (x_start + x_end) // 2
    y_center = 50
    x_radius = (x_end - x_start) // 2
    y_radius = x_radius // 2

    x, y = np.meshgrid(np.arange(height), np.arange(width))

    distance = np.sqrt(
        ((x - x_center) / x_radius) ** 2 + ((y - y_center) / y_radius) ** 2
    )
    ellipse_mask = distance <= 1

    image2 = image.copy()
    image2[ellipse_mask] = skimage.exposure.rescale_intensity(
        image2[ellipse_mask], in_range=(0, 255), out_range=(30, 250)
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    histogram1, _ = np.histogram(image, bins=256, range=(0, 256))
    histogram2, _ = np.histogram(image2, bins=256, range=(0, 256))

    axes[0][0].imshow(image, cmap="gray")
    axes[0][0].axis("off")
    axes[1][0].plot(histogram1)

    axes[0][1].imshow(image2, cmap="gray")
    axes[0][1].axis("off")
    axes[1][1].plot(histogram2)

    plt.show()


def zad3():
    image = skimage.data.chelsea()
    width = image.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    histogram_red, _ = np.histogram(image[:, :, 0], bins=256, range=(0, 256))
    histogram_green, _ = np.histogram(image[:, :, 1], bins=256, range=(0, 256))
    histogram_blue, _ = np.histogram(image[:, :, 2], bins=256, range=(0, 256))

    bars_red, bins_red = np.histogram(image[:, :, 0], bins=8)
    bars_green, bins_green = np.histogram(image[:, :, 1], bins=8)
    bars_blue, bins_blue = np.histogram(image[:, :, 2], bins=8)

    axes[0][0].imshow(image)
    axes[0][0].axis("off")

    axes[1][0].plot(histogram_red, "r")
    axes[1][0].plot(histogram_green, "g")
    axes[1][0].plot(histogram_blue, "b")

    x = np.arange(len(bins_red[:-1]))
    bar_width = 0.3
    axes[1][1].bar(x - bar_width, bars_red, width=bar_width, color="red")
    axes[1][1].bar(x, bars_green, width=bar_width, color="green")
    axes[1][1].bar(x + bar_width, bars_blue, width=bar_width, color="blue")

    plt.show()


def zad4():
    image = skimage.data.moon()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    histogram, _ = np.histogram(image, bins=256)

    x, a, b = np.arange(256), 80, 150
    lut = np.zeros((256,), dtype=np.uint8)
    lut[a:b] = 255 * (x[a:b] - a) // (b - a)
    lut[:a] = 0
    lut[b:] = 255

    axes[0].imshow(image, cmap="gray")
    axes[0].axis("off")

    axes[1].plot(histogram[a:b])

    axes[2].plot(lut)

    plt.show()


wybor = input("Które zadanie: ")
match wybor:
    case "1":
        zad1()
    case "2":
        zad2()
    case "3":
        zad3()
    case "4":
        zad4()
