import matplotlib.pyplot as plt
import numpy as np
import skimage


def zad1():
    image = skimage.data.camera()
    rows, cols = image.shape
    print(rows, cols)

    image_fft = np.fft.fft2(image)
    image_fft = np.fft.fftshift(image_fft)

    mask = np.zeros((rows, cols), dtype=np.float64)

    k = 0.5
    rows1, cols1 = int(rows * k), int(cols * k)
    y1, x1 = (rows - rows1) // 2, (cols - cols1) // 2

    mask[y1 : y1 + rows1, x1 : x1 + cols1] = skimage.filters.window(
        "hann", (rows1, cols1)
    )

    image_fft_2 = image_fft.copy()
    image_fft_2.real *= mask
    image_fft_2.imag *= mask

    image_fft_2_shifted = np.fft.ifftshift(image_fft_2)
    image_2 = np.fft.ifft2(image_fft_2_shifted).real

    axes = plt.subplots(2, 2, figsize=(10, 10))[1]

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Oryginalny obraz")

    axes[0, 1].imshow(np.log(np.abs(image_fft + 1)), cmap="gray")
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Obraz fft")

    axes[1, 0].imshow(np.log(np.abs(image_fft_2 + 1)), cmap="gray")
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Maska")

    axes[1, 1].imshow(image_2, cmap="gray")
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Zmieniony obraz")

    plt.show()


def zad2():
    image = skimage.io.imread("gray1.jpg")

    image_fft = np.fft.fft2(image)
    image_fft = np.fft.fftshift(image_fft)

    threshold = 0.1
    maxima = np.abs(image_fft) > threshold * np.max(np.abs(image_fft))
    image_fft[maxima] = 0

    image_fft_inverted = np.fft.ifftshift(image_fft)
    image_fft_inverted = np.fft.ifft2(image_fft_inverted).real

    image_2 = (image_fft_inverted - np.min(image_fft_inverted)) / np.ptp(
        image_fft_inverted
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    axes[0].imshow(image, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Oryginalny obraz")

    axes[1].imshow(image_2, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Zmieniony obraz")

    plt.show()


zad2()
