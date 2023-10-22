import cv2
import matplotlib.pyplot as plt
import numpy as np


def zad1():
    image = cv2.imread("dog.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    max_size = max(image_gray.shape)
    min_size = min(image_gray.shape)
    image_width = image_gray.shape[1]
    image_height = image_gray.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(10, 10))

    image_rotated_90 = image_gray[::-1, :].T
    image_rotated_180 = image_gray[::-1, :]
    image_rotated_270 = image_gray.T

    x_scale = max_size / image_width
    y_scale = max_size / image_height
    x_coords = (np.arange(max_size) / x_scale).astype(int)
    y_coords = (np.arange(max_size) / y_scale).astype(int)
    image_square = image_gray[y_coords][:, x_coords]

    x_start = (image_width - min_size) // 2
    y_start = (image_height - min_size) // 2
    x_end = x_start + min_size
    y_end = y_start + min_size
    image_cut = image_gray[y_start:y_end, x_start:x_end]

    axes[0][0].imshow(image)
    axes[0][0].axis("off")

    axes[0][1].imshow(image_rotated_90, cmap="gray")
    axes[0][1].axis("off")

    axes[1][0].imshow(image_rotated_180, cmap="gray")
    axes[1][0].axis("off")

    axes[1][1].imshow(image_rotated_270, cmap="gray")
    axes[1][1].axis("off")

    axes[0][2].imshow(image_square, cmap="gray")
    axes[0][2].axis("off")

    axes[1][2].imshow(image_cut, cmap="gray")
    axes[1][2].axis("off")

    plt.show()


def zad2():
    image = cv2.imread("dog.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(2, figsize=(10, 10))

    og_width = image.shape[1]
    og_height = image.shape[0]
    new_width, new_height = 640, 480

    new_image = np.zeros((new_height, new_width), dtype=np.uint8)

    x_start = max((new_width - og_width) // 2, 0)
    x_end = x_start + min(og_width, new_width - x_start)
    y_start = max((new_height - og_height) // 2, 0)
    y_end = y_start + min(og_height, new_height - y_start)

    new_image[y_start:y_end, x_start:x_end] = image[
        : y_end - y_start, : x_end - x_start
    ]

    axes[0].imshow(image)
    axes[0].axis("off")

    axes[1].imshow(new_image)
    axes[1].axis("off")

    plt.show()


def zad3():
    cuts = int(input("Podaj ilość cięć w 1 rzędzie: "))

    image = cv2.imread("dog.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_height, image_width = image.shape[:2]
    min_size = min(image_height, image_width)

    x_start = (image_width - min_size) // 2
    y_start = (image_height - min_size) // 2
    x_end = x_start + min_size
    y_end = y_start + min_size
    image_cut = image[y_start:y_end, x_start:x_end]

    cut_size = image_cut.shape[0] // cuts

    sub_images = [
        image_cut[
            y * cut_size : (y + 1) * cut_size,
            x * cut_size : (x + 1) * cut_size,
        ]
        for y in range(cuts)
        for x in range(cuts)
    ]

    np.random.shuffle(sub_images)

    combined_image = np.vstack(
        [np.hstack(row) for row in np.array_split(sub_images, cuts)]
    )

    plt.imshow(combined_image)
    plt.axis("off")
    plt.show()


wybor = input("Które zadanie: ")
match wybor:
    case "1":
        zad1()
    case "2":
        zad2()
    case "3":
        zad3()
