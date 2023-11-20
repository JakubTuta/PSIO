import matplotlib.pyplot as plt
import numpy as np
import skimage


def zad1():
    image = skimage.io.imread("brain_tumor.png")
    image = skimage.color.rgb2gray(image)

    # skimage.filters.try_all_threshold(image, figsize=(10, 10), verbose=False)

    minimum_brain_threshold = skimage.filters.threshold_minimum(image)
    thresholded_brain_image = image > minimum_brain_threshold

    brain_segments, num_brain_regions = skimage.measure.label(
        thresholded_brain_image, return_num=True
    )
    brain_regions = skimage.measure.regionprops(brain_segments)

    max_brain_area = brain_regions[0].area
    max_brain_index = 0
    for index, region in enumerate(brain_regions):
        if region.area > max_brain_area:
            max_brain_area = region.area
            max_brain_index = index

    max_brain_image = np.zeros_like(image)
    y1, x1, y2, x2 = brain_regions[max_brain_index].bbox
    max_brain_image[y1:y2, x1:x2] = brain_regions[max_brain_index].image

    tumor_image = image - max_brain_image

    mean_tumor_threshold = skimage.filters.threshold_mean(tumor_image)
    thresholded_tumor_image = tumor_image > mean_tumor_threshold

    tumor_segments, num_tumor_regions = skimage.measure.label(
        thresholded_tumor_image, return_num=True
    )
    tumor_regions = skimage.measure.regionprops(tumor_segments)

    sorted_tumor_regions = sorted(tumor_regions, key=lambda item: item.area)
    y1, x1, y2, x2 = sorted_tumor_regions[-2].bbox
    max_tumor_image = np.zeros_like(image)
    max_tumor_image[y1:y2, x1:x2] = sorted_tumor_regions[-2].image
    max_tumor_area = sorted_tumor_regions[-2].area

    red_channel = max_tumor_image
    green_channel = np.zeros_like(image)
    blue_channel = max_brain_image
    color_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    _, axes = plt.subplots(2, 4, figsize=(10, 10))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original image")

    axes[0, 1].imshow(thresholded_brain_image, cmap="gray")
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Minimum brain threshold")

    axes[0, 2].imshow(brain_segments)
    axes[0, 2].axis("off")
    axes[0, 2].set_title(f"{num_brain_regions} regions")

    axes[0, 3].imshow(max_brain_image, cmap="gray")
    axes[0, 3].axis("off")
    axes[0, 3].set_title(f"Brain size: {int(max_brain_area)}")

    axes[1, 0].imshow(tumor_image, cmap="gray")
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Original tumor image")

    axes[1, 1].imshow(thresholded_tumor_image, cmap="gray")
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Minimum tumor threshold")

    axes[1, 2].imshow(tumor_segments)
    axes[1, 2].axis("off")
    axes[1, 2].set_title(f"{num_tumor_regions} regions")

    axes[1, 3].imshow(max_tumor_image, cmap="gray")
    axes[1, 3].axis("off")
    axes[1, 3].set_title(f"Tumor size: {int(max_tumor_area)}")

    plt.figure()
    plt.imshow(color_image)
    plt.title(
        f"Tumor accounts for {int(max_tumor_area / max_brain_area * 100)}% of brain vol."
    )

    plt.show()


def zad2():
    image = skimage.io.imread("bolts.jpg")

    rescaled_image = skimage.transform.resize(image, (500, 500), anti_aliasing=True)

    gray_image = skimage.color.rgb2gray(rescaled_image)

    filtered_image = skimage.filters.gaussian(gray_image, sigma=1.5)

    canny_edges = skimage.feature.canny(filtered_image)
    canny_edges = skimage.segmentation.clear_border(canny_edges)

    closed_image = skimage.morphology.closing(canny_edges, skimage.morphology.disk(5))

    segments = skimage.measure.label(closed_image)
    regions = skimage.measure.regionprops(segments)
    sorted_regions = sorted(regions, key=lambda item: item.area)
    sorted_regions.pop(0)
    sorted_regions.pop(0)

    canvas = np.zeros_like(rescaled_image)

    for region in sorted_regions:
        y1, x1, y2, x2 = region.bbox
        canvas[y1:y2, x1:x2] = region.image[..., np.newaxis] * np.random.rand(3)

    _, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(filtered_image, cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title(f"Original image (rescaled)")

    axes[0, 1].imshow(canny_edges, cmap="gray")
    axes[0, 1].axis("off")
    axes[0, 1].set_title(f"Canny edges")

    axes[1, 0].imshow(closed_image, cmap="gray")
    axes[1, 0].axis("off")
    axes[1, 0].set_title(f"Close")

    axes[1, 1].imshow(canvas)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Labeled")

    plt.show()


def zad3():
    image = skimage.io.imread("planes.png", as_gray=True)

    canny_edges = skimage.feature.canny(image)

    segments = skimage.measure.label(canny_edges)
    regions = skimage.measure.regionprops(segments)

    canvas = np.ones_like(image)

    for region in regions:
        y1, x1, y2, x2 = region.bbox

        if region.euler_number == 1:
            canvas[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    _, axes = plt.subplots(1, 2, figsize=(10, 10))

    axes[0].imshow(image, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Original")

    axes[1].imshow(canvas, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Planes only")

    plt.show()


zad3()
