import matplotlib.pyplot as plt
import numpy as np
import skimage


def make_noise(image, amount, color):
    new_image = np.copy(image)
    x = np.random.randint(0, image.shape[1], int(amount))
    y = np.random.randint(0, image.shape[0], int(amount))
    new_image[y, x] = color
    return new_image


def make_random_noise(image, amount):
    new_image = np.copy(image)
    x = np.random.randint(0, image.shape[1], int(amount))
    y = np.random.randint(0, image.shape[0], int(amount))
    random_pixels = np.random.uniform(0, 255, (image.shape[0], image.shape[1]))
    new_image[y, x, np.random.randint(0, 3)] = random_pixels[y, x]
    return new_image


def make_random_gray_noise(image, amount):
    new_image = np.copy(image)
    x = np.random.randint(0, image.shape[1], int(amount))
    y = np.random.randint(0, image.shape[0], int(amount))
    random_pixels = np.random.uniform(0, 255, (image.shape[0], image.shape[1]))
    new_image[y, x] = random_pixels[y, x]
    return new_image


def NMSE(image1, image2):
    numerator = np.sum(np.square(image1 - image2))
    denominator = np.mean(np.square(image1))
    nmse_value = numerator / denominator

    return nmse_value


def VMF(og_image, computed_image):
    fft_image = np.fft.fftshift(np.fft.fft2(og_image))
    fft_reference = np.fft.fftshift(np.fft.fft2(computed_image))

    cross_power_spectrum = fft_image * np.conj(fft_reference)

    shifted_cross_power_spectrum = cross_power_spectrum
    for _ in range(4):
        shifted_cross_power_spectrum = np.roll(
            shifted_cross_power_spectrum, shift=1, axis=(0, 1)
        )

    magnitude_cross_power = np.abs(shifted_cross_power_spectrum)

    vmf_score = np.mean(magnitude_cross_power)

    return vmf_score


def zad1():
    og_image = skimage.data.chelsea()
    percent = [0.05, 0.1, 0.2]

    _, axes = plt.subplots(2, 2, figsize=(10, 10))

    num_salt_pixels = og_image.shape[0] * og_image.shape[1]

    salt_pepper_5 = make_noise(og_image, num_salt_pixels * percent[0], (255, 255, 255))
    salt_pepper_5 = make_noise(salt_pepper_5, num_salt_pixels * percent[0], (0, 0, 0))

    salt_pepper_10 = make_noise(og_image, num_salt_pixels * percent[1], (255, 255, 255))
    salt_pepper_10 = make_noise(salt_pepper_10, num_salt_pixels * percent[1], (0, 0, 0))

    salt_pepper_20 = make_noise(og_image, num_salt_pixels * percent[2], (255, 255, 255))
    salt_pepper_20 = make_noise(salt_pepper_20, num_salt_pixels * percent[2], (0, 0, 0))

    axes[0][0].imshow(og_image)
    axes[0][0].axis("off")
    axes[0][0].set_title("Original")

    axes[0][1].imshow(salt_pepper_5)
    axes[0][1].axis("off")
    axes[0][1].set_title("Sól i pieprz 5%")

    axes[1][0].imshow(salt_pepper_10)
    axes[1][0].axis("off")
    axes[1][0].set_title("Sól i pieprz 10%")

    axes[1][1].imshow(salt_pepper_20)
    axes[1][1].axis("off")
    axes[1][1].set_title("Sól i pieprz 20%")

    plt.show()


def zad2():
    og_image = skimage.data.coins()
    _, axes = plt.subplots(1, 3, figsize=(10, 10))
    num_salt_pixels = int(og_image.shape[0] * og_image.shape[1] * 0.1)

    noise_image = make_noise(og_image, num_salt_pixels, 255)
    noise_image = make_noise(noise_image, num_salt_pixels, 0)

    filtered_image = skimage.filters.median(noise_image, skimage.morphology.square(3))

    axes[0].imshow(og_image, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Original")

    axes[1].imshow(noise_image, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Noise image")

    axes[2].imshow(filtered_image, cmap="gray")
    axes[2].axis("off")
    axes[2].set_title("Filtr")

    plt.show()


def zad3():
    og_image = skimage.data.chelsea()

    _, axes = plt.subplots(1, 2, figsize=(10, 10))

    num_salt_pixels = og_image.shape[0] * og_image.shape[1] * 0.05
    noise_image = make_random_noise(og_image, num_salt_pixels)

    axes[0].imshow(og_image)
    axes[0].axis("off")
    axes[0].set_title("Original")

    axes[1].imshow(noise_image)
    axes[1].axis("off")
    axes[1].set_title("Szum impulsowy 5%")

    plt.show()


def zad4():
    og_image = skimage.data.chelsea()

    num_salt_pixels = og_image.shape[0] * og_image.shape[1] * 0.05
    noise_image = make_random_noise(og_image, num_salt_pixels)

    nmse_value_1 = NMSE(og_image, noise_image)
    nmse_value_2 = NMSE(noise_image, og_image)

    print(nmse_value_1)
    print(nmse_value_2)


def zad5():
    og_image = skimage.data.chelsea()
    gray_image = skimage.color.rgb2gray(og_image)

    noise_percentages = [0.02, 0.05, 0.1]

    for noise_percentage in noise_percentages:
        num_salt_pixels = gray_image.shape[0] * gray_image.shape[1] * noise_percentage
        noise_gray_image = make_random_gray_noise(gray_image, num_salt_pixels)

        filtered_image = skimage.filters.median(
            noise_gray_image, skimage.morphology.disk(2)
        )

        basic_nmse = NMSE(gray_image, filtered_image)
        mean = np.mean(filtered_image * 255)
        vmf = VMF(gray_image, filtered_image)

        print(f"{noise_percentage * 100}%: NMSE = {basic_nmse}")
        print(f"{noise_percentage * 100}%: MEAN = {mean}")
        print(f"{noise_percentage * 100}%: VMF = {vmf}")
        print()


zad5()
