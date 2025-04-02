import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local


def initial_process():
    # Load the image
    image_path = "map_processing\Mobot Satalite - crop.png"
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = np.min(image, axis=2)

    # Apply Gaussian blur to smooth out noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)


    # Apply local thresholding
    block_size = 15  # Size of neighborhood (must be odd)
    offset = -1      # Subtracted from local mean

    local_thresh = threshold_local(blurred, block_size=block_size, offset=offset)
    local_mask = blurred > local_thresh  # Line will be True (white)


    # Convert boolean mask to uint8 (0 and 255)
    binary_mask = (local_mask * 255).astype(np.uint8)

    # Optional: Morphological cleanup
    # kernel = np.ones((3, 3), np.uint8)
    # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # save the image
    # invert the mask and save it
    invert_mask = 255 - binary_mask
    # cv2.imwrite("C:\Projects\mobots_2025\map_processing\output.png", invert_mask)

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(3, 1, 2)
    plt.title("Blurred Image")
    plt.imshow(blurred, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.title("Local Thresholding")
    plt.imshow(local_mask, cmap="gray")
    plt.axis("off")


    plt.tight_layout()
    plt.show()

from skimage.morphology import skeletonize, binary_dilation, disk, binary_closing, binary_opening
def clean_map():
    # Load image in grayscale
    image_path = "map_processing\inverted_manual_crop.png"
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img.shape[2] != 4:
        raise ValueError("Image does not have an alpha channel")

    # Use alpha channel directly to create binary mask
    alpha = img[:, :, 3]
    binary = (alpha > 0).astype(np.uint8)

    # Step 1: Close small holes/gaps
    closed = binary_closing(binary, disk(2))

    # Step 2: Remove noise/speckles (optional)

    final = closed

    # Step 3: Standardize thickness (final dilation)
    # final = binary_dilation(opened, disk(1)).astype(np.uint8) * 255

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.title("Original Image")
    plt.imshow(binary, cmap="gray")
    plt.axis("off")

    # plt.subplot(3, 1, 2)
    # plt.title("pre_dilated")
    # plt.imshow(pre_dilated, cmap="gray")
    # plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.title("Cleaned Image")
    plt.imshow(final, cmap="gray")
    plt.axis("off")
    # cv2.imwrite(output_path, cleaned)

    plt.show()

def plot_image():
    path = "map_processing/inverted_manual_crop.png"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # create a mask
    mask = img[:, :, 3] > 0

    # plot the mask:
    plt.imshow(mask, cmap="gray")
    plt.show()

def convert_alpha_to_black_with_white_strip():
    src_path = "map_processing/inverted_manual_crop.png"
    dst_path = "map_processing/final_path.png"

    # Load the image
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)

    # Create a mask
    mask = img[:, :, 3] > 0


    cv2.imwrite(dst_path, mask * 255)

if __name__ == "__main__":
    # clean_map()
    # initial_process()
    # plot_image()
    convert_alpha_to_black_with_white_strip()
