import cv2
import numpy as np
import matplotlib.pyplot as plt

def keep_largest_contiguous_area(binary_mask):
    """
    Find the largest contiguous area in a binary mask and return a new mask
    containing only that area.
    
    Parameters:
    -----------
    binary_mask : numpy.ndarray
        Binary mask (0s and 1s or 0s and 255s)
        
    Returns:
    --------
    numpy.ndarray
        A binary mask with only the largest contiguous area
    """
    # Ensure the mask is binary and has the right type
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # If mask contains 1s instead of 255s, convert to 255s
    if np.max(binary_mask) == 1:
        binary_mask = binary_mask * 255
    
    # Find all contiguous regions (connected components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Skip label 0, which is the background
    if num_labels == 1:
        # No foreground components found
        return np.zeros_like(binary_mask)
    
    # Find the largest component by area (excluding background at index 0)
    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]
    
    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i
    
    # Create a new mask containing only the largest component
    largest_mask = np.zeros_like(binary_mask)
    largest_mask[labels == largest_label] = 255
    
    return largest_mask

def thresh_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply a gaussian blur to the image

    image = cv2.medianBlur(image, 7)

    # use median blurring
    very_blurred = cv2.medianBlur(image, 21)

    very_very_blurred = cv2.medianBlur(image, 251)

    # take the element-wise maximum of the blurred images
    combo = cv2.min(very_blurred, very_very_blurred)

    very_blurred = combo

    diff = np.float32(image)/np.float32(very_blurred)

    diff = np.uint8(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX))

    # apply otsu thresholding
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )


    # Assuming you already have a binary image called 'mask'
    # Create a kernel (structuring element)
    kernel_size = 3  # Adjust based on your needs
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # first, close the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    congiguous_mask = keep_largest_contiguous_area(mask)

    return congiguous_mask

if __name__ == "__main__":
    path = "/home/aigeorge/projects/mobots_2025/data/old_images/image_20250401_175437_432.jpg"

    # load image and convert to lab color space
    image = cv2.imread(path)



    # plot the results (original image, colorful image, mask)

    # create a 3x1 plot
    plt.figure(figsize=(12, 6))
    plt.subplot(5, 1, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(5, 1, 2)
    plt.title("very blurred Image")
    plt.imshow(very_blurred, cmap="gray")
    plt.axis("off")

    plt.subplot(5, 1, 3)
    plt.title("Difference")
    plt.imshow(diff, cmap="gray")
    plt.axis("off")

    plt.subplot(5, 1, 4)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(5, 1, 5)
    plt.title("Opened Mask")
    plt.imshow(congiguous_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()

    plt.show()