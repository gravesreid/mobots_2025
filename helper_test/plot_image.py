from matplotlib import pyplot as plt

image_path = "data/image_20250401_184515_960.jpg"

plt.figure()
image = plt.imread(image_path)
plt.imshow(image)
plt.show()