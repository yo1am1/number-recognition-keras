import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from crop import crop_images

model = keras.models.load_model("multiclass_classification_model.keras")
model.summary()

crop_images("./test_images/")

test_images = []
test_image_paths = []

for root, dirs, files in os.walk("./test_images"):
    for file in files:
        test_image_paths.append(os.path.join(root, file))

for test_image_path in test_image_paths:
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
    image_resized = cv2.bitwise_not(image_resized)
    test_image = np.array(image_resized)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image.astype("float32") / 255
    test_images.append(test_image)

predictions = []
for test_image in test_images:
    predictions.append(model.predict(test_image))

for i, (filename, pred) in enumerate(zip(test_image_paths, predictions)):
    print(
        f"File: {filename}, Predicted number: {np.argmax(pred)}, Accuracy: {np.max(pred)}"
    )

# Plot the images and predictions
fig = plt.figure(figsize=(10, 10))
for i, (test_image, pred, filename) in enumerate(zip(test_images, predictions, test_image_paths)):
    ax = fig.add_subplot(5, 5, i + 1)
    ax.imshow(test_image[0, :], cmap="gray")
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    last_char = base_filename[-1]
    ax.set_title(f"Prediction: {np.argmax(pred)}, {last_char}")
    ax.axis("off")
plt.show()
