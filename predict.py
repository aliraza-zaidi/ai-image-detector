from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


model = load_model("model.h5")

img_path = "me.jpg"
img = image.load_img(img_path, target_size=(96, 96))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
label = "Real" if pred[0][0] >= 0.5 else "Fake"
confidence = float(pred[0][0])

plt.imshow(img)
plt.axis('off')
plt.title(f"Prediction: {label} ({confidence:.2f})")
plt.show()