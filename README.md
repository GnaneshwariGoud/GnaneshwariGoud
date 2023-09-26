import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from keras.applications import MobileNet
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import numpy as np

# Load MobileNet with pre-trained weights and exclude the top layer
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the trained weights
model.load_weights('plant_leaf_model.h5')

def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
    if file_path:
        img = image.load_img(file_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = preprocess_input(np.expand_dims(img, axis=0))
        prediction = model.predict(img)
        plant_prob = prediction[0][0]
        leaf_prob = prediction[0][1]
        result = f"Plant Probability: {plant_prob:.2f}\nLeaf Probability: {leaf_prob:.2f}"
        messagebox.showinfo("Prediction Result", result)

# Create a simple GUI
root = tk.Tk()
root.title("Plant vs. Leaf Classifier")

label = tk.Label(root, text="Select an image for prediction:")
label.pack(pady=10)

predict_button = tk.Button(root, text="Predict", command=predict_image)
predict_button.pack(pady=10)

root.mainloop()
