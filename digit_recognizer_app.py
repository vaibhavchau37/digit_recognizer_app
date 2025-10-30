import tkinter as tk
from tkinter import Canvas, Button, Label, Frame
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import os

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        # Load model if exists, otherwise show message
        self.model = None
        if os.path.exists('mnist_cnn_model.h5'):
            try:
                self.model = tf.keras.models.load_model('mnist_cnn_model.h5')
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("Model file not found. Please train the model first.")
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = Label(main_frame, text="Handwritten Digit Recognizer", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Canvas for drawing
        self.canvas_frame = Frame(main_frame, bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(side=tk.LEFT, padx=10)
        
        self.canvas = Canvas(self.canvas_frame, width=280, height=280, bg="black")
        self.canvas.pack()
        
        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)
        self.setup_drawing()
        
        # Right frame for buttons and prediction
        right_frame = Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)
        
        # Prediction result
        self.result_label = Label(right_frame, text="Draw a digit", font=("Arial", 14))
        self.result_label.pack(pady=10)
        
        self.prediction_label = Label(right_frame, text="", font=("Arial", 60, "bold"))
        self.prediction_label.pack(pady=20)
        
        # Buttons
        button_frame = Frame(right_frame)
        button_frame.pack(pady=20)
        
        predict_button = Button(button_frame, text="Predict", command=self.predict, width=10, height=2)
        predict_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = Button(button_frame, text="Clear", command=self.clear_canvas, width=10, height=2)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Status message
        if self.model is None:
            status_label = Label(right_frame, text="Model not loaded. Train model first.", fg="red")
            status_label.pack(pady=10)
            train_button = Button(right_frame, text="Train Model", command=self.train_model, width=15)
            train_button.pack(pady=5)
    
    def setup_drawing(self):
        # Create a blank image for drawing
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
    def paint(self, event):
        # Draw on canvas
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", width=0)
        
        # Draw on PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill=255)
        
    def clear_canvas(self):
        # Clear canvas and image
        self.canvas.delete("all")
        self.setup_drawing()
        self.prediction_label.config(text="")
        self.result_label.config(text="Draw a digit")
        
    def predict(self):
        if self.model is None:
            self.result_label.config(text="Model not loaded", fg="red")
            return
            
        # Resize image to 28x28 (MNIST format)
        img = self.image.resize((28, 28), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0
        
        # Reshape for model input (add batch and channel dimensions)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Make prediction
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction[0])
        confidence = prediction[0][digit] * 100
        
        # Update UI
        self.prediction_label.config(text=f"{digit}")
        self.result_label.config(text=f"Prediction: {digit} (Confidence: {confidence:.2f}%)")
        
    def train_model(self):
        # Run the training script
        self.result_label.config(text="Training model... Please wait.", fg="blue")
        self.root.update()
        
        import subprocess
        try:
            subprocess.run(["python", "digit_recognizer.py"], check=True)
            self.result_label.config(text="Model trained successfully!", fg="green")
            # Try to load the model
            self.model = tf.keras.models.load_model('mnist_cnn_model.h5')
        except Exception as e:
            self.result_label.config(text=f"Error training model: {e}", fg="red")

def main():
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()