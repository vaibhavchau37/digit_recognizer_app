# Handwritten Digit Recognizer

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. It includes both a training script and a simple graphical interface for testing the model with your own handwritten digits.

## Project Structure

- `digit_recognizer.py` - Main script for loading data, building, training, and evaluating the CNN model
- `digit_recognizer_app.py` - GUI application for testing the trained model with custom handwritten input
- `requirements.txt` - List of required Python packages

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Pillow (PIL)
- tkinter (for the GUI application)

## Installation

1. Clone or download this repository
2. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script to download the MNIST dataset, train the CNN model, and save it:

```
python digit_recognizer.py
```

This will:
- Download and preprocess the MNIST dataset
- Build and train the CNN model
- Evaluate the model performance
- Save the trained model as `mnist_cnn_model.h5`
- Generate performance visualizations (training history, confusion matrix, example predictions)

### Testing with the GUI

After training the model, run the GUI application to test it with your own handwritten digits:

```
python digit_recognizer_app.py
```

In the application:
1. Draw a digit (0-9) in the black canvas area
2. Click "Predict" to see the model's prediction
3. Click "Clear" to erase the canvas and try another digit

If the model hasn't been trained yet, you can click the "Train Model" button in the application to run the training process.

## Model Architecture

The CNN model consists of:
- 3 convolutional layers with ReLU activation and max pooling
- Dropout layer to prevent overfitting
- Dense output layer with softmax activation for 10-class classification

## Performance

The model typically achieves 98-99% accuracy on the MNIST test set after training for 10 epochs.

## License

This project is open source and available for educational purposes."# digit_recognizer_app" 
"# digit_recognizer_app" 
