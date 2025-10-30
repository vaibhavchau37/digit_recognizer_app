import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load and preprocess the MNIST dataset
def load_and_preprocess_data():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Reshape data to include channel dimension (required for CNN)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Testing data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

# Build the CNN model
def build_model():
    print("Building CNN model...")
    model = models.Sequential([
        # First convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Add dropout to prevent overfitting
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

# Train the model
def train_model(model, x_train, y_train, x_test, y_test):
    print("Training the model...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(x_test, y_test)
    )
    return history

# Evaluate the model
def evaluate_model(model, history, x_test, y_test):
    print("Evaluating model performance...")
    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Generate predictions and create confusion matrix
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    # Display some example predictions
    plt.figure(figsize=(12, 8))
    for i in range(15):
        plt.subplot(3, 5, i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        pred_label = np.argmax(y_pred[i])
        true_label = np.argmax(y_test[i])
        plt.title(f"Pred: {pred_label}, True: {true_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('example_predictions.png')
    plt.close()

# Save the model
def save_model(model):
    print("Saving model...")
    model.save('mnist_cnn_model.h5')
    print("Model saved as 'mnist_cnn_model.h5'")

# Main function
def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Evaluate model
    evaluate_model(model, history, x_test, y_test)
    
    # Save model
    save_model(model)
    
    print("Handwritten digit recognition model training complete!")

if __name__ == "__main__":
    main()