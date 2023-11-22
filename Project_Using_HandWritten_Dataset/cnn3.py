import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
# conda run -n base python D:\Tooba___\ekAurTry6.py
# ... (Rest of your code up to creating and training the model)
def load_dataset(data_dir, img_size):
    data = []
    labels = []

    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label_name)

    data = np.array(data)
    data = data.reshape(data.shape[0], img_size, img_size, 1)
    labels = np.array(labels)

    return data, labels

data_dir = "D:\FinalYearProject\Project_Using_HandWritten_Dataset\Hand_Written_Dataset\\"
image_size = 80

data, labels = load_dataset(data_dir, image_size)

# Normalize pixel values to the range [0, 1]
data = data.astype('float32') / 255.0

# Use label encoding to convert string labels to integer labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Convert integer labels to one-hot encoding
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 30
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# Create a confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

confusion_mtx = confusion_matrix(y_true, y_pred_classes)


# Display the confusion matrix using matplotlib
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.tight_layout()


class_names = np.unique(y_true)
plot_confusion_matrix(confusion_mtx, classes=class_names)

# You can also print a classification report for more details
class_report = classification_report(y_true, y_pred_classes)
print("Classification Report:")
print(class_report)

plt.show()  # Show the confusion matrix plot
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Plot Mean Squared Error (MSE) values
train_mse = history.history['loss']
test_mse = history.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_mse, label='Training MSE')
# plt.plot(range(1, epochs + 1), test_mse, label='Test MSE')
plt.title('Mean Squared Error (MSE) during Training of cnn model using handwritten dataset:')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid()
plt.show()



