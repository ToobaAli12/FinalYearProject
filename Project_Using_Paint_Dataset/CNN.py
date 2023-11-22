import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import matplotlib.pyplot as plt

# ... (Rest of your code)
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

data_dir = "D:\\FinalYearProject\\Project_Using_Paint_Dataset\\Paint_Dataset\\"
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
# Define the number of folds (k)
k = 5

# Initialize lists to store MSE values for each fold
mse_scores = []

# Create a KFold object to split the data into k folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Iterate through each fold
for train_index, test_index in kf.split(data):
    X_train_fold, X_test_fold = data[train_index], data[test_index]
    y_train_fold, y_test_fold = labels[train_index], labels[test_index]

#
# Define the number of folds for k-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Create an array to store MSE values for each fold
mse_scores = []

# for train_index, test_index in kf.split(data):
#     X_train, X_test = data[train_index], data[test_index]
#     y_train, y_test = labels[train_index], labels[test_index]

# Create and compile the model inside the loop for each fold
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 5
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Evaluate the model on the test fold
loss, _ = model.evaluate(X_test, y_test, batch_size=batch_size)

# Calculate and store the MSE for this fold
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test.argmax(axis=1), y_pred.argmax(axis=1))
mse_scores.append(mse)

# Calculate the mean and standard deviation of MSE scores
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print(f"Mean squared error across {k_folds}-fold cross-validation: {mean_mse:.4f} ± {std_mse:.4f}")

# Optionally, you can plot the MSE values for each fold
plt.figure(figsize=(8, 4))
plt.bar(range(1, k_folds + 1), mse_scores)
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('MSE Across Cross-Validation Folds')
plt.show()
# Compute the MSE for this fold
y_pred_fold = model.predict(X_test_fold)
mse = mean_squared_error(y_test_fold, y_pred_fold)

mse_scores.append(mse)

# Plot the MSE values across different folds
plt.plot(range(1, k + 1), mse_scores, marker='o', linestyle='-')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error (MSE)')
plt.title(f'MSE Across {k}-Fold Cross-Validation')
plt.grid(True)
plt.show()