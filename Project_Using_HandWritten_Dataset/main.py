# Author: Wajahat Riaz
# License: Apache-2.0
# Github Link: https://github.com/WajahatRiaz/HandwrittenUrduCharacterRecognition
# Import classifiers and performance metrics
# "C:\ProgramData\Anaconda3\python.exe" D:\Tooba___\src\main1.py
# conda run -n base python D:\Tooba___\src\main1.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import det_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

# Standard scientific Python imports
import matplotlib.image as im
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

PIXELS = 80  # Macro defining number of pixels
DIMENSIONS = PIXELS * PIXELS  # Defining a resolution for the sample images
TRAINING_SIZE = 0.30  # By convention 30% of dataset will be used for testing

dataset_alif = []  # List declared to store image vectors of alif
dataset_bay = []  # List declared to store image vectors of bay
dataset_jeem = []  # List declared to store image vectors of jeem
dataset_daal = []  # List declared to store image vectors of daal
dataset_dhaal = []  # List declared to store image vectors of dhaal
dataset_cheey = []  # List declared to store image vectors of cheey
dataset_raa = []  # List declared to store image vectors of raa
dataset_zaa = []  # List declared to store image vectors of zaa
dataset_taa = []  # List declared to store image vectors of taa
dataset_saa = []  # List declared to store image vectors of saa

images_of_alif = 0  # To keep track of number of alif images read from folder
images_of_bay = 0  # To keep track of number of bay images read from folder
images_of_jeem = 0  # To keep track of number of jeem images read from folder
images_of_daal = 0  # To keep track of number of daal images read from folder
images_of_dhaal = 0  # To keep track of number of dhaal images read from folder
images_of_cheey = 0  # To keep track of number of cheey images read from folder
images_of_raa = 0  # To keep track of number of raa images read from folder
images_of_zaa = 0  # To keep track of number of zaa images read from folder
images_of_taa = 0  # To keep track of number of taa images read from folder
images_of_saa = 0  # To keep track of number of saa images read from folder

# reading images of alif from the folder
for filename in os.listdir("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Alif\\"):

    img = cv2.imread("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Alif\\" + filename, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Alif\\" + filename, img)

    img_50x50 = cv2.resize(img, (PIXELS, PIXELS))
    img_instance = img_50x50.flatten()

    if DIMENSIONS != img_instance.size:
        print("image pixel error")

    dataset_alif.append(img_instance)
    images_of_alif = images_of_alif + 1

# Generating matrix from list
data1 = np.empty([images_of_alif, DIMENSIONS], dtype=list)
for i in range(images_of_alif):
    data1[i] = dataset_alif[i]

# reading images of bay from the folder
for filename in os.listdir("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Bay\\"):

    img = cv2.imread("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Bay\\" + filename, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Bay\\" + filename, img)

    img_50x50 = cv2.resize(img, (PIXELS, PIXELS))
    img_instance = img_50x50.flatten()

    if DIMENSIONS != img_instance.size:
        print("image pixel error")

    dataset_bay.append(img_instance)
    images_of_bay = images_of_bay + 1

# Generating matrix from list
data2 = np.empty([images_of_bay, DIMENSIONS], dtype=list)
for i in range(images_of_bay):
    data2[i] = dataset_bay[i]

# reading images of jeem from the folder
for filename in os.listdir("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Jeem\\"):

    img = cv2.imread("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Jeem\\" + filename, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Jeem\\" + filename, img)

    img_50x50 = cv2.resize(img, (PIXELS, PIXELS))
    img_instance = img_50x50.flatten()

    if DIMENSIONS != img_instance.size:
        print("image pixel error")

    dataset_jeem.append(img_instance)
    images_of_jeem = images_of_jeem + 1

# Generating matrix from list
data3 = np.empty([images_of_jeem, DIMENSIONS], dtype=list)
for i in range(images_of_jeem):
    data3[i] = dataset_jeem[i]

# reading images of daal from the folder
for filename in os.listdir("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Daal\\"):

    img = cv2.imread("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Daal\\" + filename, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Daal\\" + filename, img)

    img_50x50 = cv2.resize(img, (PIXELS, PIXELS))
    img_instance = img_50x50.flatten()

    if DIMENSIONS != img_instance.size:
        print("image pixel error")

    dataset_daal.append(img_instance)
    images_of_daal = images_of_daal + 1

# Generating matrix from list
data4 = np.empty([images_of_daal, DIMENSIONS], dtype=list)
for i in range(images_of_daal):
    data4[i] = dataset_daal[i]

# reading images of daal from the folder
for filename in os.listdir("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Dhaal\\"):

    img = cv2.imread("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Dhaal\\" + filename,
                     cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Dhaal\\" + filename, img)

    img_50x50 = cv2.resize(img, (PIXELS, PIXELS))
    img_instance = img_50x50.flatten()

    if DIMENSIONS != img_instance.size:
        print("image pixel error")

    dataset_dhaal.append(img_instance)
    images_of_dhaal = images_of_dhaal + 1

# Generating matrix from list
data5 = np.empty([images_of_dhaal, DIMENSIONS], dtype=list)
for i in range(images_of_dhaal):
    data5[i] = dataset_dhaal[i]

# reading images of daal from the folder
for filename in os.listdir("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Cheey\\"):

    img = cv2.imread("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Cheey\\" + filename,
                     cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Cheey\\" + filename, img)

    img_50x50 = cv2.resize(img, (PIXELS, PIXELS))
    img_instance = img_50x50.flatten()

    if DIMENSIONS != img_instance.size:
        print("image pixel error")

    dataset_cheey.append(img_instance)
    images_of_cheey = images_of_cheey + 1

# Generating matrix from list
data6 = np.empty([images_of_cheey, DIMENSIONS], dtype=list)
for i in range(images_of_cheey):
    data6[i] = dataset_cheey[i]


# reading images of daal from the folder
for filename in os.listdir("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Raa\\"):

    img = cv2.imread("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Raa\\" + filename,
                     cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Raa\\" + filename, img)

    img_50x50 = cv2.resize(img, (PIXELS, PIXELS))
    img_instance = img_50x50.flatten()

    if DIMENSIONS != img_instance.size:
        print("image pixel error")

    dataset_raa.append(img_instance)
    images_of_raa = images_of_raa + 1

# Generating matrix from list
data7 = np.empty([images_of_raa, DIMENSIONS], dtype=list)
for i in range(images_of_raa):
    data7[i] = dataset_raa[i]


# reading images of Zaa from the folder
for filename in os.listdir("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Zaa\\"):

    img = cv2.imread("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Zaa\\" + filename,
                     cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Zaa\\" + filename, img)

    img_50x50 = cv2.resize(img, (PIXELS, PIXELS))
    img_instance = img_50x50.flatten()

    if DIMENSIONS != img_instance.size:
        print("image pixel error")

    dataset_zaa.append(img_instance)
    images_of_zaa = images_of_zaa + 1

# Generating matrix from list
data8 = np.empty([images_of_zaa, DIMENSIONS], dtype=list)
for i in range(images_of_zaa):
    data8[i] = dataset_zaa[i]


# reading images of Taa from the folder
for filename in os.listdir("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Taa\\"):

    img = cv2.imread("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Taa\\" + filename,
                     cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Taa\\" + filename, img)

    img_50x50 = cv2.resize(img, (PIXELS, PIXELS))
    img_instance = img_50x50.flatten()

    if DIMENSIONS != img_instance.size:
        print("image pixel error")

    dataset_taa.append(img_instance)
    images_of_taa = images_of_taa + 1

# Generating matrix from list
data9 = np.empty([images_of_taa, DIMENSIONS], dtype=list)
for i in range(images_of_taa):
    data9[i] = dataset_taa[i]


# reading images of Saa from the folder
for filename in os.listdir("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Saa\\"):

    img = cv2.imread("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Saa\\" + filename,
                     cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Hand_Written_Dataset\\Saa\\" + filename, img)

    img_50x50 = cv2.resize(img, (PIXELS, PIXELS))
    img_instance = img_50x50.flatten()

    if DIMENSIONS != img_instance.size:
        print("image pixel error")

    dataset_saa.append(img_instance)
    images_of_saa = images_of_saa + 1

# Generating matrix from list
data10 = np.empty([images_of_saa, DIMENSIONS], dtype=list)
for i in range(images_of_saa):
    data10[i] = dataset_saa[i]

# Determining total number of image instances
instances = images_of_alif + images_of_bay + images_of_jeem + images_of_daal + images_of_dhaal + images_of_cheey + images_of_raa + images_of_zaa + images_of_taa + images_of_saa
# print("Total Instances in the data set:", instances)
print("Total Features or Dimension of data set:", DIMENSIONS)

# Stacking the individual matrices
x = np.concatenate((data1, data2, data3, data4, data5, data6, data7, data8, data9, data10))

# Generating the data matrix
print("My X matrix of order", x.shape, "is given as follows: ", x)

# Generating tags for the instances
tag_alif = np.full((images_of_alif, 1), 1, dtype=int)
tag_bay = np.full((images_of_bay, 1), 2, dtype=int)
tag_jeem = np.full((images_of_jeem, 1), 3, dtype=int)
tag_daal = np.full((images_of_daal, 1), 4, dtype=int)
tag_dhaal = np.full((images_of_dhaal, 1), 5, dtype=int)
tag_cheey = np.full((images_of_cheey, 1), 6, dtype=int)
tag_raa = np.full((images_of_raa, 1), 7, dtype=int)
tag_zaa = np.full((images_of_zaa, 1), 8, dtype=int)
tag_taa = np.full((images_of_taa, 1), 9, dtype=int)
tag_saa = np.full((images_of_saa, 1), 10, dtype=int)

# Generating the tag vector
tag_vector = np.concatenate((tag_alif, tag_bay, tag_jeem, tag_daal, tag_dhaal, tag_cheey, tag_raa, tag_zaa, tag_taa, tag_saa))
print("My tags are:", tag_vector)

# Converting vector to 1D array
y = np.ravel(tag_vector, order='A')

# Splitting data for testing and training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TRAINING_SIZE, random_state=5)

print("shape of x_train", X_train.shape)
# s1 = cross_val_score(RandomForestClassifier(n_estimators=500), X_train, y_train, cv=7)
# print("Cross validation score of RandomForest = ", s1)

model_1 = RandomForestClassifier(n_estimators=500)
model_1.fit(X_train, y_train)
predictions = model_1.predict(X_test)
print("Accuracy score of Random Forest Classifier:", accuracy_score(y_test, predictions))
rf_probs = model_1.predict_proba(X_test)
rf_auc = roc_auc_score(y_test, rf_probs, multi_class='ovr')
print("Random Forest Classifier: AUROC = %.3f" % (rf_auc))
print(
    f"Classification report for Random Forest Classifier {model_1}:\n"
    f"{classification_report(y_test, predictions)}\n"
)
disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
disp.figure_.suptitle("Confusion matrix for Random Forest Classifier")
print(f"Confusion matrix for Random Forest Classifier:\n{disp.confusion_matrix}")

Model_1_report = classification_report(y_test, predictions, output_dict=True)
model_1_df = pd.DataFrame(Model_1_report).transpose()
model_1_df.to_excel("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Results\\confusion results\\random_forest.xlsx")
plt.savefig('D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Results\\confusion matrix\\random_fores.jpg')
plt.show()
# #######


model_2 = LinearSVC(max_iter=1500, multi_class='ovr')
model_2.fit(X_train, y_train)
predictions = model_2.predict(X_test)

svc_probs = model_2._predict_proba_lr(X_test)
svc_auc = roc_auc_score(y_test, svc_probs, multi_class='ovr')
print("SVM Classifier: AUROC = %.3f" % (svc_auc))

print("Accuracy score of SVM Classifier:", accuracy_score(y_test, predictions))
print(
    f"Classification report for SVM Classifier {model_2}:\n"
    f"{classification_report(y_test, predictions)}\n"
)

disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
disp.figure_.suptitle("Confusion matrix for SVM Classifier")
print(f"Confusion matrix for SVM Classifier:\n{disp.confusion_matrix}")

Model_2_report = classification_report(y_test, predictions, output_dict=True)
model_2_df = pd.DataFrame(Model_2_report).transpose()
model_2_df.to_excel("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Results\\confusion results\\svm.xlsx")
plt.savefig('D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Results\\confusion matrix\\svm.jpg')
plt.show()


scalar = StandardScaler()
X_train_scalar = scalar.fit_transform(X_train)
X_test_scalar = scalar.transform(X_test)


s1 = cross_val_score(LogisticRegression(max_iter=500), X_train, y_train, cv=7)
print("Cross validation score of RandomForest = ", s1)
model_3 = LogisticRegression(solver='newton-cg', multi_class="ovr", max_iter=15)
model_3.fit(X_train_scalar, y_train)
predictions = model_3.predict(X_test_scalar)

lr_probs = model_3.predict_proba(X_test_scalar)
lr_auc = roc_auc_score(y_test, lr_probs, multi_class='ovr')
print("Logistic Regression: AUROC = %.3f" % (lr_auc))
print("Accuracy score of Logistic Regression:", accuracy_score(y_test, predictions))
print(
    f"Classification report for Logistic Regression {model_3}:\n"
    f"{classification_report(y_test, predictions)}\n"
)

disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
disp.figure_.suptitle("Confusion matrix for Logistic Regression")
print(f"Confusion matrix for Logistic Regression:\n{disp.confusion_matrix}")

Model_3_report = classification_report(y_test, predictions, output_dict=True)
model_3_df = pd.DataFrame(Model_3_report).transpose()
model_3_df.to_excel("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Results\\confusion results\\logregression.xlsx")
plt.savefig('D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Results\\confusion matrix\\logregression.jpg')
plt.show()
#######

model_4 = DecisionTreeClassifier(criterion='entropy', splitter='best')
model_4.fit(X_train, y_train)
predictions = model_4.predict(X_test)

e_dt_probs = model_4.predict_proba(X_test)
e_dt_auc = roc_auc_score(y_test, e_dt_probs, multi_class='ovr')
print("Entropy based Decision Tree Classifier: AUROC = %.3f" % (e_dt_auc))

print("Accuracy score of Entropy based Decision Tree Classifier:", accuracy_score(y_test, predictions))
print(
    f"Classification report for Entropy based Decision Tree Classifier {model_4}:\n"
    f"{classification_report(y_test, predictions)}\n"
)
disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
disp.figure_.suptitle("Confusion matrix for Entropy based Decision Tree Classifier")
print(f"Confusion matrix for Entropy based Decision Tree Classifier:\n{disp.confusion_matrix}")

Model_4_report = classification_report(y_test, predictions, output_dict=True)
model_4_df = pd.DataFrame(Model_4_report).transpose()
model_4_df.to_excel("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Results\\confusion results\\decisiontree.xlsx")
plt.savefig('D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Results\\confusion matrix\\decisiontree.jpg')
plt.show()
#######

scalar = StandardScaler()
X_train_scalar = scalar.fit_transform(X_train)
X_test_scalar = scalar.transform(X_test)

model_5 = SGDClassifier(loss='modified_huber')
model_5.fit(X_train_scalar, y_train)
predictions = model_5.predict(X_test_scalar)

sgdc_probs = model_5.predict_proba(X_test)
sgdc_auc = roc_auc_score(y_test, sgdc_probs, multi_class='ovr')
print("Stochastic Gradient Descent Classifier: AUROC = %.3f" % (sgdc_auc))

print("Accuracy score of Stochastic Gradient Descent Classifier", accuracy_score(y_test, predictions))
print(
    f"Classification report for Stochastic Gradient Descent Classifier {model_5}:\n"
    f"{classification_report(y_test, predictions)}\n"
)

disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
disp.figure_.suptitle("Confusion matrix for Stochastic Gradient Descent Classifier")
print(f"Confusion matrix for Stochastic Gradient Descent Classifier:\n{disp.confusion_matrix}")

Model_5_report = classification_report(y_test, predictions, output_dict=True)
model_5_df = pd.DataFrame(Model_5_report).transpose()
model_5_df.to_excel("D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Results\\confusion results\\sgdc.xlsx")
plt.savefig('D:\\FinalYearProject\\Project_Using_HandWritten_Dataset\\Results\\confusion matrix\\sgdc.jpg')
plt.show()
#######

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# ... (Rest of your code remains the same)

# Define a dictionary to store MSE values for each model
mse_values = {}

# Model 1: Random Forest
model_1 = RandomForestClassifier(n_estimators=500)
model_1.fit(X_train, y_train)
predictions = model_1.predict(X_test)
mse_values["Random Forest"] = mean_squared_error(y_test, predictions)

# Model 2: SVM Classifier
model_2 = LinearSVC(max_iter=1500, multi_class='ovr')
model_2.fit(X_train, y_train)
predictions = model_2.predict(X_test)
mse_values["SVM Classifier"] = mean_squared_error(y_test, predictions)

# Model 3: Logistic Regression
model_3 = LogisticRegression(solver='newton-cg', multi_class="ovr", max_iter=15)
model_3.fit(X_train_scalar, y_train)
predictions = model_3.predict(X_test_scalar)
mse_values["Logistic Regression"] = mean_squared_error(y_test, predictions)

# Model 4: Decision Tree Classifier
model_4 = DecisionTreeClassifier(criterion='entropy', splitter='best')
model_4.fit(X_train, y_train)
predictions = model_4.predict(X_test)
mse_values["Decision Tree"] = mean_squared_error(y_test, predictions)

# Model 5: Stochastic Gradient Descent Classifier
model_5 = SGDClassifier(loss='modified_huber')
model_5.fit(X_train_scalar, y_train)
predictions = model_5.predict(X_test_scalar)
mse_values["SGD Classifier"] = mean_squared_error(y_test, predictions)

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Calculate mean squared error for each model
mse_rf = mean_squared_error(y_train, model_1.predict(X_train))
mse_svc = mean_squared_error(y_train, model_2.predict(X_train))
mse_lr = mean_squared_error(y_train, model_3.predict(X_train_scalar))
mse_dt = mean_squared_error(y_train, model_4.predict(X_train))
mse_sgdc = mean_squared_error(y_train, model_5.predict(X_train_scalar))

# Create lists to hold MSE values and model names
mse_values = [mse_rf, mse_svc, mse_lr, mse_dt, mse_sgdc]
model_names = ['Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree', 'SGD Classifier']

# Create a line graph to visualize MSE values
plt.figure(figsize=(10, 6))
plt.plot(model_names, mse_values, marker='o', linestyle='-', color='b')
plt.title('Mean Squared Error (MSE) for Different Models on Handwritten dataset')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
# #