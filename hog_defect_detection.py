import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_images_from_folders(defect_folder, no_defect_folder):
    images = []
    labels = []
    
    for filename in os.listdir(defect_folder):
        img = cv2.imread(os.path.join(defect_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
                          hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            images.append(hog_features)
            labels.append(0)  

    for filename in os.listdir(no_defect_folder):
        img = cv2.imread(os.path.join(no_defect_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            images.append(hog_features)
            labels.append(1)  
    
    return np.array(images), np.array(labels)

defect_folder = r"D:\grinding"
no_defect_folder = r"D:\good"


images, labels = load_images_from_folders(defect_folder, no_defect_folder)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

def predict_defect(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    hog_features = np.expand_dims(hog_features, axis=0)
    prediction = model.predict(hog_features)
    return 'Defect' if prediction == 0 else 'No Defect'

result = predict_defect(model, r'D:\New folder\ng.png')
print(result)
