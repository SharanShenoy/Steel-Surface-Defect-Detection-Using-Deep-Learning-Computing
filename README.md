🧠 Surface Defect Detection — Machine Learning & Deep Learning Approaches

This project demonstrates two approaches for detecting surface defects on steel or grinding surfaces:

🧰 Classical Machine Learning using HOG features and a Decision Tree classifier.

🤖 Deep Learning using a pre-trained ResNet50 model with transfer learning.

📂 Project Structure
project/
│── data/
│   ├── good/                    # Images without defects
│   ├── grinding/                # Images with defects
│   └── NEU-DET/                 # Deep learning dataset (train/validation)
│
│── models/
│   ├── hog_defect_detection.py
│   └── resnet50_defect_classification.py
│
│── results/
│   ├── metrics.txt
│   ├── confusion_matrix.png
│   └── training_curves.png
│
│── README.md

🧰 1. HOG + Decision Tree Classifier
📄 File

hog_defect_detection.py

📌 Goal

Classify images as “Defect” or “No Defect” using hand-crafted features.

✨ Features

Uses Histogram of Oriented Gradients (HOG) to extract features.

Classifies using a Decision Tree.

Prints classification report and confusion matrix.

Supports prediction on new unseen images.

⚡ Requirements
pip install opencv-python scikit-image scikit-learn numpy

🚀 How to Run
python models/hog_defect_detection.py


Make sure to update the paths in the script:

defect_folder = r"D:\grinding"
no_defect_folder = r"D:\good"
result = predict_defect(model, r"D:\New folder\ng.png")

📊 Output

Model accuracy and metrics in the terminal.

Predicted class (Defect or No Defect) for test images.

🤖 2. ResNet50 + Transfer Learning
📄 File

resnet50_defect_classification.py

📌 Goal

Build a robust classifier using pre-trained ResNet50 on a larger dataset with data augmentation.

✨ Features

Leverages ResNet50 pretrained on ImageNet.

Applies data augmentation for robustness.

Trains for a few epochs to fine-tune the top layers.

Displays accuracy and loss curves.

Outputs classification metrics and confusion matrix.

⚡ Requirements
pip install tensorflow scikit-learn matplotlib

🚀 How to Run
python models/resnet50_defect_classification.py


Update dataset paths in the script:

train_dir = r'C:\...\NEU-DET\train\images'
validation_dir = r'C:\...\NEU-DET\validation\images'

📊 Output

Accuracy & loss plots saved or displayed.

Confusion matrix and classification report in the terminal.

Trained model ready to be saved or deployed.

📝 Notes

For better accuracy, ensure your dataset has balanced classes.

For deep learning, consider increasing epochs and using GPU if available.

You can adapt resnet50_defect_classification.py to multi-class classification easily by adjusting:

class_mode='categorical'

loss='categorical_crossentropy'

Dense(num_classes, activation='softmax')

📌 Future Improvements

✅ Add model saving & loading functionality (model.save() / model.load_model()).

📈 Hyperparameter tuning for both models.

🧪 Deployment with Flask or FastAPI for real-time inference.

🧼 Automated data preprocessing pipeline.

👨‍💻 Author

Sharan Shenoy
AI & Software Engineer | Surface Defect Detection | Computer Vision

