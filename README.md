This project demonstrates the implementation of a ResNet50-based deep learning model for dual output image classification. The model is designed to classify images into two distinct categories simultaneously: CDI and Pneumonia. The model is trained and validated using custom data generators, and the implementation includes steps for preprocessing, training, validating, and testing the model.

Requirements
To run the code, you need the following libraries installed in your Python environment:

tensorflow == 1.4
keras == 2.1.5
sklearn == 0.22.2
matplotlib == 3.0.3
cv2
numpy
pandas
pathlib

you can install them using pip 

pip install tensorflow==1.4 keras==2.1.5 scikit-learn==0.22.2 matplotlib==3.0.3 opencv-python numpy pandas


Project Directory structure
├── Data
│   └── data_files
│       ├── Train.csv
│       ├── Test.csv
│   └── data_files_Pneumonia
│       ├── Train.csv
│       ├── Test.csv
├── Saved_Images
├── Saved_Model
├── Results
└── main.py





Custom Functions
The script includes several custom functions for data loading, preprocessing, and data generation:

load_samples(csv_file): Loads image filenames and labels from a CSV file.
shuffle_data(data): Shuffles the data.
preprocessing(img, label): Preprocesses the images for the CDI classification.
preprocessing2(img, label): Preprocesses the images for the Pneumonia classification.
data_generator(samples, samples2, batch_size, shuffle_data=True, resize=224): Custom data generator for the dual output model.
