import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from skimage import io, color, transform
import os

def pca_face_recognition(target_size=(50, 50)):
    # Define number of principal components to keep
    num_components = 20

    # Load images and flatten to 1D array
    images = []
    labels = []
    for person_id in range(1, 6):
        person_dir = f"data/{person_id}"
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            image = color.rgb2gray(io.imread(image_path))
            image_resized = transform.resize(image, target_size)
            images.append(image_resized.flatten())
            labels.append(person_id)

    # Convert images list to NumPy array
    X = np.array(images)

    # Subtract the mean from each image
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Compute covariance matrix and its eigenvalues/eigenvectors
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues in descending order and select top num_components
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx[:num_components]]

    # Project each image onto the subspace defined by the eigenvectors
    X_pca = np.dot(X_centered, eigenvectors)

    # Define a function to predict the label of a new image
    def predict_label(image_path):
        # Load new image and resize to target size
        image = color.rgb2gray(io.imread(image_path))
        image_resized = transform.resize(image, target_size)
        x = image_resized.flatten()

        # Subtract mean and project onto PCA subspace
        x_centered = x - X_mean
        x_pca = np.dot(x_centered, eigenvectors)

        # Compute Euclidean distance between x_pca and each image in X_pca
        distances = np.linalg.norm(X_pca - x_pca, axis=1)

        # Predict label of closest image
        closest_idx = np.argmin(distances)
        return labels[closest_idx]

    return predict_label


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_svm_with_pca(target_size=(50, 50), num_components=20, test_size=0.2, random_state=42):
    # Call the pca_face_recognition function to obtain the predict_label function
    predict_label = pca_face_recognition(target_size)

    # Prepare the data for training the SVM model
    features = []
    labels = []

    for person_id in range(1, 6):
        person_dir = f"data/{person_id}"
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            predicted_label = predict_label(image_path)
            features.append(predicted_label)
            labels.append(person_id)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)


    
    # Reshape the X_train array to 2D
    X_train = np.array(X_train).reshape(-1, 1)
    
    # Reshape the X_train array to 2D
    X_test = np.array(X_train).reshape(-1, 1)

    # Create and train the SVM classifier
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the performance of the SVM model
    # accuracy = accuracy_score(y_test, y_pred)

    return clf

def predict_with_svm(clf, predict_label, image_path):
    # Call the predict_label function from pca_face_recognition to preprocess the image
    predicted_label = predict_label(image_path)

    # Reshape the predicted_label array to have shape (1, -1)
    predicted_label = np.array(predicted_label).reshape(1, -1)

    # Make prediction using the trained SVM model
    prediction = clf.predict(predicted_label)

    return prediction



classifier = train_svm_with_pca()
# print(acc)

# Make a prediction using the trained SVM model
prediction = predict_with_svm(classifier, pca_face_recognition(), "test/25_2.jpg")

# Print the predicted label
print("Prediction:", prediction)
