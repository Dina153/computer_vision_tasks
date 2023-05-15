# importing libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# loading images from folder
def load_images_from_folder(folder):
    training_images = []
    test_images = []
    saving_indecies = []
    for root, _ , files in os.walk(folder):
        cnt = 0 
        for file in files:
            img = cv2.imread(os.path.join(root,file))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                if (cnt < 24):
                    training_images.append(img_gray)
                else:
                    test_images.append(img_gray)
                    person = os.path.basename(file)
                    prefix = person.rpartition('.')[0]
                    prefix = prefix.rsplit("_", 1)[-1]
                    saving_indecies.append(prefix)
            cnt += 1
    print(saving_indecies)
    return saving_indecies,training_images,test_images



# get mean of images 
def get_mean_img (images):
    crop_images = []
    height, width = images[0].shape
    for image in images:
        image = image.flatten()
        crop_images.append(image)
    mean_image= np.mean(crop_images,axis=0)
    mean_image = np.reshape(mean_image, (height, width))
    return mean_image

#  subtract images from mean
def get_sub_images(mean_image ,images):
    substracted_images = []
    for image in images:
            sub_img = image - mean_image
            sub_img =  np.asarray(sub_img).flatten() 
            substracted_images.append(sub_img)

    return substracted_images


# get covariance matrix 
def get_cov_mat(substracted_images,images):
    substracted_images = np.asarray(substracted_images)
    substracted_images_Transpose= np.transpose(substracted_images)
    cov_mat = np.dot(substracted_images, substracted_images_Transpose)
    no_images = len(images)
    cov_mat = (1/(no_images -1 )) *np.asarray(cov_mat)
    return cov_mat

# get eigen vectors 
def get_eigen(cov_mat,substracted_images):
    eigen_values,eigen_vectors = np.linalg.eig(cov_mat)
    eigen_vectors = np.dot(eigen_vectors, substracted_images)
    tot_eigen_values = np.sum(eigen_values)
    accepted_variance = 0
    cnt = 0
    for eigen_val in eigen_values:
        if ( accepted_variance/ tot_eigen_values < 0.9): 
            accepted_variance += eigen_val
            cnt += 1
    eigen_vectors = eigen_vectors[:cnt , : ] 
    print ("count of eigen vectors= " , cnt)
    print ("accepted_variance= " , accepted_variance/ tot_eigen_values)
    return eigen_vectors


# get simillarity between image and eigen vector of the data 
def get_projection(eigen_vectors,images):
    images = np.asarray(images)
    images = np.transpose(images)
    projected_images = np.dot(eigen_vectors, images)
    projected_images = np.transpose(projected_images)
    print (projected_images.shape)
    return projected_images


# model for face recognition

def classify(projected_training_imgs,projected_test_imgs):
    y = []
    for i in range (5):
        for num in range (0,24):
            y.append(i)
    RFC = RandomForestClassifier( random_state=5)
    RFC.fit(projected_training_imgs, y)
    probs = RFC.predict_proba(projected_test_imgs)
    score = RFC.predict(projected_test_imgs)
    

    return RFC, probs,score
   

def get_sub_images_test(mean_image,path):
    substracted_images = []
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_img = img_gray- mean_image
    sub_img =  np.asarray(sub_img).flatten() 
    substracted_images.append(sub_img)

    return substracted_images

def show_predicted_image(RFC ,mean_image,eigen_vectors, saving_indices,path):
    test_image_path = path
    image = cv2.imread(test_image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sub_img = image_gray - mean_image
    sub_img = sub_img.flatten()
    sub_img = np.transpose(sub_img)
    projected_image = np.dot(eigen_vectors, sub_img)
    projected_image = np.transpose(projected_image)
    label = RFC.predict(projected_image.reshape(1, -1))
    print("estimated: ",saving_indices[label[0]*2])
    folder_path = "Face-Recognition-master/data/" + str(saving_indices[label[0]*2])
    print(folder_path)
    for root, _, files in os.walk(folder_path):
        for count, file in enumerate(files) :
            if(count == 0 ):
                img = cv2.imread(os.path.join(root,file))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                break

    cv2.imwrite('results1.jpg', img_gray)
    return img_gray

def plot_roc_curve(tpr, fpr, scatter = True):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    figure, axis = plt.subplots()
    if scatter:
        sns.scatterplot(x = fpr, y = tpr)
    sns.lineplot(x = fpr, y = tpr)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    st.pyplot(figure)


def get_eigen_faces(path):
    folder = "Face-Recognition-master\data" 

    saving_indices , training_images , test_images = load_images_from_folder(folder)

    mean_image = get_mean_img (training_images)
    substracted_training_images = get_sub_images(mean_image,training_images)
    substracted_test_images = get_sub_images(mean_image,test_images)

    cov_mat = get_cov_mat(substracted_training_images,training_images)
    eigen_vectors = get_eigen(cov_mat,substracted_training_images)
    projected_training_imgs = get_projection(eigen_vectors,substracted_training_images)
    projected_test_imgs = get_projection(eigen_vectors,substracted_test_images)
    RFC,y_pred_proba,score  = classify(projected_training_imgs,projected_test_imgs)
    output_image = show_predicted_image(RFC ,mean_image,eigen_vectors, saving_indices,path)
    return output_image, y_pred_proba,score

