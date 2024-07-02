import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pca import PCA
from lda import LDA2

'''
1) Non-Face Images
load non faces images (550), convert into gray scale
and resize each image to 92x112 pixels
then flatten to a 1D array.
2) Face Images
load faces images (400) and flatten to a 1D array.
'''
def load_images(path):
    images = []
    labels = []
    #nonfaces --> change to greyscal , resize and flatten
    if "non" in path:
        for i, dir in enumerate(os.listdir(path)):
            for file in os.listdir(os.path.join(path, dir)):
                img = Image.open(os.path.join(path, dir, file)).convert('L')
                img = img.resize((92,112))
                images.append(np.array(img).flatten())
                labels.append(i+1)
            
    #faces --> flatten
    else:
        for i, dir in enumerate(os.listdir(path)):
            for file in os.listdir(os.path.join(path, dir)):
                img = Image.open(os.path.join(path, dir, file))
                images.append(np.array(img).flatten())
                labels.append(i+1)
    return np.array(images), np.array(labels).reshape(-1,1)

faces, labels = load_images('Data')
non_faces, non_labels = load_images('nonfaces')

'''
binary labels --> 1 for faces, 0 for non-faces
'''
faces_labels = np.ones((len(faces),1))
non_faces_labels = np.zeros((len(non_faces),1))
print("Faces= ",faces.shape, faces_labels.shape)
print("NonFaces= ",non_faces.shape, non_faces_labels.shape)


'''
shuffle face images and thier labels
shuffle non_faces images and thier labels
to prevent the model from learning any unaccurate patterns based on the order of the samples
'''
def shuffle_data(data, labels):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx], labels[idx]

faces, labels = shuffle_data(faces, labels)
non_faces, non_labels = shuffle_data(non_faces, non_labels)

'''
visualize a subset of face and non-face images along with their corresponding labels after shuffling
'''
def plot_data(faces, labels, n=100):
    num_rows = n // 10 #10 columns
    fig, axs = plt.subplots(num_rows, 10, figsize=(15, 1.5 * num_rows), gridspec_kw={'hspace': 0.3})
    axs = axs.ravel()
    for i in range(n):
        axs[i].imshow(faces[i].reshape((112, 92)), cmap="gray")
        axs[i].set_title(f"Label: {labels[i]}")
        axs[i].axis("off")
    plt.show()

plot_data(faces, labels,20) #show 20 images
plot_data(non_faces, non_labels,20)

'''
Split the data into training and testing sets then combine face and non-face data.
'''
# function to split the data into training and testing which alpha is the percentage of the training data
def split_data(faces, faces_labels, non_faces, non_faces_labels, non_faces_count, alpha, non_face_precentage_in_train=1):
    if alpha == 0.5:
        faces_train = faces[::2]
        faces_train_labels = faces_labels[::2]
        faces_test = faces[1::2]
        faces_test_labels = faces_labels[1::2]

        non_faces_train = non_faces[:int(non_faces_count*non_face_precentage_in_train):2]
        non_faces_train_labels = non_faces_labels[:int(non_faces_count*non_face_precentage_in_train):2]
        
        non_faces_test = non_faces[1:non_faces_count:2]
        non_faces_test_labels = non_faces_labels[1:non_faces_count:2]
    else:
        n = len(faces) #400 faces
        n_train = int(n*alpha) #no. faces in train 
        idx = np.random.permutation(n) #random index (400)
        train_idx = idx[:n_train] #select first (n_train) index
        test_idx = idx[n_train:] #select the remaining elements
        faces_train = faces[train_idx]
        faces_train_labels = faces_labels[train_idx]
        faces_test = faces[test_idx]
        faces_test_labels = faces_labels[test_idx]
        
        n = non_faces_count
        n_train = int(n*alpha) #no. nonfaces in train 
        idx = np.random.permutation(n)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        non_faces_train = non_faces[train_idx]
        non_faces_train_labels = non_faces_labels[train_idx]
        non_faces_test = non_faces[test_idx]
        non_faces_test_labels = non_faces_labels[test_idx]
    
    return np.append(faces_train, non_faces_train, axis=0), np.append(faces_train_labels, non_faces_train_labels, axis=0), np.append(faces_test, non_faces_test, axis=0), np.append(faces_test_labels, non_faces_test_labels, axis=0)

train_data, train_labels, test_data, test_labels = split_data(faces, faces_labels, non_faces, non_faces_labels,400 , 0.5, 1)
print("50% Train: ",train_data.shape, train_labels.shape)
print("50% Test: ",test_data.shape, test_labels.shape)
#shuffle train and test images with their labels
train_data, train_labels = shuffle_data( train_data,train_labels)
test_data, test_labels = shuffle_data( test_data,  test_labels)
#display 30 images of train and test data
plot_data(train_data, train_labels,30)
plot_data(test_data, test_labels,30)

'''PCA
Explored the variance explained by different components.
Transformed training and testing data using the selected number of components.
'''

mean, space, projected_data= PCA(train_data,0.85)
train_projected = (train_data - mean) @ space
test_projected = (test_data - mean) @ space

#plot eigen faces
def plot_eigenfaces(eigenvectors, n=10):
    num_rows = n // 5
    _, axs = plt.subplots(num_rows, 5, figsize=(15, 3 * num_rows), gridspec_kw={'hspace': 0.3})
    axs = axs.ravel()
    for i in range(n):
        axs[i].imshow(eigenvectors[:, i].reshape((112, 92)), cmap="gray")
        axs[i].set_title(f"EigenImages {i+1}")
        axs[i].axis("off")
    plt.show()
plot_eigenfaces(space, 10)

'''
KNN 
'''
def knn_classifier(train_data, train_labels, test_data, test_labels, k=1):
    knn = KNeighborsClassifier( n_neighbors=1, weights='distance')
    knn.fit( train_data, train_labels.ravel() )
    return accuracy_score(test_labels, knn.predict(test_data).ravel()), knn.predict(test_data).ravel()

print("Accuracy of KNN classifier with k=1:", knn_classifier(train_projected, train_labels, test_projected, test_labels, 1)[0])


# Compute the number of unique classes
num_classes = len(np.unique(train_labels))

# Determine the number of dominant eigenvectors for LDA
'''
the number of dominant eigenvectors used in LDA corresponds to the number of classes minus one,
 or the minimum between the number of classes and the dimensionality of the feature space.
'''
num_eigenvectors_lda = min(num_classes - 1, train_data.shape[1])  # Number of features can also be used instead of train_data.shape[1]

print("Number of dominant eigenvectors used for LDA:", num_eigenvectors_lda)

lda_space = LDA2(train_data, train_labels)
train_lda_projected = np.dot(train_data, lda_space) # project train
test_lda_projected = np.dot(test_data, lda_space) # project test
print("Accuracy of KNN classifier with k=1 after LDA:", knn_classifier(train_lda_projected, train_labels, test_lda_projected, test_labels)[0])

'''
Show failure and success 
'''
def plot_failure_and_success(data, labels, predictions, n=10):
    failure_idx = np.where(predictions != labels)[1]
    success_idx = np.where(predictions == labels)[1]
    num_rows = n // 5
    fig, axs = plt.subplots(num_rows, 5, figsize=(15, 3 * num_rows), gridspec_kw={'hspace': 0.3})
    axs = axs.ravel()

    for i in range(n):
        if i < n/2:
            axs[i].imshow(data[failure_idx[i]].reshape((112, 92)), cmap="gray")
            axs[i].set_title(f"Predicted: {predictions[failure_idx[i]]}, Actual: {labels[0,failure_idx[i]]} \n failure")
        else:
            axs[i].imshow(data[success_idx[i-len(failure_idx)]].reshape((112, 92)), cmap="gray")
            axs[i].set_title(f"Predicted: {predictions[success_idx[i-len(failure_idx)]]}, Actual: {labels[0,success_idx[i-len(failure_idx)]]}\nsuccess")
        axs[i].axis("off")
    plt.show()

plot_failure_and_success(test_data, test_labels.reshape(1,-1), knn_classifier(train_lda_projected, train_labels, test_lda_projected, test_labels, 1)[1], 10)   



'''
project train and test data


def project_data(data, eigenvectors, mean,):
    return np.dot(data - mean, eigenvectors)
'''

'''
 the accuracy vs the number of non-faces images while fixing the number of face images
'''
def plot_acc_vs_non_faces(algorithm,faces,faces_labels, non_faces,non_faces_labels,steps=50):
    acc = []
    n=len(non_faces)
    for i in range(steps,n+steps,steps):
        train_data, train_labels, test_data, test_labels = split_data(faces, faces_labels, non_faces, non_faces_labels,i,0.5,1)
        train_data, train_labels = shuffle_data( train_data,train_labels)
        test_data, test_labels = shuffle_data( test_data,  test_labels)
        if algorithm==0:
            mean, space, projected_data= PCA(train_data70,0.85)
            train_projected = (train_data70 - mean) @ space
            test_projected = (test_data70 - mean) @ space
            acc.append(knn_classifier(train_projected, train_labels, test_projected, test_labels, 1)[0]*100)
        else:
            lda_space = LDA2(train_data, train_labels)
            train_lda_projected = np.dot(train_data, lda_space)
            test_lda_projected = np.dot(test_data, lda_space)
            acc.append(knn_classifier(train_lda_projected, train_labels, test_lda_projected, test_labels, 1)[0]*100)

    plt.plot(range(steps,n+steps,steps),acc)
    plt.xlabel("Number of non faces")
    plt.ylabel("Accuracy")
    plt.show()

plot_acc_vs_non_faces(0,faces,faces_labels, non_faces,non_faces_labels,steps=50)


'''
Criticize the accuracy measure for large numbers of non-faces images in the training data
'''
def acc_vs_non_faces_in_training(algorithm, faces, faces_labels, non_faces, non_faces_labels, step=4):
    acc = []
    n = len(non_faces)
    steps=np.linspace(step/step**2,1,step)
    for i in steps:
        if algorithm==0:
            train_data, train_labels, test_data, test_labels = split_data(faces, faces_labels, non_faces, non_faces_labels, n, 0.5, i)
            train_data, train_labels = shuffle_data(train_data, train_labels)
            test_data, test_labels = shuffle_data(test_data, test_labels)
            mean, space, projected_data= PCA(train_data,0.85)
            train_projected = (train_data - mean) @ space
            test_projected = (test_data - mean) @ space
            acc.append(knn_classifier(train_projected, train_labels, test_projected, test_labels, 1)[0]*100)
        else:
            train_data, train_labels, test_data, test_labels = split_data(faces, faces_labels, non_faces, non_faces_labels, n, 0.5, i)
            train_data, train_labels = shuffle_data(train_data, train_labels)
            test_data, test_labels = shuffle_data(test_data, test_labels)
            lda_space = LDA2(train_data, train_labels)
            train_lda_projected = np.dot(train_data, lda_space)
            test_lda_projected = np.dot(test_data, lda_space)
            acc.append(knn_classifier(train_lda_projected, train_labels, test_lda_projected, test_labels, 1)[0]*100)
        

    # plot the points of the accuracy curve
    plt.plot(steps*(n//2), acc)
    plt.scatter(steps*(n//2), acc,marker='o',color='b')
    plt.grid()
    plt.xlabel("number of non faces in training set")
    plt.ylabel("Accuracy")
    plt.show()

acc_vs_non_faces_in_training(0,faces, faces_labels, non_faces, non_faces_labels)


'''
Split the data into 70% training and 30% test, and compare the accuracy with 50% split
'''
train_data70, train_labels70, test_data70, test_labels70 = split_data(faces, faces_labels, non_faces, non_faces_labels, 400, 0.7, 1)
train_data50, train_labels50, test_data50, test_labels50 = split_data(faces, faces_labels, non_faces, non_faces_labels, 400, 0.5, 1)

train_data70, train_labels70 = shuffle_data(train_data70, train_labels70)
test_data70, test_labels70 = shuffle_data(test_data70, test_labels70)

train_data50, train_labels50 = shuffle_data(train_data50, train_labels50)
test_data50, test_labels50 = shuffle_data(test_data50, test_labels50)


mean70, space70, projected_data70= PCA(train_data70,0.85)
train_projected70 = (train_data70 - mean) @ space
test_projected70 = (test_data70 - mean) @ space

mean50, space50, projected_data50= PCA(train_data50,0.85)
train_projected50 = (train_data50 - mean) @ space
test_projected50 = (test_data50 - mean) @ space


acc70 = knn_classifier(train_projected70, train_labels70, test_projected70, test_labels70, 1)[0]
acc50 = knn_classifier(train_projected50, train_labels50, test_projected50, test_labels50, 1)[0]

print("Accuracy of KNN classifier with k=1 after PCA with alpha=0.7:", acc70)
print("Accuracy of KNN classifier with k=1 after PCA with alpha=0.5:", acc50)
