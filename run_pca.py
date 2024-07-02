import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from data_preprocessing import process
from pca import PCA

#PCA (Without Tuning)
'''
print("\nAccuracy for every value of alpha :")
# Define the alpha values (from the assig.PDF)
alpha_values = [0.8, 0.85, 0.9, 0.95]

# Store accuracy for each alpha
accuracies = []

# Perform PCA and classify for each alpha value
for alpha in alpha_values:
    # Perform PCA on the training set
    mean_face, eigenfaces, projected_data = PCA(X_train, alpha)
    
    # Project the training and test sets separately
    projected_train = (X_train - mean_face) @ eigenfaces
    projected_test = (X_test - mean_face) @ eigenfaces
    
    # Use a simple classifier (K-Nearest Neighbors) to determine the class labels
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(projected_train, y_train)
    y_pred = classifier.predict(projected_test)
    
    # Calculate accuracy for this alpha
    accuracy = np.mean(y_pred == y_test) * 100
    accuracies.append(accuracy)
    # Report accuracy for this alpha
    print(f"Alpha: {alpha}, Accuracy: {accuracy:.2f}%")
    print(eigenfaces.shape)
    
    # Plot the first 10 eigenfaces for this alpha
    fig, axs = plt.subplots(1, 10, figsize=(16, 10))
    for i in range(10):
        image_array = np.reshape(eigenfaces[:, i], (112, 92))
        axs[i].imshow(image_array, cmap="gray")
        axs[i].set_title("Eigenface " + str(i + 1))
        axs[i].axis("off")
    plt.suptitle(f"Eigenfaces for Alpha = {alpha}")
    plt.show()
    
# Plot alpha values against classification accuracy
plt.plot(alpha_values, accuracies, marker='o')
plt.title('Relation between Alpha and Classification Accuracy')
plt.xlabel('Alpha')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()

'''

X_train, y_train, X_test, y_test = process()
#PCA (With Tuning)
# Perform PCA and classify for each alpha value
print("\nAccuracy for every value of alpha and k:")
alpha_values = [0.8, 0.85, 0.9, 0.95]
k_values = [1, 3, 5, 7]
accuracies = []

for alpha in alpha_values:
    mean_face, eigenfaces, projected_data = PCA(X_train, alpha)
    projected_train = (X_train - mean_face) @ eigenfaces
    projected_test = (X_test - mean_face) @ eigenfaces
    
    for k in k_values:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(projected_train, y_train)
        y_pred = classifier.predict(projected_test)
        accuracy = np.mean(y_pred == y_test) * 100
        accuracies.append(accuracy)
        print(f"Alpha: {alpha}, K: {k}, Accuracy: {accuracy:.2f}%")
        # Plot the first 10 eigenfaces for this alpha
        fig, axs = plt.subplots(1, 10, figsize=(16, 10))
        for i in range(10):
             image_array = np.reshape(eigenfaces[:, i], (112, 92))
             axs[i].imshow(image_array, cmap="gray")
             axs[i].set_title("Eigenface " + str(i + 1))
             axs[i].axis("off")
        plt.suptitle(f"Eigenfaces for Alpha = {alpha} and k-value ={k}")
        plt.show()

# Reshape accuracies into a 2D array for easier plotting
''' 
to iterate it easily for each k[i] take row[i]
[[95.  89.5 85.  80.5]
 [95.  89.5 84.5 77.5]
 [94.  89.  83.5 77. ]
 [94.  89.5 84.5 74. ]]
'''
accuracies = np.array(accuracies).reshape(len(alpha_values), len(k_values))

# Plot the performance measure (accuracy) against K value for each alpha
for i, alpha in enumerate(alpha_values):
    plt.plot(k_values, accuracies[i], marker='o', label=f"Alpha: {alpha}")
plt.title("Performance Measure (Accuracy) vs. K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.show()

# It's the same plot but this mesure the performance (accuracy) against alpha for each K value
for i, k in enumerate(k_values):
    plt.plot(alpha_values, accuracies[:, i], marker='o', label=f"K-value: {k}")
plt.title("Performance Measure (Accuracy) vs. Alpha")
plt.xlabel("Alpha")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.show()