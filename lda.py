import numpy as np

"""

     class_means
     For Each Class I Will Compute Its Mean
     Example:
         The First Class (Class A)
         ClassA Contains 10 Images (Each Image Represent A Vector Of 10304 Value)
         n = 10304
         Image1 Image2 Image3 ................. Image 10
            1     1      1                         1
            2     2      2                         2
            3     3      3                         3
            .     .      .                         .
            .     .      .                         .
            .     .      .                         .
            n     n      n                         n
        So Let's Compute Mean For Each Label
        x1Bar = sum(x1)/n 
        x2Bar = sum(x2)/n
        .
        .
        .
        x10304Bar = sum(x10304)/n
        
        classA Vector 
        [ sum(x1)/n   ]
        [ sum(x2)/n   ]
        [    ..       ]
        [    ..       ]
        [    ..       ]
        [    ..       ]
        [sum(x10304)/n]
"""
"""
Goal: 
The projected matrix obtained from LDA represents
a transformation of the original data into a lower-dimensional 
subspace that maximizes the separation between classes.

the scatter between classes is maximized while the scatter within classes is minimized

"""

"""
class_means = np.array([np.mean(X_train[y_train == i], axis=0) for i in range(1, 41)])

This Line Of Code Equivalent To This Code (Just For Simplicity)

def calculate_mean_vector(training_data , labels):
      classes = np.unique(labels)
      class_means = np.zeros((len(classes), training_data.shape[1]))
      for i, class_label in enumerate(classes):
        class_means[i] = np.mean(training_data[labels == class_label], axis=0) # axis=0 To Sum Row By Row
      return class_means
"""
def LDA(X_train, y_train):
    y_train = np.squeeze(y_train)
    class_means = np.array([np.mean(X_train[y_train == i], axis=0) for i in range(1, 41)])
    class_sizes = np.array([np.sum(y_train == i) for i in range(1, 41)])

    # Compute overall mean
    # We Need To Compute Overall Mean To 
    # Centering the Data (class_means[i(current_class) - 1(because it 0 index)] - overall_mean)
    # Centering The Data Helping For Removing Any Bias Or Shift In The Data Distribution
    overall_mean = np.mean(X_train, axis=0)

    # Compute within-class scatter matrix
    # Make A Zero Matrix Of (Num Of Features * 2) = S_W
    # Calculate The Covaraince Of Each Class 
    # Sum All The Covariances = S_W
    S_W = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(1, 41):
        # Use boolean index to select rows from X_train
        class_data = X_train[y_train == i]
        centered_data = class_data - class_means[i - 1]
        S_W += np.dot(centered_data.T, centered_data) 

    # Regularize S_W
    S_W += 1e-7 * np.identity(X_train.shape[1])

    # Compute between-class scatter matrix
    
    S_B = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(1, 41):
        # Use boolean index to select rows from X_train
        # Select All Vector Images In The Current Class
        class_data = X_train[y_train == i]
        # Subtracted From Overall_mean
        class_diff = class_means[i - 1] - overall_mean
        S_B += class_sizes[i - 1] * np.outer(class_diff, class_diff)

    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(S_W), S_B))

    # Sort Eigenvectors Based On Sorted Eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Take Only The First39 Dominant eigenvectors [In The PDF He Want The First Dominant Eigenvectors]
    projection_matrix = sorted_eigenvectors[:, :39]
    return np.real(projection_matrix)

# Split The projected_data Data Into Training And Testing Data
# We Need To Split The Data To Training And Testing Data To Test It Using (1NN)
def LDA_projected_data(training_data,test_data,projection_matrix):
    projected_X_train = np.dot(training_data, projection_matrix)
    projected_X_test = np.dot(test_data, projection_matrix)
    return projected_X_train, projected_X_test

def LDA2 (train_data, train_labels, k=1):
    # mean of each class
    mean1 = np.mean(train_data[train_labels.ravel() == 1], axis=0)
    mean0 = np.mean(train_data[train_labels.ravel() == 0], axis=0)

    # within class scatter matrix
    Sw = np.dot((train_data[train_labels.ravel() == 1] - mean1).T, 
                (train_data[train_labels.ravel() == 1] - mean1)) + np.dot((train_data[train_labels.ravel() == 0] - mean0).T, 
                                                                          (train_data[train_labels.ravel() == 0] - mean0))
    # between class scatter matrix
    Sb = np.dot((mean1 - mean0).reshape(-1,1), (mean1 - mean0).reshape(-1,1).T)

    # calculate eigenvalues and eigenvectors
    eig_values, eig_vectors = np.linalg.eigh(np.dot(np.linalg.inv(Sw), Sb))
    eig_values = np.real( eig_values)
    eig_vectors = np.real( eig_vectors)
    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:,idx]
    return eig_vectors[:,:k]

