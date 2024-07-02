import numpy as np
#Principal Component Analysis
def PCA(training_DataMatrix, alpha):
    # Step 1: Compute the mean
    mean = np.mean(training_DataMatrix, axis=0)
    
    # Step 2: Center the data (m)
    training_data_centralized = training_DataMatrix - mean
    
    # Step 3: Compute the covariance matrix
    cov_matrix = training_data_centralized @ training_data_centralized.T
    
    # Step 4: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Step 5: Sort the eigenvectors descendingly by eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 6: Compute the cumulative explained variance ratio
    explained_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    
    # Step 7: Determine the number of components to keep
    no_components = np.argmax(explained_variance_ratio >= alpha) + 1
    
    # Step 8: Reduce the basis
    eigenvectors_converted = training_data_centralized.T @ eigenvectors
    eigenfaces = eigenvectors_converted / np.linalg.norm(eigenvectors_converted, axis=0)
    
    # Step 9: Reduce the dimensionality of the data
    projected_data = training_data_centralized @ eigenfaces[:, :no_components]
    
    return mean, eigenfaces[:, :no_components], projected_data