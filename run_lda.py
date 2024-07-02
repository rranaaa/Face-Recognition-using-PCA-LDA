from lda import LDA, LDA_projected_data
from data_preprocessing import process
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


# Test The Accuracy Of LDA Using The First Nearest Neighbor (1NN)
def Test_LDA(k , LDA_projection_matrix):
    projected_X_train, projected_X_test = LDA_projected_data(X_train,X_test,LDA_projection_matrix)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(projected_X_train, y_train.ravel())
    y_pred = knn.predict(projected_X_test)
    accuracy = accuracy_score(y_test, y_pred.ravel())
    return accuracy





X_train, y_train, X_test, y_test = process()
LDA_projection_matrix = LDA(X_train,y_train)
print(LDA_projection_matrix.shape)
print("LDA Accuracy: " + str(Test_LDA(1 , LDA_projection_matrix))) # ====> 0.965

LDA_projection_matrix = LDA(X_train,y_train)
print(LDA_projection_matrix.shape)
print(LDA_projection_matrix)
print("LDA Accuracy: " + str(Test_LDA(1 , LDA_projection_matrix))) # ====> 0.965

# LDA With Tunning 
k_values = [1, 3, 5, 7, 9] # HyperParameters

# Initialize a list to store the results
results = []
projected_X_train, projected_X_test = LDA_projected_data(X_train,X_test,LDA_projection_matrix)
# Loop over the values of k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(projected_X_train, y_train.ravel())
    y_pred = knn.predict(projected_X_test)
    accuracy = accuracy_score(y_test, y_pred.ravel())
    results.append({"accuracy": accuracy})

# Convert the results to a DataFrame
df = pd.DataFrame(results, index=k_values)
df.index.name = "k"
print(df)

# So Tuning Is Not Important In This Case