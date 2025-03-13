#with library
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Sample 2x2 matrix (features) and labels
X = np.array([[2, 3], [3, 4]])  # Features (2 samples, 2 features)
y = np.array([0, 1])  # Class labels

# Create LDA model
lda = LinearDiscriminantAnalysis()

# Fit the model
lda.fit(X, y)

# Transform data to 1D LDA space
X_lda = lda.transform(X)

# Print transformed data
print("LDA Transformed Data:", X_lda)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#without libraries
# Sample 2x2 data
X = [[2, 3], [3, 4]]  # Features
y = [0, 1]  # Labels (Two classes)

# Calculating mean and scatter matrix manually
cls_0 = [X[i] for i in range(len(y)) if y[i] == 0]
cls_1 = [X[i] for i in range(len(y)) if y[i] == 1]

# Compute mean for each feature in each class
mean_0 = [sum(col) / len(col) for col in zip(*cls_0)]
mean_1 = [sum(col) / len(col) for col in zip(*cls_1)]

# Calculating within-class scatter matrix
S_w = [[0, 0], [0, 0]]  # Initialize 2x2 scatter matrix

for sample in cls_0:
    diff = [sample[i] - mean_0[i] for i in range(2)]
    S_w = [[S_w[i][j] + diff[i] * diff[j] for j in range(2)] for i in range(2)]

for sample in cls_1:
    diff = [sample[i] - mean_1[i] for i in range(2)]
    S_w = [[S_w[i][j] + diff[i] * diff[j] for j in range(2)] for i in range(2)]

# Step 3: Compute LDA direction (Eigenvector of S_w inverse * (mean_1 - mean_0))
mean_diff = [[mean_1[i] - mean_0[i]] for i in range(2)]  # Column vector

# Inverse of 2x2 matrix (Manual Calculation)
det_Sw = S_w[0][0] * S_w[1][1] - S_w[0][1] * S_w[1][0]  # Determinant
S_w_inv = [[S_w[1][1] / det_Sw, -S_w[0][1] / det_Sw], 
           [-S_w[1][0] / det_Sw, S_w[0][0] / det_Sw]]  # Inverse

# Compute LDA direction (w = S_w_inv * mean_diff)
w = [sum(S_w_inv[i][j] * mean_diff[j][0] for j in range(2)) for i in range(2)]

# Step 4: Project data onto LDA direction
X_lda = [sum(X[i][j] * w[j] for j in range(2)) for i in range(len(X))]

# Print results
print("LDA Direction (w):", w)
print("LDA Transformed Data:", X_lda)
