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

