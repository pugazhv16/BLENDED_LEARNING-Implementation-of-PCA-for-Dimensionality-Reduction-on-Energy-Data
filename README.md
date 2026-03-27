# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Data Import the dataset to begin the dimensionality reduction process.

2. Explore the Data Perform an initial analysis to understand data characteristics, distributions, and potential patterns.

3. Preprocess the Data (Feature Scaling) Scale features to ensure consistency, preparing the data for principal component analysis (PCA).

4. Apply PCA for Dimensionality Reduction Use PCA to reduce the dataset’s dimensionality while retaining the most significant features.

5. Analyze Explained Variance Assess the variance explained by each principal component to determine the effectiveness of dimensionality reduction.

6. Visualize Principal Components Create visualizations of the principal components to interpret patterns and clusters in reduced dimensions.

## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: PUGAZH V
RegisterNumber:  212225240109
*/
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('HeightsWeights.csv')

print("First 5 rows of the dataset:")
print(data.head())

X = data[['Height(Inches)', 'Weight(Pounds)']]

plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)', y='Weight(Pounds)', data=data)
plt.title('Original Data Distribution')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

## Output:

<img width="758" height="620" alt="Screenshot 2026-03-27 140936" src="https://github.com/user-attachments/assets/07b61dc4-9e29-4f72-b148-1aece4ecf694" />

<img width="589" height="522" alt="Screenshot 2026-03-27 140949" src="https://github.com/user-attachments/assets/e8f33421-c927-46c5-91dc-70d143d12f9a" />


## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
