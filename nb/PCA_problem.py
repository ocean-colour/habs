# Practice Problem 1 - PCA Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.covariance import empirical_covariance

# Load in the data
spectra_red = np.load('/Users/kassidylam/UCSC/research/Ocean_Color/summer25/PCA_practice/spectra_red.npy')
wavelength_grid_red = np.load('/Users/kassidylam/UCSC/research/Ocean_Color/summer25/PCA_practice/wavelength_grid_red.npy')

# Check shape of data
#print("Shape of spectra_red: ", spectra_red.shape)    #(50000 = galaxy spectra, 2676 = corresponding unique wavelength)
#print("Shape of wavelength_grid_red: ", wavelength_grid_red.shape)    #(2676, )

# Choosing a random galaxy spectra to check if data is loaded properly
#plt.scatter(wavelength_grid_red, spectra_red[100,:])
#plt.show()

# a. Perform a “full” PCA analysis on these data (number of components equal to number of dimensions) and then: --> 2676 components then...

# 1) First center data by calculating and subtracting by mean
spectra_mean_ = np.mean(spectra_red[:,0]) 
#print("Galaxy spectra mean: ", spectra_mean)   #0.3389
wavelength_mean = np.mean(spectra_red[0,:])
#print("Spectra red wavelength mean: ", wavelength_mean)  #0.9624

spectra_mean = np.mean(spectra_red, axis=0)
spectra_std = np.std(spectra_red, axis=0)

centered_spectras = spectra_red[0,:] - spectra_mean
print("Centered galaxy spectra: ", centered_spectras)

centered_wavelengths = spectra_red[0,:] - wavelength_mean
print("Centered spectra wavelengths: ", centered_wavelengths)

#print("Standardized new_spectra_red: ", new_spectra_red)

# 2) Solve for the covariance matrix
#covariance_matrix = empirical_covariance(spectra_red)  #np.cov?
covariance_matrix = np.cov(spectra_red)
print("Covariance matrix: ", covariance_matrix) 

# Spectras variance
# for k in centered_spectras:
#     squared_centered_spectras = []
#     squared_s = k**2
#     squared_centered_spectras.append(squared_s)
#     sum_squared_spectras = sum(squared_centered_spectras)
#     spectras_variance = sum_squared_spectras/49999

# # Wavelengths variance
# for l in centered_wavelengths:
#     squared_centered_wavelengths = []
#     squared_wls = l**2
#     squared_centered_wavelengths.append(squared_wls)
#     sum_squared_centered_wls = sum(squared_centered_wavelengths)
#     wavelengths_variance = sum_squared_centered_wls/2675

# # # Covariance
# for m in centered_spectras:
#     for n in centered_wavelengths:
#         products = []
#         spectras_times_wavelengths = m * n
#         products.append(spectras_times_wavelengths)
#         sum_products = sum(products)
#         sum_products/ # --> what is the sample size here??, figured out covariance matrix funtion is built in to package

# 3. Solve for the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
#print(covariance_matrix.shape)
#print("Eigenvalues: ", eigenvalues, "Eigenvectors: ", eigenvectors)
#print(eigenvalues.shape, eigenvectors.shape)
#index_max_eigenvalue = np.argmax(eigenvalues)
#print("Location of largest eigenvalue: ", index_max_eigenvalue)
#print("Eigenvector with largest eigenvalue: ", eigenvectors[:,0])

# i. Plot the variance captured as a function of the number of (ordered) eigenvectors.  How many eigenvectors do you need to explain ~95% of the variance in the data?
# eigenvectors = directions of new principal components

# Order the eigenvalues so that they go from highest to lowest
descending_order = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[descending_order]
eigenvectors = eigenvectors[:, descending_order]
# print("Eigenvalues: ", eigenvalues)
print("Eigenvectors: ", eigenvectors)

# Find the variances of the covariance matrix
# Principal components = eigenvectors
# Amount of variance from each PC = eigenvalue
# variances = np.diag(covariance_matrix)
# print("Variances: ", variances)

# Calculate variance for each PC
plt.figure(1)
plt.scatter(wavelength_grid_red, spectra_red[0,:])
plt.xlabel("Wavelength Data")
plt.ylabel("Unique Spectra Wavelengths")
plt.show()

pca = PCA(n_components = len(spectra_red[0,:]))
pca.fit(spectra_red)

standardized_spectra_red = (spectra_red - spectra_mean) / spectra_std
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
print("Explained_variance_ratio: ", explained_variance_ratio)

plt.figure(figsize=(12, 4))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o', linestyle='-')
plt.xlabel("Number of (Ordered) Eigenvectors (PCs)")
plt.ylabel("Cumulative Explained Variance")
plt.title("Variance Captured vs. Number of Eigenvectors")
plt.grid()
plt.show()

# ii. Plot the top several eigenvectors. Can you offer any physical explanations for what you find? Do you see any non-physical behavior?

optimal_eigenvectors = np.argmax(cumulative_variance >= 0.95) + 1
print("Optimal Eigenvectors: ", optimal_eigenvectors)  #5

top_eigenvectors = np.arange(eigenvectors(0, 5))
transformed = np.dot(standardized_spectra_red, top_eigenvectors)

plt.figure(3)
plt.scatter(len(transformed), transformed, alpha=0.2)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Computed")
plt.grid()
plt.show()

# b. Now perform a PCA analysis with only a handful of components (justify your choice).
# Choose based on explained variance ratio
# Variance reaches ~95% at 5 components
pca = PCA(5)
pca.fit_transform(
handful_spectra_red = spectra_red[0:5,0]
handful_spectra_mean_ = np.mean(handful_spectra_red) 
handful_spectra_std = np.std(handful_spectra_red, axis=0)

handful_centered_spectras = handful_spectra_red - handful_spectra_mean

handful_covariance_matrix = empirical_covariance(handful_spectra_red)

eigenvalues, eigenvectors = np.linalg.eig(handful_covariance_matrix)


# i. Plot a few example spectra and your PCA fit.  How well is the data explained by the model?

# ii. Plot the first two eigenvalues for every galaxy.  Can you make sense of the distribution?
# iii. Explore a few of the features that you see.

