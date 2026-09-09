"""
Facial Recognition and Reconstruction using Eigenfaces (PCA).

This script downloads an image dataset from Dropbox, preprocesses grayscale
images, computes Principal Component Analysis (PCA) to extract Eigenfaces, and
performs face reconstruction alongside error analysis.
"""

import os
import shutil
import sys
import zipfile

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests

# ==============================================================================
# 1. Dataset Download and Extraction
# ==============================================================================

# Clean up previously extracted files if needed (uncomment to re-download)
# shutil.rmtree('/content/extracted_files', ignore_errors=True)

dropbox_url = (
    'https://www.dropbox.com/scl/fo/vvhwf946e8rr7g7ektgm6/'
    'AJDPZOe5Lh2g4vQLFYJ5lis?rlkey=zq9ugsjmy19nuisk1zn8jktqg&st=mmncjzt2&dl=1'
)
output_file = 'example.zip'
extracted_dir = 'extracted_files'

# Download the archive from Dropbox
print("Downloading dataset...")
response = requests.get(dropbox_url, stream=True)
with open(output_file, 'wb') as file:
    for chunk in response.iter_content(chunk_size=512):
        if chunk:  # Filter out keep-alive chunks
            file.write(chunk)
print("Download completed.")

# Extract the archive
print(f"Extracting dataset to '{extracted_dir}'...")
with zipfile.ZipFile(output_file, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)
print(f"Extraction completed. Files available in '{extracted_dir}' directory.")


# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def read_images(path, sz=None):
    """
    Read grayscale images from a nested directory structure.

    Parameters
    ----------
    path : str
        Root path containing subject folders with images.
    sz : tuple of int, optional
        Target size (width, height) to resize images.

    Returns
    -------
    list
        [X, y] where X is a list of image matrices and y is the class label list.
    """
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert('L')

                    if sz is not None:
                        # Resampling filter: Image.Resampling.LANCZOS or Image.ANTIALIAS
                        resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
                        im = im.resize(sz, resample_filter)

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)

                except IOError as e:
                    print(f"I/O error({e.errno}): {e.strerror}")
                except Exception:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c += 1
    return [X, y]


def subplot(title, images, rows, cols, sptitle="subplot", sptitles=None,
            colormap=cm.gray, ticks_visible=False, filename=None):
    """
    Plot a grid of images.

    Parameters
    ----------
    title : str
        Super title for the entire figure.
    images : list of ndarray
        List of 2D arrays representing images.
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    sptitle : str, optional
        Prefix for each subplot title.
    sptitles : list, optional
        Custom titles/identifiers for each subplot.
    colormap : matplotlib.colors.Colormap, optional
        Colormap used to render images.
    ticks_visible : bool, optional
        Whether to show axis ticks.
    filename : str, optional
        If provided, saves figure to this path instead of displaying it.
    """
    if sptitles is None:
        sptitles = []

    fig = plt.figure()
    fig.text(0.5, 0.95, title, horizontalalignment='center')

    for i in range(len(images)):
        ax0 = fig.add_subplot(rows, cols, i + 1)
        plt.setp(ax0.get_xticklabels(), visible=ticks_visible)
        plt.setp(ax0.get_yticklabels(), visible=ticks_visible)

        if len(sptitles) == len(images):
            plt.title(f"{sptitle} #{sptitles[i]}", fontsize=9)
        else:
            plt.title(f"{sptitle} #{i + 1}", fontsize=9)

        plt.imshow(np.asarray(images[i]), cmap=colormap)

    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)


def normalize(X, low, high, dtype=None):
    """
    Linearly normalize an array to a specified range [low, high].

    Parameters
    ----------
    X : array_like
        Data to normalize.
    low : float
        Lower bound of the target range.
    high : float
        Upper bound of the target range.
    dtype : numpy.dtype, optional
        Data type of the returned array.

    Returns
    -------
    ndarray
        Normalized array.
    """
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)

    # Normalize to [0...1]
    X = X - float(minX)
    X = X / float(maxX - minX)

    # Scale to [low...high]
    X = X * (high - low)
    X = X + low

    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


def asRowMatrix(X):
    """
    Flatten each 2D image into a row vector and stack into a 2D matrix.

    Parameters
    ----------
    X : list of ndarray
        List of image arrays.

    Returns
    -------
    ndarray
        Matrix of shape (n_samples, n_features).
    """
    if len(X) == 0:
        return np.array([])

    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
    return mat


def asColumnMatrix(X):
    """
    Flatten each 2D image into a column vector and stack into a 2D matrix.

    Parameters
    ----------
    X : list of ndarray
        List of image arrays.

    Returns
    -------
    ndarray
        Matrix of shape (n_features, n_samples).
    """
    if len(X) == 0:
        return np.array([])

    mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
    for col in X:
        mat = np.hstack((mat, np.asarray(col).reshape(-1, 1)))
    return mat


# ==============================================================================
# 3. Principal Component Analysis (PCA) & Projection
# ==============================================================================

def pca(X, y, num_components=0):
    """
    Perform Principal Component Analysis (PCA) using the snapshot/Gram method.

    Parameters
    ----------
    X : ndarray
        Data matrix of shape (n_samples, n_features).
    y : list or ndarray
        Class labels corresponding to the samples.
    num_components : int, optional
        Number of principal components to keep (default 0 keeps all).

    Returns
    -------
    list
        [eigenvalues, eigenvectors, mean_vector]
    """
    n, d = X.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n

    mu = X.mean(axis=0)
    X = X - mu

    if n > d:
        # Standard covariance formulation
        C = np.dot(X.T, X)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
    else:
        # Snapshot method (Gram matrix) when n <= d for efficiency
        C = np.dot(X, X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in range(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    # Sort eigenvectors in descending order of their eigenvalues
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Retain the requested number of principal components
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:, 0:num_components].copy()

    return [eigenvalues, eigenvectors, mu]


def project(W, X, mu=None):
    """Project data X into the subspace spanned by eigenvectors W."""
    if mu is None:
        return np.dot(X, W)
    return np.dot(X - mu, W)


def reconstruct(W, Y, mu=None):
    """Reconstruct data from subspace coordinates Y using eigenvectors W."""
    if mu is None:
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu


# ==============================================================================
# 4. Pipeline Execution & Eigenfaces Extraction
# ==============================================================================

# Load dataset
[X, y] = read_images("/content/extracted_files")

# Preview first 50 faces
plt.figure(figsize=(5, 10))
subplot(
    title="First 50 faces",
    images=X[0:50],
    rows=5,
    cols=10,
    sptitle="Face",
    colormap=cm.gray,
    filename="python_pca_dataset_preview.png"
)

# Normalize dataset into row matrix format
X_normalized = normalize(asRowMatrix(X), 0, 255)
n, d = X_normalized.shape

# Compute PCA
[D, W, mu] = pca(X_normalized, y)

# Mean face visualization
MU = []
mu_image = mu.reshape(X[0].shape)
MU.append(normalize(mu_image, 0, 255))
subplot(
    title="Mean Face",
    images=MU,
    rows=1,
    cols=1,
    sptitle="Mean Face",
    colormap=cm.gray,
    filename="mu_face.png"
)

# Eigenfaces visualization (top 16 components)
E = []
for i in range(min(len(X), 16)):
    e = W[:, i].reshape(X[0].shape)
    E.append(normalize(e, 0, 255))

subplot(
    title="Eigenfaces",
    images=E,
    rows=4,
    cols=4,
    sptitle="Eigenface",
    colormap=cm.gray,
    filename="python_pca_eigenfaces.png"
)

# Explained variance analysis (95% threshold)
threshold = 0.95
explained_variance_ratio = D / np.sum(D)
cumulative_variance = np.cumsum(explained_variance_ratio)
num_components = np.argmax(cumulative_variance >= threshold) + 1

print(f"Number of principal components for 95% variance: {num_components}")


# ==============================================================================
# 5. Reconstruction Across Incremental Subspace Sizes
# ==============================================================================

steps = [i for i in range(10, min(len(X), 320), 20)]
E = []
for i in range(min(len(steps), 16)):
    numEvs = steps[i]
    P = project(W[:, 0:numEvs], X[0].reshape(1, -1), mu)
    R = reconstruct(W[:, 0:numEvs], P, mu)
    R = R.reshape(X[0].shape)
    E.append(normalize(R, 0, 255))

subplot(
    title="Reconstructed face",
    images=E,
    rows=4,
    cols=4,
    sptitle="Eigenvectors",
    sptitles=steps,
    colormap=cm.gray,
    filename="python_pca_reconstruction.png"
)


# ==============================================================================
# 6. Evaluation: Dataset Face Reconstruction (Index 111)
# ==============================================================================

# Target face
A = []
face = X[111].reshape(X[0].shape)
A.append(normalize(face, 0, 255))

plt.figure()
plt.imshow(A[0], plt.cm.gray)
plt.title("Original face")

# Reconstruction with 190 eigenvectors
U = []
numEvs = 190
P = project(W[:, 0:numEvs], X[111].reshape(1, -1), mu)
R = reconstruct(W[:, 0:numEvs], P, mu)
R = R.reshape(X[111].shape)
U.append(normalize(R, 0, 255))

print("Projected shape:", asRowMatrix(U).shape)

plt.figure()
plt.imshow(U[0], plt.cm.gray)
plt.title("Reconstructed face")

# Error calculation
err = np.linalg.norm((face - U[0]), 2)
err_rel = err / np.linalg.norm(face, 2)
print("Distance =", err)
print("Relative error =", err_rel)


# ==============================================================================
# 7. Evaluation: External Face Image
# ==============================================================================

im = Image.open("/content/resized_image_face.png").convert("L")
im_array = np.array(im)

E = []
numEvs = 190
P = project(W[:, 0:numEvs], im_array.reshape(1, -1), mu)
R = reconstruct(W[:, 0:numEvs], P, mu)
R = R.reshape(im_array.shape)
E.append(normalize(R, 0, 255))

# Plot Comparison
plt.figure()
plt.subplot(1, 2, 1)
p = im_array - mu.reshape(im_array.shape)
plt.imshow(p, plt.cm.gray)
plt.title("Original face (Mean Centered)")

plt.subplot(1, 2, 2)
plt.imshow(E[0], plt.cm.gray)
plt.title("Reconstructed face")

# Error metrics
err = np.linalg.norm((im_array - E[0]), 2)
err_rel = err / np.linalg.norm(im_array, 2)
print("Distance (External Face) =", err)
print("Relative error (External Face) =", err_rel)


# ==============================================================================
# 8. Evaluation: Non-Face Image (Mug)
# ==============================================================================

im = Image.open("/content/resized_image_mug.png").convert("L")
im_array = np.array(im)

E = []
numEvs = 190
P = project(W[:, 0:numEvs], im_array.reshape(1, -1), mu)
R = reconstruct(W[:, 0:numEvs], P, mu)
R = R.reshape(im_array.shape)
E.append(normalize(R, 0, 255))

# Plot Comparison
plt.figure()
plt.subplot(1, 2, 1)
p = im_array - mu.reshape(im_array.shape)
plt.imshow(p, plt.cm.gray)
plt.title("Original mug (Mean Centered)")

plt.subplot(1, 2, 2)
plt.imshow(E[0], plt.cm.gray)
plt.title("Reconstructed mug")

# Error metrics
err = np.linalg.norm((im_array - E[0]), 2)
err_rel = err / np.linalg.norm(im_array, 2)
print("Distance (Mug) =", err)
print("Relative error (Mug) =", err_rel)
