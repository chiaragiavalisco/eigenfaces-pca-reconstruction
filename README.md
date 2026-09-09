# Eigenfaces Recognition using PCA

An implementation of the classic Eigenfaces facial recognition and reconstruction algorithm using Principal Component Analysis (PCA) in Python.

**Authors:** Raffaela Vetrano, Chiara Giavalisco

---

## Overview

The Eigenfaces method, introduced by Sirovich and Kirby in 1987, applies Principal Component Analysis (PCA) to project high-dimensional face images into a lower-dimensional subspace termed **"face space"**. 

By calculating the eigenvectors of the image covariance matrix, we identify the primary directions of variance across the dataset. These orthogonal components—the **eigenfaces**—capture the most prominent shared facial features (e.g., eyes, nose, mouth positions) and serve as a basis to reconstruct, represent, and classify face images.

### Key Objectives
* Reduce facial data dimensionality using SVD/PCA.
* Determine the optimal number of principal components to preserve at least **95% of the total variance**.
* Evaluate face reconstruction performance over an increasing number of eigenvectors.
* Test model robustness on:
  1. A face present within the training set.
  2. An external face outside the training set.
  3. A non-face object (a mug) to evaluate rejection capabilities.

---

## Mathematical Background

1. **Vector Representation:**  
   Each grayscale image of dimensions $M \times N$ ($92 \times 112$) is flattened into a single vector of length $D = M \cdot N = 10{,}304$.

2. **Data Centering:**  
   Given the data matrix $X$ of stacked images, we subtract the mean face $\mu$:
   $$X = X - \mu$$

3. **Covariance & Eigenvectors:**  
   The sample covariance matrix is formed:
   $$C = X \cdot X^T$$
   The eigenvectors of $C$, ordered by their corresponding eigenvalues in descending order, form the orthogonal basis for the face space.

4. **Projection & Reconstruction:**  
   Any centered image $x - \mu$ is projected into the subspace via the projection matrix $W$:
   $$y = (x - \mu) \cdot W$$
   The image can then be reconstructed using:
   $$\hat{x} = y \cdot W^T + \mu$$

---

## Dataset

* **Structure:** 40 distinct individuals, each with 10 images (total of 400 images).
* **Resolution:** $92 \times 112$ pixels, converted to grayscale.
* **Format:** Stored across subject subdirectories and processed into a flattened matrix format.

---

## Workflow & Pipeline

1. **Data Preprocessing:**  
   * Load images, convert to single-channel grayscale (`L`), and normalize pixel intensities to the range $[0, 255]$.
   * Compute the **Mean Face** $\mu$ across all training samples.

2. **Subspace Computation (PCA):**  
   * Compute eigenvalues and eigenvectors using economy-size/Gram matrix formulation.
   * Extract top eigenfaces ($M = 16$ for direct visualization).

3. **Variance Thresholding:**  
   * Analyze the cumulative explained variance.
   * Preserving **95% of the total variance** requires **190 principal components**.

4. **Iterative Reconstruction:**  
   * Reconstruct images using varying numbers of eigenvectors (from 10 up to 320).
   * 10 components fail to preserve identity, whereas 190 components achieve high-fidelity facial reconstruction.

---

## Experimental Results

| Test Subject | Description | Result | Reconstruction Fidelity |
| :--- | :--- | :--- | :--- |
| **In-Dataset Face** (`Index 111`) | Sample from the 400 training images | Low reconstruction error | **High** (Accurate reconstruction) |
| **External Face** | Frontal face outside training set | Moderate/low relative error | **Partial/Distorted** (Generic human features captured) |
| **Non-Face Object** | Photograph of a mug | Failed reconstruction | **None** (Highlights model bounds) |

### Why Did the External Face Yield a Relatively Low Error?
Even though the external face was not in the training set, the relative error was lower than expected due to:
* **Common Facial Geometry:** Eigenfaces encode generic features (eyes, nose, mouth) shared by human faces.
* **Component Volume:** Retaining 190 components provides enough spatial information to partially approximate unfamiliar face structures.
* **Preprocessing:** Grayscale conversion and uniform scaling discard color variations and subtle lighting cues that would otherwise penalize reconstruction error.
* **Model Linearity:** PCA assumes a strictly linear subspace, meaning a low algebraic error does not always reflect visual identity match.

> **Takeaway:** A low relative numerical error alone is not sufficient proof of identity; qualitative visual reconstruction remains critical.

---

## Limitations

* **Linearity Assumption:** The model assumes a linear relationship between image features and identity, which does not hold under severe pose, expression, or illumination shifts (Tabachnick & Fidell, 2013).
* **Sensitivity to Background & Occlusions:** Facial hair, glasses, and background variations can disrupt eigenvector projections.

---

## References

* Sirovich, L., & Kirby, M. (1987). *Low-dimensional procedure for the characterization of human faces*. Journal of the Optical Society of America A, 4(3), 519-524.
* Tabachnick, B. G., & Fidell, L. S. (2013). *Using Multivariate Statistics*. Pearson.
