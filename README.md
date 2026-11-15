### Mesh & Manifold Coordinate Reduction Notes
#### Core:
QR Decomposition: Uses a QR decomposition matrix to extract R (depth/scale) and rotate the coordinate space. \
Parameter Storage: Stores the depth (R) with counted parameters of n × N - 2. \
Optimization Deviation: Uses a 45deg deviation for the `curve_fit` optimization function.

---
#### Approximated reduction size:
<img width="695" height="175" alt="image" src="https://github.com/user-attachments/assets/9c00191e-8bd7-4e77-a528-b70d52fcaf33" />

---
#### Fourier-Power improvements
* Structural Underfitting: Mesh structures such as planes, point clouds, and cubes suffer from structural underfitting (R^2 scores between -0.12 and 0.16) when mapping and reconstructing the y' coordinate.
* **Fourier-Power scalars:** Three additional scalars (amplitude, oscillation along x', and phase)—improved reconstruction across every tested geometry:
  * 3D Plane MSE: `0.1198` -> `0.1066`
  * 20D Sinusoidal Manifold MSE: `0.3459` -> `0.1759`
* Low High-Dimensional MSE Caveat: Some reconstructions show Mean Squared Errors around 7 × 10^−5 which happens only because the dimensionality reduction is minimal and almost nothing is actually being reconstructed.

---

#### Suggestions along final notes
* Recommended Scope: Use this approach for 3D to 4D meshes only and avoid higher dimensions.
* **Deformation:** Although a maximum reduction of **50%** is possible on 3D meshes, the reconstructed meshes can still be significantly deformed.
* **Included Files:**
  * Annotated Jupyter Notebook Full walkthrough with annotations and visualizations.
  * Generalized Python Script Standalone script to run and inspect the pure metrics directly without scrolling through the notebook.
