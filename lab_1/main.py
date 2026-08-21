import numpy as np
import skimage
from skimage import io, filters
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# Q1
A = np.array([[1, 2, 3], [4, 5, 6]])
print("Entire matrix:\n", A)
print("Second row:\n", A[1, :])
print("Third element of second row:", A[1, 2])
print("Last element of first row:", A[0, -1])

# Q2
B = np.random.randint(1, 10, size=(3, 3))
C = np.random.randint(1, 10, size=(3, 3))

dot_1 = np.dot(B, C)
dot_2 = np.matmul(B, C)
dot_3 = B @ C

print("Equal outputs across methods:", np.array_equal(dot_1, dot_2) and np.array_equal(dot_2, dot_3))

# Q3
# 5x_1 + 2x_2 = 33
# 1x_1 + 4x_2 = 21
A_sys = np.array([[5, 2], [1, 4]])
b_sys = np.array([33, 21])
x = np.linalg.solve(A_sys, b_sys)
print(f"x_1 = {x[0]:.2f}, x_2 = {x[1]:.2f}")

# Q4
matrix = np.array([[1, 3], [4, -3]])
eigenvalues, eigenvectors = np.linalg.eig(matrix)
determinant = np.linalg.det(matrix)

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
print("Determinant:", determinant)

# Q5
# 1. Load Data
data = pd.read_csv("Heart Attack.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 2. Min-Max Normalization (0 to 1)
X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# 3. Plot normalized features in subplots
num_cols = X_norm.shape[1]
fig, axes = plt.subplots(nrows=(num_cols + 2) // 3, ncols=3, figsize=(12, 8))
axes = axes.flatten()
for i in range(num_cols):
    axes[i].plot(X_norm[:, i])
    axes[i].set_title(f"Feature {i}")
plt.tight_layout()
plt.show()

# 4. Covariance & Distance matrix
data_cov = np.cov(X_norm, rowvar=False)

# Distance matrix / Dot product across feature vectors
dist_ij = np.dot(X_norm.T, X_norm)
plt.figure(figsize=(6, 5))
plt.imshow(dist_ij, cmap='viridis')
plt.colorbar(label='Dot Product Distance')
plt.title("Distance Matrix (dist_ij)")
plt.show()

# 5. Histogram fitting to Gaussian PDF
hist_i = 2  # Feature index to plot
selected_col = X_norm[:, hist_i]

# Fit parameters
mu, std = norm.fit(selected_col)

plt.figure()
count, bins, ignored = plt.hist(selected_col, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
pdf = norm.pdf(x_axis, mu, std)
plt.plot(x_axis, pdf, 'k', linewidth=2, label=f"Fit: μ={mu:.2f}, σ={std:.2f}")
plt.xlabel(f"Column {hist_i}")
plt.ylabel("Probability Density")
plt.title(f"Gaussian Fit for Feature {hist_i}")
plt.legend()
plt.show()

# Q6
a = np.ones((100, 1), dtype=np.uint8) * 255
b = np.zeros((100, 1), dtype=np.uint8)

# Horizontal & Vertical Concatenation
h_concat = np.hstack((a, b))
v_concat = np.vstack((a, b))

print("Horizontal Concat Shape:", h_concat.shape)
print("Vertical Concat Shape:", v_concat.shape)

# Convert grid array to synthetic image
img = np.hstack([a if i % 2 == 0 else b for i in range(100)])  # 100x100 grid
print("Generated Image Shape:", img.shape)

# Save and Read back
cv.imwrite("synthetic_img.png", img)
img_read = cv.imread("synthetic_img.png", cv.IMREAD_GRAYSCALE)

print("Read Image Shape:", img_read.shape)
print("Pixel value at (100, 50):", img_read[99, 49])  # 0-indexed bounds

# Q7
img_leaf = cv.imread("practice_raw_image.PNG")
img_leaf_gray = cv.imread("practice_raw_image.PNG", cv.IMREAD_GRAYSCALE)

# Plot Color vs Gray
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(cv.cvtColor(img_leaf, cv.COLOR_BGR2RGB))
ax[0].set_title("Color Image")
ax[1].imshow(img_leaf_gray, cmap='gray')
ax[1].set_title("Gray Scale Image")
plt.show()

# Sobel (a) Standard Library
sobel_x = cv.Sobel(img_leaf_gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(img_leaf_gray, cv.CV_64F, 0, 1, ksize=3)
sobel_lib = np.hypot(sobel_x, sobel_y)

# Sobel (b) Custom Kernel
Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
custom_x = cv.filter2D(img_leaf_gray, cv.CV_64F, Kx)
custom_y = cv.filter2D(img_leaf_gray, cv.CV_64F, Ky)
sobel_custom = np.hypot(custom_x, custom_y)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(sobel_lib, cmap='gray')
ax[0].set_title("Sobel (Standard Lib)")
ax[1].imshow(sobel_custom, cmap='gray')
ax[1].set_title("Sobel (Custom Kernel)")
plt.show()

# Q8
# (a) Gaussian Standard Lib
blur_lib = cv.GaussianBlur(img_leaf_gray, (5, 5), 1)

# (b) Custom Gaussian Kernel
def get_gaussian_kernel(ksize, sigma):
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    gauss = np.exp(-0.5 * np.square(ax / sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

kernel_custom = get_gaussian_kernel(5, 1)
blur_custom = cv.filter2D(img_leaf_gray, -1, kernel_custom)

# Gaussian Noise
noise = np.random.normal(0, np.sqrt(2), img_leaf_gray.shape)
img_noisy = np.clip(img_leaf_gray + noise, 0, 255).astype(np.uint8)
img_denoised = cv.GaussianBlur(img_noisy, (5, 5), 1)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[0, 0].imshow(img_leaf_gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 1].imshow(img_noisy, cmap='gray')
axes[0, 1].set_title("Noisy")
axes[1, 0].imshow(img_denoised, cmap='gray')
axes[1, 0].set_title("Denoised")
axes[1, 1].imshow(blur_custom, cmap='gray')
axes[1, 1].set_title("Custom Blur")
plt.tight_layout()
plt.show()

# Q9
# 1D Fourier Signal Analysis
t = np.linspace(0, 1, 1000, endpoint=False)
sig1 = 5 * np.sin(2 * np.pi * 100 * t)
sig2 = 1 * np.sin(2 * np.pi * 100 * t) + 2 * np.sin(2 * np.pi * 200 * t)

fft1 = np.fft.fft(sig1)
freqs = np.fft.fftfreq(len(t), 1/1000)

plt.figure(figsize=(10, 3))
plt.plot(freqs[:500], np.abs(fft1)[:500])
plt.title("1D FFT Spectrum (100 Hz Signal)")
plt.xlabel("Frequency (Hz)")
plt.show()

# 2D Fourier Transformation on Image
dft = np.fft.fft2(img_leaf_gray)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
phase_spectrum = np.angle(dft_shift)

# Low-pass / High-pass Masks
rows, cols = img_leaf_gray.shape
crow, ccol = rows // 2, cols // 2
r = 30
y_idx, x_idx = np.ogrid[:rows, :cols]
mask_area = (x_idx - ccol)**2 + (y_idx - crow)**2 <= r**2

# Low Pass Mask
lpf_mask = np.zeros((rows, cols), np.uint8)
lpf_mask[mask_area] = 1

# Apply Low Pass Filter
fshift_lpf = dft_shift * lpf_mask
img_lpf = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_lpf)))

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
axes[0].imshow(img_leaf_gray, cmap='gray')
axes[0].set_title("Original")
axes[1].imshow(magnitude_spectrum, cmap='gray')
axes[1].set_title("Magnitude Spectrum")
axes[2].imshow(lpf_mask, cmap='gray')
axes[2].set_title("LPF Mask")
axes[3].imshow(np.log(np.abs(fshift_lpf) + 1), cmap='gray')
axes[3].set_title("Masked FFT")
axes[4].imshow(img_lpf, cmap='gray')
axes[4].set_title("LPF Image")
plt.tight_layout()
plt.show()

#Q10
cap = cv.VideoCapture("20250828_130534.mp4")
ret, first_frame = cap.read()

if ret:
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    frame_count = 0

    while cap.isOpened() and frame_count < 5:
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Dense Optical Flow computation
        flow = cv.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        print(f"Frame {frame_count + 1} Mean Pixel Displacement: {np.mean(magnitude):.4f} pixels")
        
        prev_gray = curr_gray
        frame_count += 1

    cap.release()