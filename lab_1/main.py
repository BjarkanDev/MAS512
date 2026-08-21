import numpy as np
import skimage
import cv2 as cv
import matplotlib.pyplot as plt

#Q1

A = np.array(
    [[1, 2, 3],
     [4, 5, 6]]
)

print("Entire matrix:")
print(A)

second_row = A[1, :]
print("Second row:")
print(second_row)

third_element_second_row = A[1, 2]
print("Third element of the second row:")
print(third_element_second_row)

last_element_first_row = A[0, -1]
print("Last element of the first row:")
print(last_element_first_row)

# Q2
B = np.random.randint(1, 10, size=(3, 3))
print("Random 3x3 matrix B:")
print(B)

C = np.random.randint(1, 10, size=(3, 3))
print("Random 3x3 matrix C:")
print(C)

dot_product = np.dot(B, C)
print("Dot product of B and C:")
print(dot_product)

# Q3

# Solve the system of equations:
# 5x_1 + 2x_2 = 33
# x_1 + 4x_2 = 21

# x_1 = 21 - 4x_2
# 5 * (21 - 4x_2) + 2x_2 = 33
# 105 - 20x_2 + 2x_2 = 33
# -18x_2 = -72
# x_2 = 4

x_2 = 4
x_1 = 21 - 4 * x_2

print(f"x_1 = {x_1}, x_2 = {x_2}")

# Q4

matrix = np.array([[1, 3], [4, -3]])

# Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("Eigenvalues:")
print(eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# Determinant
determinant = np.linalg.det(matrix) # Alternative is multiplying all eigenvalues together
print("Determinant:")
print(determinant)
