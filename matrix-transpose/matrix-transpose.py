import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    n, m = A.shape # shape 2 3
    A_transpose = np.zeros((m, n)) # shape 3 2
    for i in range(m): # 3 rows
        for j in range(n): # 2 columns
            A_transpose[i, j] = A[j, i] 
    return A_transpose