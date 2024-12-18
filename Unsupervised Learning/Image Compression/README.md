# Image Compression using SVD
___

### Overview

Image compression using Singular Value Decomposition (SVD) is a technique that leverages matrix factorization to reduce the storage requirements of digital images while preserving important features. Here's how it works:

## Mathematical Foundation

SVD decomposes a matrix $A$ into three matrices: $A = UΣV^T$

- $U$ and $V$ are orthogonal matrices containing left and right singular vectors
- $Σ$ is a diagonal matrix containing singular values in descending order

For an $m x n$ image matrix, we get:
- $U$: $m x m$ matrix
- $Σ$: $m x n$ diagonal matrix
- $V^T$: $n x n$ matrix

## Compression Process

1. Convert the image to a matrix of pixel values.
2. Apply SVD to decompose the matrix.
3. Keep only the $k$ largest singular values and corresponding vectors.
4. Reconstruct the image using: $A_k = U_k * Σ_k * V_k^T$

Where $k < min(m,n)$, resulting in a low-rank approximation of the original image.

#### Benefits

1. **Efficient storage**: Compressed representation requires $k(m+n+1)$ values instead of $m*n$.
2. **Energy compaction**: Most image information is concentrated in the largest singular values.
3. **Adaptability**: Compression ratio can be adjusted by varying k.

#### Limitations

1. **Computational complexity**: SVD calculation can be intensive for large images.
2. **Loss of information**: Some detail is lost in the compression process.
3. **Not optimal for all image types**: Performance varies based on image characteristics.

SVD-based compression offers a flexible approach to balance image quality and file size, making it useful in 
various digital image processing applications.
___

### Datesets

This uses an image provided in the working directory called 'image.png'. The user is welcome to supply their own
images for experimentation.

### Reproducibility

If you intend to use your own images, ensure you change the `image_path` variable to the proper path in your environment