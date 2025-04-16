import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from scipy.sparse import csr_matrix
import os
import time

def fft_1d(x):
    x = x.astype(np.complex128)  
    N = x.shape[0]
    
    if N & (N - 1) != 0:
        raise ValueError("Size of x must be a power of two")
    
    reversed_indices = bit_reverse_indices(N)
    x = x[reversed_indices]
    
    step = 2
    while step <= N:
        half_step = step // 2
        exp_factor = np.exp(-2j * np.pi * np.arange(half_step) / step)
        for k in range(0, N, step):
            idx1 = k + np.arange(half_step)
            idx2 = k + np.arange(half_step, step)
            temp = x[idx2] * exp_factor
            x[idx2] = x[idx1] - temp
            x[idx1] = x[idx1] + temp
        step *= 2
    return x

def bit_reverse_indices(N):
    m = N.bit_length() - 1
    n = np.arange(N)
    reversed_n = np.zeros(N, dtype=int)
    for i in range(m):
        reversed_n |= ((n >> i) & 1) << (m - 1 - i)
    return reversed_n

def ifft_1d(X):
    x = fft_1d(np.conj(X))
    return np.conj(x) / X.shape[0]

def fft_2d(image):
    rows, cols = image.shape
    
    if (rows & (rows - 1)) != 0 or (cols & (cols - 1)) != 0:
        raise ValueError("Image dimensions must be powers of two")
    
    fft_rows = np.array([fft_1d(row) for row in image])
    
    fft2d = np.array([fft_1d(col) for col in fft_rows.T]).T
    return fft2d

def ifft_2d(fft2d):
    rows, cols = fft2d.shape
    
    ifft_cols = np.array([ifft_1d(col) for col in fft2d.T]).T
    
    ifft2d = np.array([ifft_1d(row) for row in ifft_cols])
    return np.real(ifft2d)

def pad_image(image):
    rows, cols = image.shape
    n_rows = 1 << (rows - 1).bit_length()
    n_cols = 1 << (cols - 1).bit_length()
    padded_image = np.zeros((n_rows, n_cols), dtype=image.dtype)
    
    padded_image[:rows, :cols] = image
    return padded_image

def mode_one(image):
    
    padded_image = pad_image(image)
    
    fft_result_custom = fft_2d(padded_image)
    magnitude_spectrum_custom = np.abs(fft_result_custom)
    magnitude_spectrum_custom = np.fft.fftshift(magnitude_spectrum_custom)
    
    fft_result_numpy = np.fft.fft2(padded_image)
    magnitude_spectrum_numpy = np.abs(fft_result_numpy)
    magnitude_spectrum_numpy = np.fft.fftshift(magnitude_spectrum_numpy)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(np.log1p(magnitude_spectrum_custom), cmap='gray')
    plt.title('Fourier Transform (Custom FFT)')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(np.log1p(magnitude_spectrum_numpy), cmap='gray')
    plt.title('Fourier Transform (NumPy FFT)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def mode_two(image, radius_fraction=0.075):
    padded_image = pad_image(image)  
    fft_result = fft_2d(padded_image) 

    rows, cols = fft_result.shape
    crow, ccol = rows // 2, cols // 2  


    mask = np.zeros((rows, cols), dtype=np.uint8)
    radius = int(radius_fraction * min(rows, cols))  
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= radius**2
    mask[mask_area] = 1

    fft_result_shifted = np.fft.fftshift(fft_result)  
    fft_result_shifted *= mask  
    fft_result_filtered = np.fft.ifftshift(fft_result_shifted)  
    denoised_image = np.abs(ifft_2d(fft_result_filtered))  

   
    denoised_image = denoised_image[:image.shape[0], :image.shape[1]]


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image, cmap='gray')
    plt.title(f'Denoised Image (Radius: {radius})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def mode_three(image):
    padded_image = pad_image(image)
    fft_result = fft_2d(padded_image) 
    magnitude = np.abs(fft_result)
    
    flat = magnitude.flatten()
    total_coeffs = len(flat)

    compression_levels = [0, 50, 90, 95, 99, 99.9]
    plt.figure(figsize=(12, 8))


    for i, level in enumerate(compression_levels):
        if level == 0:
            compressed_fft = fft_result.copy()
        else:
            keep_coeffs = int(total_coeffs * ((100 - level) / 100)) 
            sorted_indices = np.argsort(-flat)  
            mask = np.zeros(total_coeffs, dtype=bool)
            mask[sorted_indices[:keep_coeffs]] = True  

            compressed_fft = np.zeros_like(fft_result, dtype=np.complex128)
            compressed_fft.flat[mask] = fft_result.flat[mask]

        sparse_fft = csr_matrix(compressed_fft)
        filename = f"sparse_compression_{level:.1f}.npz"
        sparse_fft_file = os.path.join('.', filename)
        np.savez_compressed(
            sparse_fft_file,
            data=sparse_fft.data,
            indices=sparse_fft.indices,
            indptr=sparse_fft.indptr,
            shape=sparse_fft.shape,
        )


        file_size = os.path.getsize(sparse_fft_file)


        compressed_image = np.abs(ifft_2d(compressed_fft))
        compressed_image = compressed_image[:image.shape[0], :image.shape[1]] 

        plt.subplot(2, 3, i+1)
        plt.imshow(compressed_image, cmap='gray')
        plt.title(f'{level}% Compression\nFile Size: {file_size} bytes')
        plt.axis('off')

        non_zero_coeffs = np.count_nonzero(compressed_fft)
        print(f"Compression Level: {level}%")
        print(f"File size of sparse matrix: {file_size} bytes")
        print(f"Number of non-zero coefficients: {non_zero_coeffs}")

    plt.tight_layout()
    plt.show()


def mode_four():
    sizes = [2 ** n for n in range(5, 12)]  
    num_runs = 10 

    naive_times = []
    naive_stds = []

    fft_times = []
    fft_stds = []

    for N in sizes:
        naive_run_times = []
        fft_run_times = []

        for _ in range(num_runs):
            x = np.random.rand(N)

            # Naive DFT timing
            start_time = time.perf_counter()
            naive_dft(x) 
            end_time = time.perf_counter()
            naive_run_times.append(end_time - start_time)

            # Iterative FFT timing
            start_time = time.perf_counter()
            fft_1d(x)  
            end_time = time.perf_counter()
            fft_run_times.append(end_time - start_time)


        naive_times.append(np.mean(naive_run_times))
        naive_stds.append(np.std(naive_run_times))
        fft_times.append(np.mean(fft_run_times))
        fft_stds.append(np.std(fft_run_times))

    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes, naive_times, yerr=[2 * s for s in naive_stds], fmt='o-', label='Naive DFT', capsize=5)
    plt.errorbar(sizes, fft_times, yerr=[2 * s for s in fft_stds], fmt='o-', label='Iterative FFT', capsize=5)
    plt.xlabel('Input Size N')
    plt.ylabel('Average Time (s)')
    plt.title('Runtime Complexity with Confidence Intervals')
    plt.legend()
    plt.grid(True)
    plt.xscale('log', base=2)
    plt.show()

    print('Input Sizes:', sizes)
    print('Naive DFT Times (Mean):', naive_times)
    print('Naive DFT Times (Std Dev):', naive_stds)
    print('FFT Times (Mean):', fft_times)
    print('FFT Times (Std Dev):', fft_stds)

def naive_dft(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return np.dot(W, x)

def main():
    parser = argparse.ArgumentParser(description='FFT Processing')
    parser.add_argument('-m', '--mode', type=int, default=1, help='Mode of operation')
    parser.add_argument('-i', '--image', type=str, default='image.jpg', help='Image filename')
    args = parser.parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print('Error: Image not found.')
        return
    image = image.astype(np.float64)
    if args.mode == 1:
        mode_one(image)
    elif args.mode == 2:
        mode_two(image)
    elif args.mode == 3:
        mode_three(image)
    elif args.mode == 4:
        mode_four()
    else:
        print('Invalid mode selected.')

if __name__ == '__main__':
        main()