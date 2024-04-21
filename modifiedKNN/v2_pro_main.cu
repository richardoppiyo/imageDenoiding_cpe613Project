#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define KNN_WINDOW_RADIUS 1

const char* inputFilePath = "input_rgb_values.txt";
const char* outputFilePath = "outputImage.txt";

struct Pixel {
    unsigned char r, g, b;
};

__global__ void denoiseImage(const Pixel* input, Pixel* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    int count = 0;

    for (int dy = -KNN_WINDOW_RADIUS; dy <= KNN_WINDOW_RADIUS; ++dy) {
        for (int dx = -KNN_WINDOW_RADIUS; dx <= KNN_WINDOW_RADIUS; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                Pixel neighbor = input[ny * width + nx];
                sumR += neighbor.r;
                sumG += neighbor.g;
                sumB += neighbor.b;
                count++;
            }
        }
    }

    Pixel& currentPixel = output[y * width + x];
    currentPixel.r = static_cast<unsigned char>(sumR / count);
    currentPixel.g = static_cast<unsigned char>(sumG / count);
    currentPixel.b = static_cast<unsigned char>(sumB / count);
}


void readImage(const char* filePath, Pixel** pixels, int* width, int* height) {
    FILE* file = fopen(filePath, "r");
    if (!file) {
        fprintf(stderr, "Error opening input file\n");
        exit(1);
    }

    fscanf(file, "%d %d", width, height);
    *pixels = (Pixel*)malloc(sizeof(Pixel) * (*width) * (*height));
    
    for (int i = 0; i < (*width) * (*height); ++i) {
        int r, g, b;
        fscanf(file, "%d %d %d", &r, &g, &b);
        (*pixels)[i] = {static_cast<unsigned char>(r), static_cast<unsigned char>(g), static_cast<unsigned char>(b)};
    }

    fclose(file);
}

void writeImage(const char* filePath, const Pixel* pixels, int width, int height) {
    FILE* file = fopen(filePath, "w");
    if (!file) {
        fprintf(stderr, "Error opening output file\n");
        exit(1);
    }

    fprintf(file, "%d %d\n", width, height);
    for (int i = 0; i < width * height; ++i) {
        fprintf(file, "%d %d %d\n", pixels[i].r, pixels[i].g, pixels[i].b);
    }

    fclose(file);
}


int main() {
    int width, height;
    Pixel *h_inputImage, *h_outputImage, *d_inputImage, *d_outputImage;

    // Setup and memory allocation remain unchanged

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Execute the kernel
    denoiseImage<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    // Stop timing and copy output image back to host remain unchanged

    // Calculate metrics
    size_t totalPixels = static_cast<size_t>(width) * height;
    float executionTimeSeconds = elapsedTime / 1000.0f; // Convert milliseconds to seconds
    float throughput = totalPixels / executionTimeSeconds; // Pixels processed per second

    size_t operationsPerPixel = 14; // Simplified operation count per pixel
    size_t totalOperations = totalPixels * operationsPerPixel;
    size_t totalBytesProcessed = 2 * totalPixels * sizeof(Pixel); // Account for both input and output
    float operationsPerByte = static_cast<float>(totalOperations) / totalBytesProcessed;

    printf("Execution Time: %f ms\n", elapsedTime);
    printf("Throughput: %f pixels/s\n", throughput);
    printf("Operations per Byte: %f\n", operationsPerByte);

    // Free memory and other finalization steps remain unchanged

    return 0;
}



int main() {
    int width, height;
    Pixel *h_inputImage, *h_outputImage, *d_inputImage, *d_outputImage;

    // Read input image
    readImage(inputFilePath, &h_inputImage, &width, &height);

    // Allocate memory
    size_t imageSize = sizeof(Pixel) * width * height;
    h_outputImage = (Pixel*)malloc(imageSize);
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    // Copy input image to device
    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    // Setup execution parameters
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Execute the kernel
    denoiseImage<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    // Stop timing and copy output image back to host remain unchanged

    // Calculate metrics
    size_t totalPixels = static_cast<size_t>(width) * height;
    float executionTimeSeconds = elapsedTime / 1000.0f; // Convert milliseconds to seconds
    float throughput = totalPixels / executionTimeSeconds; // Pixels processed per second

    size_t operationsPerPixel = 14; // Simplified operation count per pixel
    size_t totalOperations = totalPixels * operationsPerPixel;
    size_t totalBytesProcessed = 2 * totalPixels * sizeof(Pixel); // Account for both input and output
    float operationsPerByte = static_cast<float>(totalOperations) / totalBytesProcessed;

    printf("Execution Time: %f ms\n", elapsedTime);
    printf("Throughput: %f pixels/s\n", throughput);
    printf("Operations per Byte: %f\n", operationsPerByte);

    // Copy output image back to host
    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Write output image
    writeImage(outputFilePath, h_outputImage, width, height);

    // Free memory
    free(h_inputImage);
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    return 0;
}
