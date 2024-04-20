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

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f, sumWeights = 0.0f;

    for (int dy = -KNN_WINDOW_RADIUS; dy <= KNN_WINDOW_RADIUS; ++dy) {
        for (int dx = -KNN_WINDOW_RADIUS; dx <= KNN_WINDOW_RADIUS; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float weight = 1.0f / (1.0f + dx * dx + dy * dy); // Closer pixels have more weight
                Pixel neighbor = input[ny * width + nx];
                sumR += neighbor.r * weight;
                sumG += neighbor.g * weight;
                sumB += neighbor.b * weight;
                sumWeights += weight;
            }
        }
    }

    Pixel& currentPixel = output[y * width + x];
    currentPixel.r = static_cast<unsigned char>(sumR / sumWeights);
    currentPixel.g = static_cast<unsigned char>(sumG / sumWeights);
    currentPixel.b = static_cast<unsigned char>(sumB / sumWeights);
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

    // Setup CUDA event for timing
    cudaEvent_t start, stop;
    float milliseconds = 0;

    readImage(inputFilePath, &h_inputImage, &width, &height);

    size_t imageSize = sizeof(Pixel) * width * height;
    h_outputImage = (Pixel*)malloc(imageSize);
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    // Start measuring time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the denoise kernel
    denoiseImage<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    // Stop measuring time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);
    writeImage(outputFilePath, h_outputImage, width, height);

    free(h_inputImage);
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Calculate metrics
    size_t totalPixels = static_cast<size_t>(width) * height;
    float executionTimeSeconds = milliseconds / 1000.0f; // Convert milliseconds to seconds
    float throughput = totalPixels / executionTimeSeconds; // Pixels processed per second

    // Assuming roughly 8 operations (3 additions, 3 multiplications, 2 divisions) per pixel for simplicity
    size_t totalOperations = totalPixels * 8;
    size_t totalBytesProcessed = totalPixels * sizeof(Pixel); // Total bytes processed
    float operationsPerByte = static_cast<float>(totalOperations) / totalBytesProcessed;

    printf("Execution Time: %.3f ms\n", milliseconds);
    printf("Throughput: %.3f pixels/s\n", throughput);
    printf("Operations per Byte: %.3f\n", operationsPerByte);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
