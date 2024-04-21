#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define FILTER_RADIUS 1

const char* inputFilePath = "input_rgb_values.txt";
const char* outputFilePath = "outputImage.txt";

struct Pixel {
    unsigned char r, g, b;
};

__global__ void denoiseImageWiener(const Pixel* input, Pixel* output, int width, int height, float noise_variance) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int count = 0;
    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    float sumR2 = 0.0f, sumG2 = 0.0f, sumB2 = 0.0f;

    // Calculate local mean and variance
    for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy) {
        for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                Pixel neighbor = input[ny * width + nx];
                sumR += neighbor.r;
                sumG += neighbor.g;
                sumB += neighbor.b;
                sumR2 += neighbor.r * neighbor.r;
                sumG2 += neighbor.g * neighbor.g;
                sumB2 += neighbor.b * neighbor.b;
                count++;
            }
        }
    }

    float meanR = sumR / count;
    float meanG = sumG / count;
    float meanB = sumB / count;
    float varR = sumR2 / count - meanR * meanR;
    float varG = sumG2 / count - meanG * meanG;
    float varB = sumB2 / count - meanB * meanB;

    // Apply Wiener filter
    Pixel currentPixel = input[y * width + x];
    output[y * width + x].r = static_cast<unsigned char>((varR / (varR + noise_variance)) * (currentPixel.r - meanR) + meanR);
    output[y * width + x].g = static_cast<unsigned char>((varG / (varG + noise_variance)) * (currentPixel.g - meanG) + meanG);
    output[y * width + x].b = static_cast<unsigned char>((varB / (varB + noise_variance)) * (currentPixel.b - meanB) + meanB);
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

    readImage(inputFilePath, &h_inputImage, &width, &height);

    size_t imageSize = sizeof(Pixel) * width * height;
    h_outputImage = (Pixel*)malloc(imageSize);
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    float noise_variance = 0.02;  // Adjust this value based on your noise estimation
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    denoiseImageWiener<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, noise_variance);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Compute throughput and operations per byte
    double totalPixels = static_cast<double>(width) * static_cast<double>(height);
    double throughput = totalPixels / (milliseconds / 1000.0);  // pixels per second
    double operationsPerByte = totalPixels * 5.0 / imageSize;  // example operation estimate

    printf("Execution Time (ms): %f\n", milliseconds);
    printf("Throughput (pixels/s): %f\n", throughput);
    printf("Operations per Byte: %f\n", operationsPerByte);

    writeImage(outputFilePath, h_outputImage, width, height);

    free(h_inputImage);
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

