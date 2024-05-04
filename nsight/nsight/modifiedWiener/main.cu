#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define FILTER_RADIUS 1

struct Pixel {
    unsigned char r, g, b;
};

__global__ void coarsenedDenoiseImageWiener(Pixel* inputImage, Pixel* outputImage, int width, int height, float noiseVariance) {
    extern __shared__ Pixel sharedPixels[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    int areaWidth = BLOCK_SIZE_X + 2 * FILTER_RADIUS;
    int areaHeight = BLOCK_SIZE_Y + 2 * FILTER_RADIUS;

    for (int i = 0; i < (areaWidth * areaHeight) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); i++) {
        int localId = threadId + i * BLOCK_SIZE_X * BLOCK_SIZE_Y;
        int localX = localId % areaWidth;
        int localY = localId / areaWidth;
        int globalX = blockIdx.x * blockDim.x + localX - FILTER_RADIUS;
        int globalY = blockIdx.y * blockDim.y + localY - FILTER_RADIUS;

        if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height && localId < areaWidth * areaHeight) {
            sharedPixels[localY * areaWidth + localX] = inputImage[globalY * width + globalX];
        }
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    float sumR = 0.0, sumG = 0.0, sumB = 0.0;
    float sumRSq = 0.0, sumGSq = 0.0, sumBSq = 0.0;
    int count = 0;

    for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
        for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
            int nx = threadIdx.x + dx;
            int ny = threadIdx.y + dy;
            if (nx >= 0 && nx < BLOCK_SIZE_X + 2 * FILTER_RADIUS && ny >= 0 && ny < BLOCK_SIZE_Y + 2 * FILTER_RADIUS) {
                Pixel p = sharedPixels[ny * areaWidth + nx];
                sumR += p.r;
                sumG += p.g;
                sumB += p.b;
                sumRSq += p.r * p.r;
                sumGSq += p.g * p.g;
                sumBSq += p.b * p.b;
                count++;
            }
        }
    }

    float meanR = sumR / count;
    float meanG = sumG / count;
    float meanB = sumB / count;
    float varR = sumRSq / count - meanR * meanR;
    float varG = sumGSq / count - meanG * meanG;
    float varB = sumBSq / count - meanB * meanB;

    Pixel currentPixel = inputImage[y * width + x];
    outputImage[y * width + x].r = meanR + varR / (varR + noiseVariance) * (currentPixel.r - meanR);
    outputImage[y * width + x].g = meanG + varG / (varG + noiseVariance) * (currentPixel.g - meanG);
    outputImage[y * width + x].b = meanB + varB / (varB + noiseVariance) * (currentPixel.b - meanB);
}

void readImage(const char* filePath, Pixel** pixels, int* width, int* height) {
    FILE* file = fopen(filePath, "r");
    if (!file) {
        fprintf(stderr, "Error opening input file\n");
        exit(1);
    }

    fscanf(file, "%d %d", width, height);
    *pixels = (Pixel*)malloc(sizeof(Pixel) * (*width) * (*height));

    for (int i = 0; i < (*width) * (*height); i++) {
        fscanf(file, "%hhu %hhu %hhu", &(*pixels)[i].r, &(*pixels)[i].g, &(*pixels)[i].b);
    }

    fclose(file);
}

void writeImage(const char* filePath, Pixel* pixels, int width, int height) {
    FILE* file = fopen(filePath, "w");
    if (!file) {
        fprintf(stderr, "Error opening output file\n");
        exit(1);
    }

    fprintf(file, "%d %d\n", width, height);
    for (int i = 0; i < width * height; i++) {
        fprintf(file, "%hhu %hhu %hhu\n", pixels[i].r, pixels[i].g, pixels[i].b);
    }

    fclose(file);
}
int main() {
    const char* inputFilePath = "input_rgb_values.txt";
    const char* outputFilePath = "outputImage.txt";
    int width, height;
    float noiseVariance = 0.02;

    Pixel *h_inputImage, *h_outputImage;
    readImage(inputFilePath, &h_inputImage, &width, &height);

    // Allocate memory for the output image on the host
    h_outputImage = (Pixel*)malloc(width * height * sizeof(Pixel));  // Ensure memory is allocated

    Pixel *d_inputImage, *d_outputImage;
    cudaMalloc(&d_inputImage, width * height * sizeof(Pixel));
    cudaMalloc(&d_outputImage, width * height * sizeof(Pixel));

    cudaMemcpy(d_inputImage, h_inputImage, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    int sharedMemSize = (BLOCK_SIZE_X + 2 * FILTER_RADIUS) * (BLOCK_SIZE_Y + 2 * FILTER_RADIUS) * sizeof(Pixel);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    coarsenedDenoiseImageWiener<<<gridSize, blockSize, sharedMemSize>>>(d_inputImage, d_outputImage, width, height, noiseVariance);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;

    cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);
    writeImage(outputFilePath, h_outputImage, width, height);

    size_t totalPixels = width * height;
    size_t totalOperations = totalPixels * 8;
    size_t totalBytesProcessed = totalPixels * sizeof(Pixel);
    float operationsPerByte = totalOperations / (float)totalBytesProcessed;
    float throughput = totalPixels / seconds;

    printf("Execution Time: %.3f ms\n", milliseconds);
    printf("Throughput: %.3f pixels/s\n", throughput);
    printf("Operations per Byte: %.3f\n", operationsPerByte);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    free(h_inputImage);
    free(h_outputImage);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

