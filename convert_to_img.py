from PIL import Image
import numpy as np

def reconstruct_image_from_file(input_filename, output_image_path):
    with open(input_filename, 'r') as file:
        # Read the dimensions from the first line
        width, height = map(int, file.readline().strip().split())

        # Prepare an empty array to hold the image data
        image_data = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Read the RGB values and fill the array
        for y in range(height):
            for x in range(width):
                r, g, b = map(int, file.readline().strip().split())
                image_data[y, x] = [r, g, b]
                
    # Create an image from the array
    image = Image.fromarray(image_data, 'RGB')
    
    # Save the image to the specified path
    image.save(output_image_path)
    print(f"Image reconstructed and saved to {output_image_path}")

# Example usage
input_filename ='v2_outputImage.txt'  # Name of the input file with the RGB values
output_image_path = 'v2_outputImage.png'  # Desired path for the reconstructed image
reconstruct_image_from_file(input_filename, output_image_path)
