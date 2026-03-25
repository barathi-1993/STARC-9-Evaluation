import os
from PIL import Image
import staintools
from tqdm import tqdm

def count_png_files(input_folder):
    """
    Count total number of PNG files to process
    """
    total_files = 0
    for root, _, files in os.walk(input_folder):
        total_files += sum(1 for file in files if file.endswith(".tiff"))
    return total_files

def macenko_stain_normalization(input_folder, reference_image_path, output_folder): 
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the reference image for normalization
    reference_image = staintools.read_image(reference_image_path)
    
    # Initialize and fit the stain normalizer
    normalizer = staintools.StainNormalizer(method='macenko')
    normalizer.fit(reference_image)
    
    # Count total PNG files for progress bar
    total_files = count_png_files(input_folder)
    
    # Progress bar for normalization
    with tqdm(total=total_files, desc="Normalizing Images", unit="image") as pbar:
        # Walk through input folder, processing files in subdirectories
        for root, dirs, files in os.walk(input_folder):
            # Determine the relative path for the current directory
            relative_path = os.path.relpath(root, input_folder)
            
            # Create the corresponding directory in the output folder
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)
            
            # Process each file in the current directory
            for tile_file in files:
                if tile_file.endswith(".tiff"):
                    input_tile_path = os.path.join(root, tile_file)
                    output_tile_path = os.path.join(output_subfolder, tile_file)
                    
                    # Load the tile
                    tile_image = staintools.read_image(input_tile_path)
                    
                    # Apply stain normalization
                    try:
                        normalized_tile = normalizer.transform(tile_image)
                        
                        # Save the normalized tile
                        normalized_tile_image = Image.fromarray(normalized_tile)
                        normalized_tile_image.save(output_tile_path)
                        
                        # Update progress bar
                        pbar.update(1)
                    except Exception as e:
                        print(f"\nError processing {input_tile_path}: {e}")
                        # Still update progress bar for failed files
                        pbar.update(1)

def main():
    # Specify input folder, reference image, and output folder
    input_folder = "path"  # Replace with the path to your tiles folder
    reference_image_path = "reference_image.png"  # Replace with the path to your reference image
    output_folder = "path"  # Replace with the path to save normalized tiles
    
    # Run the normalization
    macenko_stain_normalization(input_folder, reference_image_path, output_folder)
    
    print("\nStain normalization complete!")

if __name__ == "__main__":
    main()