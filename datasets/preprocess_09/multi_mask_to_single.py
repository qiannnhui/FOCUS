import os
import cv2
import numpy as np
import argparse

def convert_non_black_to_white(img):
    # Assume img is a BGR image (OpenCV default) and set all non-black pixels to white
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])
    
    black_mask = cv2.inRange(img, lower_black, upper_black)
    img[black_mask == 0] = [255, 255, 255]
    
    return img

def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(file_path)
            if img is not None:
                result = convert_non_black_to_white(img)
                
                output_path = os.path.join(folder_path, f"{filename}")
                cv2.imwrite(output_path, result)
                # print(f"Processed: {filename}")
            else:
                print(f"Failed to read: {filename}")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert non-black pixels to white in images.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images.")
    args = parser.parse_args()

    # folder_path = "new09_DS/ground_truth_mask"
    process_images_in_folder(args.folder_path)
    print("Processing completed.")
