import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import os, sys
import numpy as np
def PSNR(path1, path2):
    img1 = np.array(cv.imread(path1) / 255., dtype=np.float32)
    img2_raw = cv.imread(path2, cv.IMREAD_UNCHANGED) / 255.
    img2_raw = img2_raw[..., :3] * img2_raw[..., 3:]
    img2 = np.array(img2_raw, dtype=np.float32)
    return compute_psnr(img1, img2)

if __name__ == "__main__":
    # Get the arguments: scene_name
    scene_name = sys.argv[1]
    freq = sys.argv[2]
    # Generate the path to the scene
    path = os.path.join("data", "nerf_synthetic", scene_name, "test", "r_0.png")
    
    our_path = os.path.join(".", "output.png")
    
    # Compute the PSNR
    psnr = PSNR(our_path, path)
    print("PSNR: ", psnr)
    
    # Write to the last line of the file
    file_name = f"History_{freq}MHz_{scene_name}.txt"
    with open(file_name, "a") as f:
        f.write(f"PSNR(dB): {psnr}\n")