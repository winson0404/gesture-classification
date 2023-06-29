import os
import cv2
from tqdm import tqdm

def rescale_image(img):
    max_dim = max(img.shape[0], img.shape[1])
    scale_factor = max_dim / 320
    new_width = int(img.shape[1] / scale_factor)
    new_height = int(img.shape[0] / scale_factor)
    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image

def post_download(root, category, output_path, target):
    
    os.makedirs(f"{root}train/{target}_temp", exist_ok=True)
    # os.system("unzip -j " + output_path +" -d " + f"{root}train/{target}")

    # os.system("rm " + output_path)
    for image in tqdm(os.listdir(f"{root}{category}/{target}")):
        # print(image)
        img = rescale_image(cv2.imread(f"{root}{category}/{target}/{image}"))
        cv2.imwrite(f"{root}{category}/{target}_temp/{image}", img)
        
        os.system("rm -rf " + f"{root}{category}/{target}/{image}")
        
    os.system("rm -r " + f"{root}{category}/{target}")
    os.system("mv " + f"{root}{category}/{target}_temp " + f"{root}{category}/{target}")
    
    
if __name__ == "__main__":
    category = "test"
    for target in os.listdir(f"../data/{category}/") :
    #    print(target)
        post_download("../data/", category, f"../data/{category}/{target}", target)