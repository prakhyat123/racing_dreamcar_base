import argparse
import os
from pathlib import Path
import cv2
import numpy as np

#https://www.hackster.io/kemfic/simple-lane-detection-c3db2f
def color_filter(image):
    #convert to HLS to mask based on HLS
    image = np.array(image)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    yellower = np.array([10,0,90])
    yelupper = np.array([50,255,255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def main(args):
    print("Hello World")
    inPath = args.input_dir
    outPath = args.output_dir

    if not os.path.exists(outPath):
        os.makedirs(outPath)

    for imagePath in os.listdir(inPath):
        #image input path
        imgPath = os.path.join(inPath, imagePath)
        #Convert the image
        img = cv2.imread(imgPath)
        img = color_filter(img)
        #image output path
        imgOutPath = os.path.join(outPath, 'filter_'+imagePath)
        #Save image in the output path
        print(imgOutPath)
        cv2.imwrite(imgOutPath, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Representation Learning for DonkeyCar")
    parser.add_argument("--input-dir", type=Path, required=True, help='Path to the input Images')
    parser.add_argument("--output-dir", type=Path, required=True, help='Path to the output Image folder ')
    args = parser.parse_args()
    main(args)





