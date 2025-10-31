#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
from argparse import ArgumentParser
import cmapy
from PIL import Image
sys.path.append('../..')
from inference.scene_3d_infer import Scene3DNetworkInfer


def main(): 

    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath", help="path to input image which will be processed by SceneSeg")
    args = parser.parse_args() 

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = Scene3DNetworkInfer(checkpoint_path=model_checkpoint_path)
  
    # Reading input image
    input_image_filepath = args.input_image_filepath
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((640, 320))

    # Run inference
    prediction = model.inference(image_pil)
    prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))

    # Transparency factor
    alpha = 0.97

    # Create visualization
    prediction_image = 255.0*((prediction - np.min(prediction))/ (np.max(prediction) - np.min(prediction)))
    prediction_image = prediction_image.astype(np.uint8)
    prediction_image = cv2.applyColorMap(prediction_image, cmapy.cmap('viridis'))
    image_vis_obj = cv2.addWeighted(prediction_image, alpha, frame, 1 - alpha, 0)

    # Display depth map
    window_name = 'depth'
    cv2.imshow(window_name, image_vis_obj)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
# %%