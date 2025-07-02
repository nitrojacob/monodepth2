# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Run through a video file, and dump the depths and the pose for each frame
# Normal inference does only depth estimation through depth encoder-decoder
# We load the pose encoder-decoder pair that was used during training to
# calculate the camera pose between frames and writes the pose matrix to disk.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import time

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth, transformation_from_parameters
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "baseline_mono_640x192",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"
    data_dir = "/mnt/wksp/sfm/MonoNav/data/demo_video"
    camera_source='crazyflie'
    # Set & create directories for images
    rgb_dir = os.path.join(data_dir, camera_source + "-rgb-images")
    pose_dir = os.path.join(data_dir, camera_source + "-poses")
    os.mkdir(pose_dir) if not os.path.exists(pose_dir) else None
    kinect_img_dir = os.path.join(data_dir, "kinect-rgb-images")
    os.mkdir(kinect_img_dir) if not os.path.exists(kinect_img_dir) else None
    kinect_depth_dir = os.path.join(data_dir, "kinect-depth-images")
    os.mkdir(kinect_depth_dir) if not os.path.exists(kinect_depth_dir) else None


    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    #download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")


    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device, weights_only=True)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device, weights_only=True)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    #print('ENCODER')
    #print(encoder)
    #print('DECODER')
    #print(depth_decoder)

    print("   Loading pretrained pose-encoder")
    pose_encoder = networks.ResnetEncoder(18, False, num_input_images=2)
    loaded_dict = torch.load(pose_encoder_path, map_location=device, weights_only=True)
    pose_encoder.load_state_dict(loaded_dict)
    pose_encoder.to(device)
    pose_encoder.eval()
    
    print("   Loading pretrained pose-decoder")
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc,num_input_features=1, num_frames_to_predict_for=2)
    loaded_dict = torch.load(pose_decoder_path, map_location=device, weights_only=True)
    pose_decoder.load_state_dict(loaded_dict)
    pose_decoder.to(device)
    pose_decoder.eval()

    print("   Predicting...")
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        fps = 1
        video = cv2.VideoCapture()
        video.open("/mnt/wksp/sfm/Videos/qubo/video_1718100136003.mp4")
        time_start = time.time()
        frame_number = 0
        imgs = [None, None, None]
        alpha=0.1
        aspect=3.33
        while True:
            retval, orig_image = video.read()
            h,w,c = orig_image.shape
            crop_begin = int(h/2-w/aspect/2)
            crop_end = int(h/2 + w/aspect/2)
            orig_image = orig_image[crop_begin:crop_end,:]  #Crop the image to same aspect ratio as training image
            orig_image = cv2.resize(orig_image, (640, 192))
            input_image=pil.fromarray(orig_image)
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            imgs[2] = imgs[1]
            imgs[1] = imgs[0]
            imgs[0] = input_image.to(device)

            if imgs[2] == None or imgs[1] == None:
                continue
            pose_input = torch.cat((imgs[0], imgs[1]), dim=1)

            # PREDICTION
            features = encoder(imgs[1])
            outputs = depth_decoder(features)
            pose_features = [pose_encoder(pose_input)]
            #print('Pose features:', len(pose_features),type(pose_features))

            axis_angle, translation = pose_decoder(pose_features)
            xform_matrix = transformation_from_parameters(axis_angle[:,0], translation[:,0], False) 
            print(xform_matrix)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            scaled_disp, depth = disp_to_depth(disp_resized_np, 1, 100000)
            depth *= 1000   #Convert to mm
            print("max", np.max(depth), "min", np.min(depth))
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            cv2.imshow("MONODEPTH2", colormapped_im)
            cv2.imshow("Original", orig_image)
            if time.time()-time_start > 0:
                fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start)  # exponential moving average
                time_start = time.time()
            print(f"\rFPS: {round(fps,2)}", end="")

            cv2.imwrite(kinect_img_dir + "/kinect_frame-%06d.rgb.jpg"%(frame_number), orig_image)
            cv2.imwrite(kinect_depth_dir + "/" + "kinect_frame-%06d.depth.jpg"%(frame_number), colormapped_im)
            np.save(kinect_depth_dir + "/" + "kinect_frame-%06d.depth.npy"%(frame_number), depth) # saved in meters
            np.savetxt(pose_dir + "/" + camera_source + "_frame-%06d.pose.txt"%(frame_number), xform_matrix[0,:])


            if cv2.waitKey(1) == 27:  # Escape key
                break

            frame_number += 1

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
