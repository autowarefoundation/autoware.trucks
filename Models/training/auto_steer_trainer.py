#! /usr/bin/env python3

import torch
from torchvision import transforms
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
from typing import Literal, get_args
import sys

sys.path.append('..')
from model_components.auto_steer_network import AutoSteerNetwork
from data_utils.augmentations import Augmentations
from data_utils.load_data_auto_steer import VALID_DATASET_LIST


class AutoSteerTrainer():
    def __init__(
        self,  
        checkpoint_path = ""
    ):
        
        # Initializing Data
        self.homotrans_mat = None
        self.bev_image = None
        self.ego_lanes_seg = None
        self.data = None
        self.perspective_image = None
        self.bev_egopath = None
        self.bev_egoleft = None
        self.bev_egoright = None
        self.reproj_egopath = None
        self.reproj_egoleft = None
        self.reproj_egoright = None
        self.perspective_H = None
        self.perspective_W = None
        self.BEV_H = None
        self.BEV_W = None

        # Initializing BEV to Image transformation matrix
        self.homotrans_mat_tensor = None

        # Initializing BEV Image tensor
        self.bev_image_tensor = None

        # Initializing perspective Image tensor
        self.perspective_image_tensor = None     

        # Initializing Binary Segmentation Mask tensor
        self.binary_seg_tensor = None

        # Initializing Lanes Segmentation Mask tensor
        self.egolanes_tensor = None

        # Initializing Ground Truth Tensors
        self.gt_data_tensor = None
        self.gt_bev_egopath_tensor = None
        self.gt_bev_egoleft_lane_tensor = None
        self.gt_bev_egoright_lane_tensor = None
        self.gt_reproj_egopath_tensor = None
        self.gt_reproj_egoleft_lane_tensor = None
        self.gt_reproj_egoright_lane_tensor = None

        # Model predictions
        self.pred_bev_ego_path_tensor = None
        self.pred_bev_egoleft_lane_tensor = None
        self.pred_bev_egoright_lane_tensor = None
        self.pred_binary_seg_tensor = None
        self.pred_data_tensor = None
        self.pred_noisy_data_tensor = None
        self.pred_egolanes_tensor = None

        # Losses
        self.data_loss = None

        self.BEV_FIGSIZE = (4, 8)
        self.ORIG_FIGSIZE = (8, 4)

        # Currently limiting to available datasets only. Will unlock eventually
        self.VALID_DATASET_LIST = VALID_DATASET_LIST
        
        # Checking devices (GPU vs CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        print(f"Using {self.device} for inference.")

        # Instantiate model
        self.model = AutoSteerNetwork()
        
        if(checkpoint_path):
            print("Loading trained AutoSteer model from checkpoint")
            self.model.load_state_dict(torch.load \
                (checkpoint_path, weights_only = True))  
        else:
            print("Loading vanilla AutoSteer model for training")
            
        self.model = self.model.to(self.device)
        
        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = 0.0001
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.seg_loader = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

        # Gradient filters
        # Gradient - x
        self.gx_filter = torch.Tensor([[0.125, 0, -0.125],
        [0.25, 0, -0.25],
        [0.125, 0, -0.125]])
        self.gx_filter = self.gx_filter.view((1,1,3,3))
        self.gx_filter = self.gx_filter.type(torch.cuda.FloatTensor)
        self.gx_filter.to(self.device)
        
        # Gradient - y
        self.gy_filter = torch.Tensor([[0.125, 0.25, 0.125],
        [0, 0, 0],
        [-0.125, -0.25, -0.125]])
        self.gy_filter = self.gy_filter.view((1,1,3,3))
        self.gy_filter = self.gy_filter.type(torch.cuda.FloatTensor)
        self.gy_filter.to(self.device)

    # Zero gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Learning rate adjustment
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        
    # Assign input variables
    def set_data(
            self, 
            # homotrans_mat, 
            # bev_image, 
            perspective_image, 
            ego_lanes_seg, 
            # data,
            # bev_egopath, 
            # bev_egoleft, 
            # bev_egoright, 
            # reproj_egopath,
            # reproj_egoleft, 
            # reproj_egoright
        ):

        # self.homotrans_mat = np.array(homotrans_mat, dtype = "float32")
        # self.bev_image = np.array(bev_image)
        self.ego_lanes_seg = np.array(ego_lanes_seg)
        self.perspective_image = np.array(perspective_image)
        # self.data = np.array(data)
        # self.bev_egopath = np.array(bev_egopath, dtype = "float32").transpose()
        # self.bev_egoleft = np.array(bev_egoleft, dtype = "float32").transpose()
        # self.bev_egoright = np.array(bev_egoright, dtype = "float32").transpose()
        # self.reproj_egopath = np.array(reproj_egopath, dtype = "float32").transpose()
        # self.reproj_egoleft = np.array(reproj_egoleft, dtype = "float32").transpose()
        # self.reproj_egoright = np.array(reproj_egoright, dtype = "float32").transpose()
        self.perspective_H, self.perspective_W, _ = self.perspective_image.shape
        # self.BEV_H, self.BEV_W, _ = self.bev_image.shape

    # Image agumentations
    def apply_augmentations(self, is_train):

        aug = Augmentations(
            is_train = is_train, 
            data_type = "SEGMENTATION"
        )

        aug.setData(self.perspective_image, self.ego_lanes_seg)
        self.perspective_image, self.ego_lanes_seg = aug.applyTransformSeg(
            self.perspective_image, 
            self.ego_lanes_seg
        )

    # Load data as Pytorch tensors
    def load_data(self):

        # # BEV to Image matrix
        # homotrans_mat_tensor = torch.from_numpy(self.homotrans_mat)
        # homotrans_mat_tensor = homotrans_mat_tensor.type(torch.FloatTensor)
        # self.homotrans_mat_tensor = homotrans_mat_tensor.to(self.device)

        # # BEV Image
        # bev_image_tensor = self.image_loader(self.bev_image)
        # bev_image_tensor = bev_image_tensor.unsqueeze(0)
        # self.bev_image_tensor = bev_image_tensor.to(self.device)

        # Perspective Image
        perspective_image_tensor = self.image_loader(self.perspective_image)
        perspective_image_tensor = perspective_image_tensor.unsqueeze(0)
        self.perspective_image_tensor = perspective_image_tensor.to(self.device)


        # Egolanes Segmentation
        egolanes_tensor = self.seg_loader(self.ego_lanes_seg)
        egolanes_tensor = egolanes_tensor.type(torch.cuda.FloatTensor)
        egolanes_tensor = egolanes_tensor.unsqueeze(0)
        self.egolanes_tensor = egolanes_tensor.to(self.device)

        # # Data Tensor
        # data_tensor = torch.from_numpy(self.data)
        # data_tensor = data_tensor.type(torch.FloatTensor).unsqueeze(0)
        # self.gt_data_tensor = data_tensor.to(self.device)

        # # BEV Egopath
        # bev_egopath_tensor = torch.from_numpy(self.bev_egopath)
        # bev_egopath_tensor = bev_egopath_tensor.type(torch.FloatTensor)
        # self.gt_bev_egopath_tensor = bev_egopath_tensor.to(self.device)

        # # BEV Egoleft Lane
        # bev_egoleft_lane_tensor = torch.from_numpy(self.bev_egoleft)
        # bev_egoleft_lane_tensor = bev_egoleft_lane_tensor.type(torch.FloatTensor)
        # self.gt_bev_egoleft_lane_tensor = bev_egoleft_lane_tensor.to(self.device)

        # # BEV Egoright Lane
        # bev_egoright_lane_tensor = torch.from_numpy(self.bev_egoright)
        # bev_egoright_lane_tensor = bev_egoright_lane_tensor.type(torch.FloatTensor)
        # self.gt_bev_egoright_lane_tensor = bev_egoright_lane_tensor.to(self.device)
        
        # # Reprojected Egopath
        # reproj_egopath_tensor = torch.from_numpy(self.reproj_egopath)
        # reproj_egopath_tensor = reproj_egopath_tensor.type(torch.FloatTensor)
        # self.gt_reproj_egopath_tensor = reproj_egopath_tensor.to(self.device)

        # # Reprojected Egoleft Lane
        # reproj_egoleft_lane_tensor = torch.from_numpy(self.reproj_egoleft)
        # reproj_egoleft_lane_tensor = reproj_egoleft_lane_tensor.type(torch.FloatTensor)
        # self.gt_reproj_egoleft_lane_tensor = reproj_egoleft_lane_tensor.to(self.device)

        # # Reprojected Egoright Lane
        # reproj_egoright_lane_tensor = torch.from_numpy(self.reproj_egoright)
        # reproj_egoright_lane_tensor = reproj_egoright_lane_tensor.type(torch.FloatTensor)
        # self.gt_reproj_egoright_lane_tensor = reproj_egoright_lane_tensor.to(self.device)
    
    # Run Model
    def run_model(self):
        
        self.pred_binary_seg_tensor = self.model(self.perspective_image_tensor)

        # Segmentation Loss
        self.total_loss = self.calc_ego_lanes_loss()

    # Data loss
    def calc_data_loss(self):
        mAE_loss = nn.L1Loss()
        data_loss = mAE_loss(self.pred_data_tensor, self.gt_data_tensor)
        return data_loss
    
    # Data loss
    def calc_denoising_loss(self):
        mAE_loss = nn.L1Loss()
        denoising_loss = mAE_loss(self.pred_data_tensor, self.pred_noisy_data_tensor)
        return denoising_loss
    
    # EgoLanes Loss
    def calc_ego_lanes_loss(self):
        pred_ego_left_lane = self.pred_binary_seg_tensor[:, 0, :, :]
        gt_ego_left_lane = self.egolanes_tensor[:, 0, :, :]
        left_lane_loss = (
            self.calc_multi_scale_edge_loss(pred_ego_left_lane, gt_ego_left_lane) + \
            self.calc_segmentation_loss(pred_ego_left_lane, gt_ego_left_lane)
        )

        pred_ego_right_lane = self.pred_binary_seg_tensor[:, 1, :, :]
        gt_ego_right_lane = self.egolanes_tensor[:, 1, :, :]
        right_lane_loss = (
            self.calc_multi_scale_edge_loss(pred_ego_right_lane, gt_ego_right_lane) + \
            self.calc_segmentation_loss(pred_ego_right_lane, gt_ego_right_lane)
        )

        pred_other_lane = self.pred_binary_seg_tensor[:, 2, :, :]
        gt_other_lane = self.egolanes_tensor[:, 2, :, :]
        other_lane_loss = (
            self.calc_multi_scale_edge_loss(pred_other_lane, gt_other_lane) + \
            self.calc_segmentation_loss(pred_other_lane, gt_other_lane)
        )

        ego_lanes_loss = 2 * left_lane_loss + 2 * right_lane_loss + other_lane_loss

        return ego_lanes_loss


    # Segmentation Loss
    def calc_segmentation_loss(self, pred, gt):
        
        BCELoss = nn.BCEWithLogitsLoss()
        segmentation_loss = BCELoss(pred, gt)

        return segmentation_loss


    def calc_multi_scale_edge_loss(self, pred, gt):

        downsample = nn.AvgPool2d(2, stride=2)
        threshold = torch.nn.Threshold(0.0, 1.0, inplace=False)
        relu = nn.ReLU()
        pred = relu(pred)

        prediction_thresholded = threshold(pred)
        prediction_d1 = downsample(prediction_thresholded)
        prediction_d2 = downsample(prediction_d1)
        prediction_d3 = downsample(prediction_d2)
        prediction_d4 = downsample(prediction_d3)
        gt_d1 = downsample(gt)
        gt_d2 = downsample(gt_d1)
        gt_d3 = downsample(gt_d2)
        gt_d4 = downsample(gt_d3)

        edge_loss_d0 = self.calc_edge_loss(pred, gt)
        edge_loss_d1 = self.calc_edge_loss(prediction_d1, gt_d1)
        edge_loss_d2 = self.calc_edge_loss(prediction_d2, gt_d2)
        edge_loss_d3 = self.calc_edge_loss(prediction_d3, gt_d3)
        edge_loss_d4 = self.calc_edge_loss(prediction_d4, gt_d4)

        multi_scale_edge_loss = \
            (edge_loss_d0 + edge_loss_d1 + edge_loss_d2 + edge_loss_d3 + edge_loss_d4)/5

        return multi_scale_edge_loss

    
    def calc_edge_loss(self, prediction, gt):
        G_x_pred = nn.functional.conv2d(prediction, self.gx_filter, padding=1)
        G_y_pred = nn.functional.conv2d(prediction, self.gy_filter, padding=1)

        G_x_gt = nn.functional.conv2d(gt, self.gx_filter, padding=1)
        G_y_gt = nn.functional.conv2d(gt, self.gy_filter, padding=1)

        edge_diff_mAE = torch.abs(G_x_pred - G_x_gt) + \
                            torch.abs(G_y_pred - G_y_gt)
        edge_loss = torch.mean(edge_diff_mAE)

        return edge_loss

    # BEV Data Loss for the entire driving corridor
    def calc_BEV_data_loss_driving_corridor(self):

        BEV_egopath_data_loss = \
            self.calc_BEV_data_loss(self.gt_bev_egopath_tensor, self.pred_bev_ego_path_tensor)

        BEV_egoleft_lane_data_loss = \
            self.calc_BEV_data_loss(self.gt_bev_egoleft_lane_tensor, self.pred_bev_egoleft_lane_tensor)
        
        BEV_egoright_lane_data_loss = \
            self.calc_BEV_data_loss(self.gt_bev_egoright_lane_tensor, self.pred_bev_egoright_lane_tensor)

        BEV_data_loss_driving_corridor = (BEV_egopath_data_loss +  \
            BEV_egoleft_lane_data_loss + BEV_egoright_lane_data_loss)/3
        
       
        
        return BEV_data_loss_driving_corridor
    
    # BEV Gradient Loss for the entire driving corridor
    def calc_BEV_gradient_loss_driving_corridor(self):

        BEV_egopath_gradient_loss = \
            self.calc_BEV_graient_loss(self.gt_bev_egopath_tensor, 
                                       self.pred_bev_ego_path_tensor)

        BEV_egoleft_lane_gradient_loss = \
            self.calc_BEV_graient_loss(self.gt_bev_egoleft_lane_tensor, 
                                       self.pred_bev_egoleft_lane_tensor)
   
        BEV_egoright_lane_gradient_loss = \
            self.calc_BEV_graient_loss(self.gt_bev_egoright_lane_tensor, 
                                       self.pred_bev_egoright_lane_tensor)


        BEV_gradient_loss_driving_corridor = (BEV_egopath_gradient_loss +  \
            BEV_egoleft_lane_gradient_loss + BEV_egoright_lane_gradient_loss)/3
        
        return BEV_gradient_loss_driving_corridor
    
    # Reprojected Data Loss for the entire driving corridor
    def calc_reprojected_data_loss_driving_corridor(self):

        reprojected_ego_path_data_loss =  \
            self.calc_reprojected_data_loss(self.gt_reproj_egopath_tensor, 
                                            self.gt_bev_egopath_tensor, 
                                            self.pred_bev_ego_path_tensor)
        
        reprojected_egoleft_lane_data_loss =  \
            self.calc_reprojected_data_loss(self.gt_reproj_egoleft_lane_tensor, 
                                            self.gt_bev_egopath_tensor, 
                                            self.pred_bev_egoleft_lane_tensor)
        
        reprojected_egoright_lane_data_loss =  \
            self.calc_reprojected_data_loss(self.gt_reproj_egoright_lane_tensor, 
                                            self.gt_bev_egopath_tensor, 
                                            self.pred_bev_egoright_lane_tensor)

        reprojected_data_loss_driving_corridor = (reprojected_ego_path_data_loss + \
            reprojected_egoleft_lane_data_loss + reprojected_egoright_lane_data_loss)/3
        
        return reprojected_data_loss_driving_corridor
    
    # Reprojected Gradient Loss for the entire driving corridor
    def calc_reprojected_gradient_loss_driving_corridor(self):

        reprojected_ego_path_gradient_loss =  \
            self.calc_reprojected_gradient_loss(self.gt_reproj_egopath_tensor, 
                                            self.gt_bev_egopath_tensor, 
                                            self.pred_bev_ego_path_tensor)
        
        reprojected_egoleft_lane_gradient_loss =  \
            self.calc_reprojected_gradient_loss(self.gt_reproj_egoleft_lane_tensor, 
                                            self.gt_bev_egopath_tensor, 
                                            self.pred_bev_egoleft_lane_tensor)
        
        reprojected_egoright_lane_gradient_loss =  \
            self.calc_reprojected_gradient_loss(self.gt_reproj_egoright_lane_tensor, 
                                            self.gt_bev_egopath_tensor, 
                                            self.pred_bev_egoright_lane_tensor)

        reprojected_gradient_loss_driving_corridor = (reprojected_ego_path_gradient_loss + \
            reprojected_egoleft_lane_gradient_loss + reprojected_egoright_lane_gradient_loss)/3
        
        return reprojected_gradient_loss_driving_corridor
    
    # BEV Data Loss for a single lane/path element
    # Mean absolute error on predictions
    def calc_BEV_data_loss(self, gt_tensor, pred_tensor):

        gt_tensor_x_vals = gt_tensor[0,:]
        pred_tensor_x_vals = pred_tensor[0]

        data_error_sum = 0

        for i in range(0, len(gt_tensor_x_vals)):
                
            error = torch.abs(gt_tensor_x_vals[i] - pred_tensor_x_vals[i])
            
            data_error_sum = data_error_sum + error

        bev_data_loss = data_error_sum/len(gt_tensor_x_vals)

        return bev_data_loss
    
    # BEV gradient loss for a single lane/path element
    # Sum of finite difference gradients
    def calc_BEV_graient_loss(self, gt_tensor, pred_tensor):

        gt_tensor_x_vals = gt_tensor[0,:]
        pred_tensor_x_vals = pred_tensor[0]

        bev_gradient_error = 0

        for i in range(0, len(gt_tensor_x_vals) - 1):

            gt_gradient = gt_tensor_x_vals[i+1] - gt_tensor_x_vals[i]
            pred_gradient = pred_tensor_x_vals[i+1] - pred_tensor_x_vals[i]

            error = torch.abs(gt_gradient - pred_gradient)
            
            bev_gradient_error = bev_gradient_error + error

        bev_gradient_loss = bev_gradient_error/len(gt_tensor_x_vals)
        return bev_gradient_loss
    
    # Reprojected Data Loss for a single lane/path element
    # Mean absolute error on predictions
    def calc_reprojected_data_loss(self, gt_reprojected_tesnor, gt_tensor, pred_tensor):

        prediction_reprojected, _ = \
            self.getPerspectivePointsFromBEV(gt_tensor, pred_tensor)
        
        gt_tensor_x_vals = gt_tensor[0,:]
        gt_reprojected_tensor_x_vals = gt_reprojected_tesnor[0,:]
        gt_reprojected_tensor_y_vals = gt_reprojected_tesnor[1,:]

        data_error_sum = 0

        for i in range(0, len(gt_tensor_x_vals)):

            gt_reprojected_x = gt_reprojected_tensor_x_vals[i]
            prediction_reprojected_x = prediction_reprojected[i][0]
            
            gt_reprojected_y = gt_reprojected_tensor_y_vals[i]
            prediction_reprojected_y = prediction_reprojected[i][1]
            
            x_error = torch.abs(gt_reprojected_x - prediction_reprojected_x)
            y_error = torch.abs(gt_reprojected_y - prediction_reprojected_y)
            L1_error = x_error + y_error
          
            data_error_sum = data_error_sum + L1_error

        reprojected_data_loss = data_error_sum/len(gt_tensor_x_vals)
        return reprojected_data_loss
    
    # Reprojected points gradient loss for a single lane/path element
    # Sum of finite difference gradients
    def calc_reprojected_gradient_loss(self, gt_reprojected_tesnor, gt_tensor, pred_tensor):

        prediction_reprojected, _ = \
            self.getPerspectivePointsFromBEV(gt_tensor, pred_tensor)
        
        gt_tensor_x_vals = gt_tensor[0,:]
        gt_reprojected_tensor_x_vals = gt_reprojected_tesnor[0,:]

        reprojected_gradient_error = 0

        for i in range(0, len(gt_tensor_x_vals)-1):

            gt_reprojected_gradient = gt_reprojected_tensor_x_vals[i+1] \
                - gt_reprojected_tensor_x_vals[i]
            
            prediction_reprojected_gradient = prediction_reprojected[i+1][0] \
                - prediction_reprojected[i][0]
            
            error = torch.abs(gt_reprojected_gradient - prediction_reprojected_gradient)

            
            reprojected_gradient_error = reprojected_gradient_error + error

        reprojected_gradient_loss = reprojected_gradient_error/len(gt_tensor_x_vals)
            
        return reprojected_gradient_loss

    # Get the list of reprojected points from X,Y BEV coordinates
    def getPerspectivePointsFromBEV(self, gt_tensor, pred_tensor):
        gt_tensor_y_vals = gt_tensor[1,:]
        pred_tensor_x_vals = pred_tensor[0]

        perspective_image_points, perspective_image_points_normalized = \
            self.projectBEVtoImage(pred_tensor_x_vals, gt_tensor_y_vals)

        return perspective_image_points_normalized, perspective_image_points

    # Reproject BEV points to perspective image
    def projectBEVtoImage(self, bev_x_points, bev_y_points):

        perspective_image_points = []
        perspective_image_points_normalized = []

        for i in range(0, len(bev_x_points)):
            
            image_homogenous_point_x = self.BEV_W*bev_x_points[i]*self.homotrans_mat_tensor[0][0] + \
                self.BEV_H*bev_y_points[i]*self.homotrans_mat_tensor[0][1] + self.homotrans_mat_tensor[0][2]
            
            image_homogenous_point_y = self.BEV_W*bev_x_points[i]*self.homotrans_mat_tensor[1][0] + \
                self.BEV_H*bev_y_points[i]*self.homotrans_mat_tensor[1][1] + self.homotrans_mat_tensor[1][2]
            
            image_homogenous_point_scale_factor = self.BEV_W*bev_x_points[i]*self.homotrans_mat_tensor[2][0] + \
                self.BEV_H*bev_y_points[i]*self.homotrans_mat_tensor[2][1] + self.homotrans_mat_tensor[2][2]
            
            image_point = [(image_homogenous_point_x/image_homogenous_point_scale_factor), \
                (image_homogenous_point_y/image_homogenous_point_scale_factor)]
            
            image_point_normalized = [image_point[0]/self.perspective_W, image_point[1]/self.perspective_H]

            perspective_image_points.append(image_point)
            perspective_image_points_normalized.append(image_point_normalized)

        return perspective_image_points, perspective_image_points_normalized

    # Loss backward pass
    def loss_backward(self):
        self.total_loss.backward()

    # Get total loss value
    def get_total_loss(self):
        return self.total_loss.item()
    
    # BEV data loss value
    def get_validation_loss_value(self):
        return self.total_loss.detach().cpu().numpy()
    
    def get_bev_loss(self):
        return self.BEV_loss.item()
    
    def get_reprojected_loss(self):
        return self.reprojected_loss.item()
    
    def get_segmentation_loss(self):
        return self.segmentation_loss.item()

    def get_edge_loss(self):
        return self.edge_loss.item()
    
    def get_data_loss(self):
        return self.data_loss.item()
    
    def get_denoising_loss(self):
        return self.denoising_loss.item()

    # Logging losses - Total, BEV, Reprojected
    def log_loss(self, log_count):
        self.writer.add_scalar(
            "Loss", self.get_total_loss(),
            (log_count)
        )

    # Run optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Set train mode
    def set_train_mode(self):
        self.model = self.model.train()

    # Set evaluation mode
    def set_eval_mode(self):
        self.model = self.model.eval()

    # Save model
    def save_model(self, model_save_path):
        torch.save(
            self.model.state_dict(), 
            model_save_path
        )

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print("Finished training")

    # Save predicted visualization
    def save_visualization(
            self, 
            log_count, 
            vis_path = "", 
            is_train = False
    ):

        # Visualize Binary Segmentation - Ground Truth and Predictions
        fig_lane_seg, axs_seg = plt.subplots(2,1, figsize=(8, 8))

        # Visualize Binary Segmentation - Raw Lane Segs
        fig_raw_lane_seg, axs_raw_seg = plt.subplots(2,1, figsize=(8, 8))
              
        # blend factor
        alpha = 0.5

        # Creating visualization image
        vis_predict_object = np.zeros((320, 640, 3), dtype = "uint8")
        vis_predict_object = np.array(self.perspective_image)
        gt_object = np.zeros((320, 640, 3), dtype = "uint8")
        gt_object = np.array(self.perspective_image)

        # Creating raw visualization images
        vis_raw_predict_object = np.zeros((320, 640, 3), dtype = "uint8")
        gt_raw_object = np.zeros((320, 640, 3), dtype = "uint8")

        # Prediction
        egolanes_prediction = torch.squeeze(self.pred_binary_seg_tensor, 0)
        egolanes_prediction = torch.squeeze(egolanes_prediction, 0)
        egolanes_prediction = egolanes_prediction.cpu().detach().numpy()
        
        # GT
        egolanes_gt = torch.squeeze(self.egolanes_tensor, 0)
        egolanes_gt = torch.squeeze(egolanes_gt, 0)
        egolanes_gt = egolanes_gt.cpu().detach().numpy()
        
        # Getting prediction and ground truth labels
        pred_egoleft_lanes = np.where(egolanes_prediction[0,:,:] > 0)
        pred_egoright_lanes = np.where(egolanes_prediction[1,:,:] > 0)
        pred_other_lanes = np.where(egolanes_prediction[2,:,:] > 0)

        gt_egoleft_lanes = np.where(egolanes_gt[0,:,:] > 0)
        gt_egoright_lanes = np.where(egolanes_gt[1,:,:] > 0)
        gt_other_lanes = np.where(egolanes_gt[2,:,:] > 0)

        # Visualize EgoLeft Lane
        vis_predict_object[pred_egoleft_lanes[0], pred_egoleft_lanes[1], 0] = 0
        vis_predict_object[pred_egoleft_lanes[0], pred_egoleft_lanes[1], 1] = 255
        vis_predict_object[pred_egoleft_lanes[0], pred_egoleft_lanes[1], 2] = 255
        gt_object[gt_egoleft_lanes[0], gt_egoleft_lanes[1], 0] = 0
        gt_object[gt_egoleft_lanes[0], gt_egoleft_lanes[1], 1] = 255
        gt_object[gt_egoleft_lanes[0], gt_egoleft_lanes[1], 2] = 255

        vis_raw_predict_object[pred_egoleft_lanes[0], pred_egoleft_lanes[1], 0] = 0
        vis_raw_predict_object[pred_egoleft_lanes[0], pred_egoleft_lanes[1], 1] = 255
        vis_raw_predict_object[pred_egoleft_lanes[0], pred_egoleft_lanes[1], 2] = 255
        gt_raw_object[gt_egoleft_lanes[0], gt_egoleft_lanes[1], 0] = 0
        gt_raw_object[gt_egoleft_lanes[0], gt_egoleft_lanes[1], 1] = 255
        gt_raw_object[gt_egoleft_lanes[0], gt_egoleft_lanes[1], 2] = 255

        # Visualize EgoRight Lane
        vis_predict_object[pred_egoright_lanes[0], pred_egoright_lanes[1], 0] = 255
        vis_predict_object[pred_egoright_lanes[0], pred_egoright_lanes[1], 1] = 0
        vis_predict_object[pred_egoright_lanes[0], pred_egoright_lanes[1], 2] = 200
        gt_object[gt_egoright_lanes[0], gt_egoright_lanes[1], 0] = 255
        gt_object[gt_egoright_lanes[0], gt_egoright_lanes[1], 1] = 0
        gt_object[gt_egoright_lanes[0], gt_egoright_lanes[1], 2] = 200

        vis_raw_predict_object[pred_egoright_lanes[0], pred_egoright_lanes[1], 0] = 255
        vis_raw_predict_object[pred_egoright_lanes[0], pred_egoright_lanes[1], 1] = 0
        vis_raw_predict_object[pred_egoright_lanes[0], pred_egoright_lanes[1], 2] = 200
        gt_raw_object[gt_egoright_lanes[0], gt_egoright_lanes[1], 0] = 255
        gt_raw_object[gt_egoright_lanes[0], gt_egoright_lanes[1], 1] = 0
        gt_raw_object[gt_egoright_lanes[0], gt_egoright_lanes[1], 2] = 200

        # Visualize Other Lanes
        vis_predict_object[pred_other_lanes[0], pred_other_lanes[1], 0] = 0
        vis_predict_object[pred_other_lanes[0], pred_other_lanes[1], 1] = 255
        vis_predict_object[pred_other_lanes[0], pred_other_lanes[1], 2] = 145
        gt_object[gt_other_lanes[0], gt_other_lanes[1], 0] = 0
        gt_object[gt_other_lanes[0], gt_other_lanes[1], 1] = 255
        gt_object[gt_other_lanes[0], gt_other_lanes[1], 2] = 145

        vis_raw_predict_object[pred_other_lanes[0], pred_other_lanes[1], 0] = 0
        vis_raw_predict_object[pred_other_lanes[0], pred_other_lanes[1], 1] = 255
        vis_raw_predict_object[pred_other_lanes[0], pred_other_lanes[1], 2] = 145
        gt_raw_object[gt_other_lanes[0], gt_other_lanes[1], 0] = 0
        gt_raw_object[gt_other_lanes[0], gt_other_lanes[1], 1] = 255
        gt_raw_object[gt_other_lanes[0], gt_other_lanes[1], 2] = 145

        # Alpha blended visualization
        prediction_vis = cv2.addWeighted(vis_predict_object, \
            alpha, self.perspective_image, 1 - alpha, 0)    

        gt_vis = cv2.addWeighted(gt_object, \
            alpha, self.perspective_image, 1 - alpha, 0)

        # FOR NORMAL VIS
        
        # Prediction
        axs_seg[0].set_title('Prediction',fontweight ="bold") 
        axs_seg[0].imshow(prediction_vis)
        # Ground Truth
        axs_seg[1].set_title('Ground Truth',fontweight ="bold") 
        axs_seg[1].imshow(gt_vis)
        # Save figure to Tensorboard
        if(is_train):
            self.writer.add_figure("Train (Seg)", fig_lane_seg, global_step = (log_count))
        else:
            fig_lane_seg.savefig(vis_path + '_seg.png')

        plt.close(fig_lane_seg)

        # FOR RAW VIS

        # Prediction
        axs_raw_seg[0].set_title('Prediction (RAW)',fontweight ="bold") 
        axs_raw_seg[0].imshow(vis_raw_predict_object)
        # Ground Truth
        axs_raw_seg[1].set_title('Ground Truth (RAW)',fontweight ="bold") 
        axs_raw_seg[1].imshow(gt_raw_object)
        # Save figure to Tensorboard
        if(is_train):
            self.writer.add_figure("Train (Seg) RAW", fig_raw_lane_seg, global_step = (log_count))
        else:
            fig_raw_lane_seg.savefig(vis_path + '_seg.png')

        plt.close(fig_raw_lane_seg)

    # Log validation loss for each dataset to TensorBoard
    def log_validation_dataset(self, dataset, validation_loss_dataset_total, log_count):
         self.writer.add_scalar(f"{dataset} (Validation)", validation_loss_dataset_total, log_count)

    # Log overall validation loss across all datasets to TensorBoard
    def log_validation_overall(self, overall_val_score, log_count):
        self.writer.add_scalar("Overall (Validation)", overall_val_score, log_count)
