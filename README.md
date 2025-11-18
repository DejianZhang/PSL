# Physics-guided Self-supervised Learning for Non-line-of-sight Imaging Code & Datasets

This repository contains code for the paper _Physics-guided Self-supervised Learning for Non-line-of-sight Imaging_ by Ziyang Chen,Jumin Qiu,Zhongyun Chen,Dejian Zhang,Tongbiao Wang,Qiegen Liu, and Tianbao Yu.

## Data

### "airplane" and "table" 
- "airplane": video-confocal-gray-full-10.mp4
- "table": video-confocal-gray-full-4.mp4
- Description: TOF data of Lambertian objects placed at a distance of approximately 1m from the wall. 
- Resolution: 512 x 512
- Scanned Area: 2 m x 2 m planar wall
- Acquisition method: Simulation

## Code

### net
The code includes a self-supervised main architecture: an U-net network based on 3D convolution and 2D convolution, differentiable forward light transport model, and tagless data reading class.

### train_ss
This code implements physics-guided self-supervised learning for non-line-of-sight (NLOS) imaging on a single data sample. The specific workflow is as follows: First, read the Time-of-Flight (TOF) data of the airplane target from the video file video-confocal-gray-full-10.mp4 and downsample it to a resolution of [64,64]. Meanwhile, read the 3D voxel data constructed via the inverse light transport model, which serves as a reference for training.
In the training phase, self-supervised learning is initiated: input the downsampled TOF data into the U-net network to obtain the network-predicted 3D voxel. Subsequently, process this predicted 3D voxel using the differentiable forward light transport model to generate predicted TOF data. Based on (the real TOF data, predicted TOF data,) and (predicted 3D voxel,  Reference voxel,) (3D Tenengrad metric of predicted 3D voxel), we calculate the reverse consistency loss, forward consistency loss, and voxel regularization loss respectively. After obtaining the total loss through weighted summation, backpropagate it to the U-net network to complete the iterative training and optimization of the model.
