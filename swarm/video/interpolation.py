import torch
import torchvision
import torchvision.transforms as transforms
from .MotionEstimationNet import MotionEstimationNet
import numpy as np
import cv2


def interpolate_frames(frame1_filename, frame2_filename, num_intermediate_frames):
    """Interpolate between two frames.

    Args:
        frame1: First frame.
        frame2: Second frame.
        num_intermediate_frames: Number of intermediate frames to generate.

    Returns:
        List of intermediate frames.
    """
    frame1 = cv2.imread(frame1_filename)
    frame2 = cv2.imread(frame2_filename)
    intermediate_frames = []
    for i in range(num_intermediate_frames):
        alpha = i / (num_intermediate_frames - 1)
        intermediate_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        intermediate_frames.append(intermediate_frame)

    return intermediate_frames


def interpolate_frame_optical_flow(frame1_filename, frame2_filename):
    # Load the two frames
    frame1 = cv2.imread(frame1_filename)
    frame2 = cv2.imread(frame2_filename)

    # Convert the frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Estimate the motion vectors using the block-matching algorithm
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Use the estimated motion vectors to compensate for the motion in the frames
    height, width = frame1.shape[:2]
    compensated_frame = np.zeros_like(frame1)
    for y in range(height):
        for x in range(width):
            dx, dy = flow[y, x]
            x_comp = int(x + dx)
            y_comp = int(y + dy)
            if x_comp >= 0 and x_comp < width and y_comp >= 0 and y_comp < height:
                compensated_frame[y, x] = frame2[y_comp, x_comp]

    # Save the compensated frame
    return compensated_frame


def interpolate_frame_motion_stimation(frame1_filename, frame2_filename):
    # Load the two frames as PyTorch tensors
    frame1 = transforms.ToTensor()(cv2.imread(frame1_filename))
    frame2 = transforms.ToTensor()(cv2.imread(frame2_filename))

    # Create the motion estimation network
    motion_estimation_net = MotionEstimationNet()

    # Estimate the motion vectors using the network
    motion_vectors = motion_estimation_net(torch.cat((frame1, frame2), dim=0))
    motion_vectors = motion_vectors.view(2, -1, 2)

    # Use the estimated motion vectors to compensate for the motion in the frames
    height, width = frame1.shape[1:]
    # compensated_frame = torch.zeros_like(frame1)
    grid_x, grid_y = torch.meshgrid(torch.arange(width), torch.arange(height))
    grid_x = grid_x.float() + motion_vectors[0, 0, 0]
    grid_y = grid_y.float() + motion_vectors[0, 0, 1]
    grid_x = grid_x.clamp(0, width - 1)
    grid_y = grid_y.clamp(0, height - 1)
    grid_x = grid_x.unsqueeze(0).expand(height, width)
    grid_y = grid_y.unsqueeze(0).expand(height, width)
    return torch.nn.functional.grid_sample(
        frame2,
        torch.stack((grid_y, grid_x), dim=0),
        mode="bilinear",
        padding_mode="zeros",
    )
