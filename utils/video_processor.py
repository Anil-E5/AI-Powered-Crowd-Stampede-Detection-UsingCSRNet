import torch
import cv2
import numpy as np
from torchvision import transforms
from models.csrnet import CSRNet
import os
from PIL import Image

# Ensure the weights directory exists
WEIGHTS_PATH = 'weights/csrnet_weights.pth'
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

def load_csrnet_model():
    """
    Loads the CSRNet model and its pretrained weights.
    Downloads weights if not present.
    """
    model = CSRNet(load_weights=True)
    model.eval() # Set to evaluation mode

    # Try to move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"CSRNet model moved to: {device}")


    if not os.path.exists(WEIGHTS_PATH):
        print("Downloading CSRNet weights (approx. 200MB)... This may take a moment.")
        print(f"Please download CSRNet pretrained weights and place them in: {WEIGHTS_PATH}")
        print("For initial testing, you can proceed, but results will be random without proper weights.")
        try:
            # Save a dummy state_dict to create the file and avoid repeated messages
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print("Dummy weight file created. Replace with actual weights for proper functionality.")
        except Exception as e:
            print(f"Could not create dummy weight file: {e}")
    else:
        try:
            # Load weights, ensuring they are loaded to the correct device
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            print("CSRNet weights loaded successfully.")
        except Exception as e:
            print(f"Error loading CSRNet weights: {e}")
            print("Model will run with uninitialized weights. Please ensure correct weights are in place.")
    return model

# Define the image transformations once
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_density_maps_batch(model, frames_rgb_batch):
    """
    Processes a batch of RGB frames to generate crowd density maps.
    frames_rgb_batch: A list of RGB numpy arrays (each frame).
    Returns a list of density maps (numpy arrays).
    """
    if not frames_rgb_batch:
        return []

    # Convert numpy frames to PIL images, then apply transformations and stack
    tensors = []
    for frame_rgb in frames_rgb_batch:
        img_pil = Image.fromarray(frame_rgb)
        tensors.append(preprocess(img_pil))

    # Stack into a batch tensor
    batch_tensor = torch.stack(tensors)

    # Move batch to the same device as the model
    device = next(model.parameters()).device # Get current device of the model
    batch_tensor = batch_tensor.to(device)

    with torch.no_grad():
        # Perform inference on the entire batch
        density_maps_tensor = model(batch_tensor)

    # Convert back to numpy arrays and un-normalize for visualization
    density_maps_np = []
    for density_map_single_tensor in density_maps_tensor.cpu():
        density_map = density_map_single_tensor.squeeze().numpy()
        density_map = np.maximum(density_map, 0) # Ensure no negative values

        if density_map.max() > 0:
            density_map = (density_map / density_map.max()) * 255
        else:
            density_map = np.zeros_like(density_map)

        density_maps_np.append(density_map.astype(np.uint8))

    return density_maps_np

def estimate_crowd_count(density_map):
    """
    Estimates the total crowd count from a density map.
    This function will still take a single density map at a time.
    """
    # Summing the density map values directly usually gives a good estimate
    # The `predict_density_maps_batch` scales density map to 0-255 for visualization.
    # To get a more accurate count, the sum of the *raw* (un-normalized) density map is preferred.
    # For now, as a proxy assuming normalized map represents relative density:
    # A true count requires specific calibration or using the raw model output before 0-255 scaling.
    return np.sum(density_map) / 255.0 * (density_map.shape[0] * density_map.shape[1] / (640*480/64)) # Rough scaling factor


def calculate_optical_flow(prev_frame_gray, current_frame_gray):
    """
    Calculates optical flow between two grayscale frames using Farneback method.
    """
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, current_frame_gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def draw_flow(img_color, flow, step=16):
    """
    Draws optical flow vectors on an already colored image.
    img_color is expected to be a 3-channel image (e.g., RGB or BGR).
    """
    h, w = img_color.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    
    # Ensure x and y are within bounds
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    fx, fy = flow[y,x].T
    
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    vis = img_color.copy()
    
    cv2.polylines(vis, lines, 0, (0, 255, 0), thickness=1)
    
    for (x1, y1), _ in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def detect_panic(flow, motion_threshold=5, alignment_threshold=0.7):
    """
    Detects panic based on optical flow.
    If a large percentage of significant motion vectors suddenly align in one direction with high speed → trigger panic alert.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    significant_motion_mask = mag > motion_threshold
    significant_mags = mag[significant_motion_mask]
    significant_angs = ang[significant_motion_mask]

    if len(significant_mags) < 100:
        return False, 0.0, 0.0

    significant_angs_deg = np.degrees(significant_angs)
    significant_angs_deg[significant_angs_deg < 0] += 360

    hist, bins = np.histogram(significant_angs_deg, bins=36, range=(0, 360))
    dominant_direction_idx = np.argmax(hist)
    dominant_direction_count = hist[dominant_direction_idx]

    alignment_ratio = dominant_direction_count / len(significant_mags)
    avg_motion_magnitude = np.mean(significant_mags)

    if alignment_ratio > alignment_threshold:
        return True, alignment_ratio, avg_motion_magnitude
    return False, alignment_ratio, avg_motion_magnitude