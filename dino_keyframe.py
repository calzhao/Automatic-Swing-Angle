import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn as nn
from math import atan2, degrees
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import os


class DinoCNN(nn.Module):
    def __init__(self, input_dim, num_filters=64, kernel_size=3):
        super(DinoCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(num_filters)  # Batch Normalization
        self.relu1 = nn.ReLU()

        # Second convolutional layer (deeper)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.relu2 = nn.ReLU()

        # Third convolutional layer (deeper)
        self.conv3 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(num_filters)
        self.relu3 = nn.ReLU()

        # Fully connected layer to reduce output to a single confidence score per item
        self.fc = nn.Conv1d(in_channels=num_filters, out_channels=1, kernel_size=1)

        # Sigmoid activation for the confidence score between 0 and 1
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x shape: (N, D)

        # Add batch dimension and rearrange the dimensions to fit Conv1d input
        x = x.unsqueeze(0).permute(0, 2, 1)  # Shape: (1, D, N)

        # First convolutional block
        x = self.conv1(x)  # Shape: (1, num_filters, N)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second convolutional block
        x = self.conv2(x)  # Shape: (1, num_filters, N)
        x = self.bn2(x)
        x = self.relu2(x)

        # Third convolutional block
        x = self.conv3(x)  # Shape: (1, num_filters, N)
        x = self.bn3(x)
        x = self.relu3(x)

        # Fully connected layer to output confidence scores for each sequence item
        x = self.fc(x)  # Shape: (1, 1, N)

        # Remove extra channel dimension
        x = x.squeeze(1)  # Shape: (1, N)

        # Apply sigmoid for confidence score between 0 and 1
        confidence_scores = x.squeeze(0)  # Shape: (N,)

        return confidence_scores


def read_video_and_fps(video_path, resize_dim=(224, 224)):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return None, None

    # Get frames per second (FPS)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Read video frames
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Resize the frame to 224x224
        resized_frame = cv2.resize(frame, resize_dim)
        frames.append(resized_frame)

    # Release the video capture object
    video.release()

    return np.stack(frames), fps


def localize_snippet(video, fps, use_s3d=False):
    if use_s3d:
        raise NotImplementedError('Localize snippet with S3D is not supported right now!')

    # video shape should be [F, H, W, 3]
    if fps < 31:
        return video[60:150]
    else:
        return video[120:300]


def load_vid(path):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    cap = cv2.VideoCapture(path)

    # Append frames to list
    frames = []

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Store the resulting frame
            frames.append(frame)
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    frames = np.stack(frames)

    return frames


def preprocess_image_sequence(image_sequence: torch.Tensor, target_size=(224, 224)):
    """
    Preprocess a sequence of torch tensor images for DINO v2-like models.
    1. Convert from uint8 [0, 255] to float32 [0, 1].
    2. Resize each image to the target size.
    3. Normalize each image using ImageNet mean and std.

    Args:
    - image_sequence (torch.Tensor): Input image sequence as a torch tensor
                                     of shape (num_image, H, W, 3) in uint8 format.
    - target_size (tuple): Desired output size for each image (height, width).

    Returns:
    - torch.Tensor: Preprocessed image sequence as a PyTorch tensor
                    of shape (num_image, 3, target_size[0], target_size[1]).
    """

    num_images = image_sequence.shape[0]
    processed_images = []

    # ImageNet normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    for i in range(num_images):
        image = image_sequence[i]  # Get each image (H, W, 3)

        # Convert from uint8 to float32 and scale to [0, 1]
        image = image.to(torch.float32) / 255.0

        # Permute dimensions from (H, W, 3) to (3, H, W)
        image = image.permute(2, 0, 1)

        # Resize the image to target size
        image = TF.resize(image, target_size, antialias=True)

        # Normalize using ImageNet mean and std
        image = TF.normalize(image, mean=mean, std=std)

        # Add the processed image to the list
        processed_images.append(image)

    # Stack all processed images into a single tensor (num_image, 3, target_size[0], target_size[1])
    processed_images_tensor = torch.stack(processed_images)

    return processed_images_tensor


def extract_dino_features(video_snippet, model, device):
    """
    Args:
        video_snippet: [3*fps, H, W, 3] np array
        model: dino model
        device: device to run dino

    Returns: [3*fps, d] a dino feature sequence tensor
    """
    video_snippet_tensor = preprocess_image_sequence(torch.from_numpy(video_snippet)).to(device)
    model.eval()
    with torch.no_grad():
        feature_sequence = model(video_snippet_tensor).detach()

    return feature_sequence


def keyframe_detection(video_path, dino_model, s2f_model, device, use_s3d=False, resize_dim=(224, 224)):
    """
    Args:
        video_path: video path
        dino_model: dino model to extract image feature
        s2f_model: model that predict snippet dino feature to keyframe
        device:
        use_s3d: whether use s3d to localize the snippet or not
        resize_dim: resized image size for dino input

    Returns:
        keyframe: the keyframe image in original size [H, W, 3]
    """
    cur_vid_resized, cur_fps = read_video_and_fps(video_path, resize_dim)
    cur_vid_snippet_array = localize_snippet(cur_vid_resized, cur_fps, use_s3d)
    cur_vid_dino_tensor = extract_dino_features(cur_vid_snippet_array, dino_model, device)
    with torch.no_grad():
        cur_vid_kf_idx = torch.argmax(s2f_model(cur_vid_dino_tensor)).squeeze().item()

    org_video_snippet = localize_snippet(load_vid(video_path), cur_fps, use_s3d)
    cur_keyframe = org_video_snippet[cur_vid_kf_idx]

    keyframe_idx = cur_vid_kf_idx
    if cur_fps < 31:
        keyframe_idx += 60
    else:
        keyframe_idx += 120

    return cur_keyframe, keyframe_idx

def process_predictions(predictions, vid_shape):
    fields = predictions['instances'].get_fields()
    pred_masks = fields['pred_masks'].to('cpu').numpy()
    pred_scores = fields['scores'].to('cpu').numpy()
    pred_boxes = fields['pred_boxes'].tensor.to('cpu').numpy()
    # pred_classes = fields['pred_classes'].to('cpu').numpy()

    if len(pred_scores) > 1:
        idx = np.argmax(pred_scores)
        pred_masks = pred_masks[idx]
        pred_boxes = pred_boxes[idx]
        pred_scores = pred_scores[idx]
    elif len(pred_scores) == 0:
        pred_scores = np.array([0])
        pred_boxes = np.zeros((4,))
        # pred_masks = np.zeros((720, 1280))
        pred_masks = np.zeros(vid_shape)

    return pred_scores.squeeze(), pred_boxes.squeeze(), pred_masks.squeeze()

def get_prediction(predictor, keyframe):
    predictions, visualized_output = predictor.run_on_image(keyframe)
    H, W, _ = keyframe.shape
    pred_scores, pred_boxes, pred_masks = process_predictions(predictions, (H, W))

    return {'boxes':pred_boxes, 'scores':pred_scores, 'masks':pred_masks.astype(np.uint8)}

def angle_estimation(mask):
    kernel = np.ones((2, 2), np.uint8)
    cleaned_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    y_coords, x_coords = np.where(mask)
    points = np.column_stack((x_coords, y_coords))

    if len(points) == 0:
        return -1, 0

    pca = PCA(n_components=2)
    pca.fit(points)
    center = pca.mean_
    first_component = pca.components_[0]

    angle = atan2(first_component[1], first_component[0])
    y_coords, x_coords = np.where(mask)
    m = np.tan(angle)
    b = center[1] - m * center[0]  # y = mx + b => b = y - mx
    predicted_y = m * x_coords + b
    r_squared = r2_score(y_coords, predicted_y)

    orientation_angle = degrees(angle)
    orientation_angle = (orientation_angle + 360) % 180

    est_angle = orientation_angle
    if est_angle > 90:
        est_angle = 180 - est_angle

    return est_angle, r_squared


if __name__ == '__main__':
    vid_path = './videos/amateur_videos_2/0008FC50-1B72-4CF7-88CE-09498211F0FD.mp4'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
    s2f_model = DinoCNN(input_dim=1536).to(device).eval()
    s2f_model.load_state_dict(torch.load('./dino/cnn_deep_dinov2_mix.pth'))

    keyframe, keyframe_idx = keyframe_detection(vid_path, dino_model, s2f_model, device)
    print(keyframe_idx, keyframe.shape)

    # # visualize for demo
    # import matplotlib.pyplot as plt
    # keyframe_rgb = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)
    # # Plot the frame
    # plt.imshow(keyframe_rgb)
    # plt.axis('off')  # Hide axis for better visualization
    # plt.title(f"KeyFrame for video {vid_path.split('/')[-1]}")
    # plt.show()