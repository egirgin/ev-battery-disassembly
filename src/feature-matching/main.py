target_image_path = 'module1.jpg'
larger_image_path = 'pack1.png'

import cv2
import numpy as np
from sliding_window import sliding_window
from feature_matching import calculate_image_similarity_orb, calculate_image_similarity_sift
from visualization import visualize_results, cluster_similarities

def GW_white_balance(img):
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image

target_image = cv2.imread(target_image_path)
img = GW_white_balance(target_image)
target_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load images
larger_image = cv2.imread(larger_image_path)
larger_image = GW_white_balance(larger_image)
larger_image = cv2.cvtColor(larger_image, cv2.COLOR_BGR2GRAY)

#larger_image = cv2.resize(larger_image, (larger_image.shape[0]//2, larger_image.shape[1]//2))

### SELECT DETECTOR

# Initialize ORB detector
orb = cv2.ORB_create()
# Find keypoints and descriptors for the target image
target_keypoints, target_descriptors = orb.detectAndCompute(target_image, None)


# Initialize SIFT detector
#sift = cv2.SIFT_create()
# Find keypoints and descriptors for the target image
#target_keypoints, target_descriptors = sift.detectAndCompute(target_image, None)

# Set window size and step size
window_size = (int(target_image.shape[0]*0.7), int(target_image.shape[1]*0.7)) 
step_size = (20, 20)

# Calculate similarity
similarities, positions, rotations = calculate_image_similarity_orb(target_descriptors, target_keypoints, larger_image, orb, sliding_window, step_size, window_size)
#similarities, positions, rotations = calculate_image_similarity_sift(target_descriptors, target_keypoints, larger_image, sift, sliding_window, step_size, window_size)

cluster_positions, cluster_rotations = cluster_similarities(similarities, positions, rotations)

#sorted_indices = np.argsort(similarities)
#sorted_similarities = [similarities[i] for i in sorted_indices]
#sorted_positions = [positions[i] for i in sorted_indices]
#sorted_rotations = [rotations[i] for i in sorted_indices]

# Visualize results
visualize_results(larger_image_path, cluster_positions, cluster_rotations, w_size=window_size)
