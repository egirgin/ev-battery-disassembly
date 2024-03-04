import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def visualize_results(larger_image_path, positions, rotations, w_size):
    # Load larger image for visualization
    larger_image_vis = cv2.imread(larger_image_path)

    # Collect bounding boxes
    boxes = []
    # Display top similar windows as bounding boxes on the larger image
    for position, rotation in zip(positions, rotations):
        x, y = position
        window_size = w_size  # Adjust window size as needed
        rotated_window_size = (window_size[1], window_size[0]) if rotation % 180 == 90 else window_size
        boxes.append([x, y, x + rotated_window_size[0], y + rotated_window_size[1]])
        cv2.rectangle(larger_image_vis, (x, y), (x + rotated_window_size[0], y + rotated_window_size[1]), (0, 255, 0), 2)

    # Display the larger image with bounding boxes
    plt.imshow(cv2.cvtColor(larger_image_vis, cv2.COLOR_BGR2RGB))
    plt.title('Top Similar Windows with Bounding Boxes')
    plt.axis('off')
    plt.show()


def visualize_results(larger_image_path, positions, rotations, w_size):
    # Load larger image for visualization
    larger_image_vis = cv2.imread(larger_image_path)

    # Collect bounding boxes
    boxes = []
    for position, rotation in zip(positions, rotations):
        x, y = position
        window_size = w_size  # Adjust window size as needed
        rotated_window_size = (window_size[1], window_size[0]) if rotation % 180 == 90 else window_size
        boxes.append([x, y, x + rotated_window_size[0], y + rotated_window_size[1]])

    # Convert boxes to numpy array
    boxes = np.array(boxes)

    # Apply non-max suppression
    nms_boxes = non_max_suppression(boxes, 0.3)  # Adjust overlap threshold as needed

    # Draw bounding boxes on the larger image
    for (startX, startY, endX, endY) in nms_boxes:
        cv2.rectangle(larger_image_vis, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)

    # Display the larger image with bounding boxes
    plt.imshow(cv2.cvtColor(larger_image_vis, cv2.COLOR_BGR2RGB))
    plt.title('Top Similar Windows with Bounding Boxes (NMS)')
    plt.axis('off')
    plt.show()


def cluster_similarities(similarities, positions, rotations):
    # Sorting similarities and positions based on similarity scores
    sorted_indices = np.argsort(similarities)
    sorted_similarities = [similarities[i] for i in sorted_indices]
    sorted_positions = [positions[i] for i in sorted_indices]
    sorted_rotations = [rotations[i] for i in sorted_indices]

    # Apply KMeans clustering to select top similar ones
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(np.array(sorted_similarities).reshape(-1, 1))
    ultimate_cluster = kmeans.cluster_centers_.argmax(axis=0)
    
    top_indices = []
    for id, sim in enumerate(sorted_similarities):
        if kmeans.labels_[id] == ultimate_cluster:
            top_indices.append(id)

    cluster_positions = [sorted_positions[i] for i in top_indices]
    cluster_rotations = [sorted_rotations[i] for i in top_indices]
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    # Plot similarity scores
    plt.figure(figsize=(10, 6))
    for id, sim in enumerate(sorted_similarities):
        plt.scatter(id, sim, color=colors[kmeans.labels_[id]%10])
    #plt.plot(sorted_similarities, marker='o', linestyle='-', color='b')
    plt.title('Similarity Scores Sorted by Ascending Order')
    #plt.xlabel('')
    plt.ylabel('Similarity Score')
    plt.grid(True)
    plt.show()

    return cluster_positions, cluster_rotations

def non_max_suppression(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []

    # Initialize list to keep track of picked indices
    pick = []

    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute areas of bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by their bottom-right y-coordinate
    idxs = np.argsort(y2)

    # Iterate over the indices of the bounding boxes
    while len(idxs) > 0:
        # Get the index of the last bounding box
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y)-coordinates for the start of
        # the bounding box and the smallest (x, y)-coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding boxes
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                                np.where(overlap > overlap_thresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick]