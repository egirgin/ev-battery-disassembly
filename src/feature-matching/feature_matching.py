import cv2

def calculate_image_similarity_orb(target_descriptors, target_keypoints, larger_image, orb, sliding_window, step_size, window_size):
    similarities = []
    positions = []
    rotations = []

    # Slide window over the larger image with different rotations
    for (x, y, window, rotation) in sliding_window(larger_image, step_size, window_size):
        # Find keypoints and descriptors for the window
        window_keypoints, window_descriptors = orb.detectAndCompute(window, None)

        # Initialize brute-force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        if target_descriptors is not None and window_descriptors is not None:
            matches = bf.match(target_descriptors, window_descriptors)

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Calculate similarity
            similarity = sum([match.distance for match in matches]) / len(matches)

            similarities.append(similarity)
            positions.append((x, y))
            rotations.append(rotation)

    return similarities, positions, rotations


import cv2

def calculate_image_similarity_sift(target_descriptors, target_keypoints, larger_image, sift, sliding_window, step_size, window_size):

    similarities = []
    positions = []
    rotations = []

    # Slide window over the larger image with different rotations
    for (x, y, window, rotation) in sliding_window(larger_image, step_size, window_size):
        # Find keypoints and descriptors for the window
        window_keypoints, window_descriptors = sift.detectAndCompute(window, None)

        # Initialize brute-force matcher
        bf = cv2.BFMatcher()

        # Match descriptors
        if target_descriptors is not None and window_descriptors is not None:
            matches = bf.knnMatch(target_descriptors, window_descriptors, k=2)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # Calculate similarity
            similarity = len(good_matches) / max(len(target_descriptors), len(window_descriptors))

            similarities.append(similarity)
            positions.append((x, y))
            rotations.append(rotation)

    return similarities, positions, rotations
