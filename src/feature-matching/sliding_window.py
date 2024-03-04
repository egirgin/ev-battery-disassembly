import numpy as np

def sliding_window(image, step_size, window_size, rotations=[0, 90]): #45, 90
    for rotation in rotations:
        rotated_window_size = (window_size[1], window_size[0]) if rotation % 180 == 90 else window_size
        for y in range(0, image.shape[0] - rotated_window_size[1] + 1, step_size[1]):
            for x in range(0, image.shape[1] - rotated_window_size[0] + 1, step_size[0]):
                window = image[y:y + rotated_window_size[1], x:x + rotated_window_size[0]]
                if rotation != 0:
                    window = np.rot90(window, k=rotation // 45)
                yield (x, y, window, rotation)
