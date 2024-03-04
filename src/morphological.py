from dis import show_code
import cv2, math
import os, sys, time
import numpy as np
import random
from PIL import Image, ImageDraw


def im_show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def paste_image(background, foreground):
    # Random location for pasting the foreground image
    x_offset = random.randint(0, background.shape[1] - foreground.shape[1])
    y_offset = random.randint(0, background.shape[0] - foreground.shape[0])

    # Create a mask for the foreground image
    mask = np.where(foreground != 0, 255, 0).astype('uint8')
    
    alpha = foreground[:, :, 3] / 255.0

    for c in range(0, 3):
        background[y_offset:y_offset+foreground.shape[0], x_offset:x_offset+foreground.shape[1], c] = \
            alpha * foreground[:, :, c] + \
            (1 - alpha) * background[y_offset:y_offset+foreground.shape[0], x_offset:x_offset+foreground.shape[1], c]


    return background


def read_img(path, angle=0, show=False, occlusion=False, hazard=False):
    img = cv2.imread(path)

    img = cv2.resize(img, (1080, 720))

    if show:
        im_show(img)

    if angle != 0:
        # Get the image center coordinates
        center = (img.shape[1] // 2, img.shape[0] // 2)

        # Perform the rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Determine the size of the rotated image
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        bound_w = int(img.shape[1] * abs_cos + img.shape[0] * abs_sin)
        bound_h = int(img.shape[1] * abs_sin + img.shape[0] * abs_cos)

        # Adjust the rotation matrix to ensure the entire image is visible
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]

        # Apply rotation without cropping onto the white background
        img = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h), borderValue=(250, 250, 250))

    if occlusion:
        occlusion_img = cv2.imread("./imgs/occlusion-hand.png",  cv2.IMREAD_UNCHANGED)
        img = paste_image(img, occlusion_img)

    if hazard:
        flame_img = cv2.imread("./imgs/flame.png",  cv2.IMREAD_UNCHANGED)
        img = paste_image(img, flame_img)
    return img


def GW_white_balance(img):
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image


def find_orange(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for orange color in HSV
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    
    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    
    # Bitwise-AND mask and original image
    #orange_parts = cv2.bitwise_and(image, image, mask=mask)
    
    return mask


def find_red(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the red color range
    #lower_red = np.array([0, 100, 100])
    #upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([150, 100, 0])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the HSV image to get only red colors
    #mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the masks
    mask = mask2 #cv2.bitwise_or(mask1, mask2)

    return mask

def find_cable_centers(img):
    overlay = img.copy()

    # Create a mask to threshold the black color
    img = find_orange(img)

    img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=5)
    
    img = cv2.erode(img, np.ones((5, 5), np.uint8), iterations=7)
    
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=5)

    #im_show(img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = np.zeros((len(contours), 2), dtype=int)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        centers[i] = [x+w//2, y+h//2]


    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    drawing[img == 255] = [255, 0, 0]

    alpha = 0.5  # Transparency coefficient

    cv2.addWeighted(overlay, alpha, drawing, 1 - alpha, 0, drawing)

    #im_show(drawing)

    return centers


def find_module_centers(img):
    overlay = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds for the black color range in HSV
    lower_black = np.array([0, 0, 0])  # Lower bound for black
    upper_black = np.array([255, 255, 110])  # Upper bound for black

    # Create a mask to threshold the black color
    img = cv2.inRange(hsv, lower_black, upper_black)

    img = 255 - img 

    #im_show(img)

    img = cv2.dilate(img, np.ones((9, 9), np.uint8), iterations=8)

    #im_show(img)

    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=10)

    #im_show(img)

    img = 255 - img

    #im_show(img)

    """
    img = cv2.GaussianBlur(img, (5, 5), 5) 

    img = cv2.Canny(img, 50, 100)
    im_show(img)

    img = cv2.dilate(img, np.ones((9, 9), np.uint8), iterations=7)
    
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    img = 255 - img

    img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=7)
    """

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000] # 30000 if not cropped

    centers = np.zeros((len(contours), 2), dtype=int)
    import random as rng
    # Draw contours
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        x, y, w, h = cv2.boundingRect(contours[i])
        centers[i] = [x+w//2, y+h//2]
        cv2.rectangle(drawing, (x, y), (x + w, y + h), color, cv2.FILLED)

    alpha = 0.5  # Transparency coefficient

    # Apply alpha blending
    cv2.addWeighted(drawing, alpha, overlay, 1 - alpha, 0, drawing)

    #im_show(drawing)

    return centers, drawing

def hazard_detection(img):

    overlay = img.copy()

    # Create a mask to threshold the black color
    img = find_red(img)
    
    img = cv2.erode(img, np.ones((5, 5), np.uint8), iterations=1)
    
    img = cv2.dilate(img, np.ones((7, 7), np.uint8), iterations=10)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    hazard_center_offset = 250
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(drawing, (x, y+hazard_center_offset), (x + w, y + hazard_center_offset + h), (0,0,255))
        drawing = cv2.putText(drawing, 'Overheat', (x, y+hazard_center_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA) 
        fire_location =  [x+w//2, y+h//2]

    alpha = 0.5  # Transparency coefficient

    cv2.addWeighted(overlay, alpha, drawing, 1 - alpha, 0, drawing)

    im_show(drawing)

    return fire_location


def find_closest_pairs(list1, list2):
    closest_pairs = []
    

    for point1 in list1:
        min_distance = float('inf')
        for point2 in list2:
            distance = np.linalg.norm(point1 - point2)
            if distance < min_distance:
                min_distance = distance
                closest_pair = (point1, point2)

        closest_pairs.append(closest_pair)

    return closest_pairs

def draw_arrow(image, start_point, end_point, color=(0, 0, 255), thickness=2, arrow_length=20, arrow_angle=np.pi/6):
    # Convert coordinates to tuples
    start_point = tuple(start_point)
    end_point = tuple(end_point)

    # Draw line
    cv2.line(image, start_point, end_point, color, thickness)

    # Calculate arrow points
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    angle = np.arctan2(dy, dx)

    x1 = end_point[0] - arrow_length * np.cos(angle - arrow_angle)
    y1 = end_point[1] - arrow_length * np.sin(angle - arrow_angle)
    x2 = end_point[0] - arrow_length * np.cos(angle + arrow_angle)
    y2 = end_point[1] - arrow_length * np.sin(angle + arrow_angle)

    # Draw arrow head
    cv2.line(image, end_point, (int(x1), int(y1)), color, thickness)
    cv2.line(image, end_point, (int(x2), int(y2)), color, thickness)

img_list = os.listdir("./imgs")


for im_filename in img_list:
    if not "cropped" in im_filename:
        continue

    hazard_flag = False
    occlusion_flag = False
    rotation_angle = 0
    img = read_img("./imgs/{}".format(im_filename), angle=rotation_angle, occlusion=occlusion_flag, show=False, hazard=hazard_flag)

    img = GW_white_balance(img)

    im_show(img)

    start = time.time()


    if hazard_flag:
        hazard_detection(img)
    else:

        cable_centers = find_cable_centers(img)

        module_centers, img = find_module_centers(img)

        pairs = find_closest_pairs(module_centers, cable_centers)
        for pair in pairs:
            #print(pair)
            draw_arrow(image=img, start_point=pair[0], end_point=pair[1])

        print(time.time() - start)

        im_show(img)






    
