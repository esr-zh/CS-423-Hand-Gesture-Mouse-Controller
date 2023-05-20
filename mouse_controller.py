import math
from keras_preprocessing.image import img_to_array
import numpy as np
import cv2
import pyautogui
import tensorflow as tf
import time

pyautogui.FAILSAFE = False

# Constants
FRAME_WIDTH = 2000
FRAME_HEIGHT = 1160  # 580
FRAME_RATE = 15  # new constant
MODEL_PATH = 'saved_models/hand_classifier1.h5'
WEIGHT = 0.2
ROI_START_X = 200
ROI_END_X = 600
ROI_START_Y = 600  # 300
ROI_END_Y = 1000
THRESHOLD = 17  # BINARY threshold
BLUR_VAL = 41  # GaussianBlur parameter

# Initialize video capture
video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
# video_capture.set(cv2.CAP_PROP_FPS, FRAME_RATE)  # set frame rate

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
num_frames = 0
background_image = None
last_defect_point = None
moving_mode = False
selected_defect_index = 0

actions = {
    0: "click_right",
    1: "click_left",
    2: "move",
    3: "double_click",
    4: "scroll_up",
    5: "scroll_down",
}


def update_background(img, weight):
    global background_image
    if background_image is None:
        background_image = img.copy().astype('float')
        return
    cv2.accumulateWeighted(img, background_image, weight)



def segment(img, thres=THRESHOLD):
    global background_image
    diff = cv2.absdiff(background_image.astype('uint8'), img)
    _, thresh = cv2.threshold(diff, thres, 255, cv2.THRESH_BINARY)
    cont, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cont) == 0:
        return
    else:
        seg = max(cont, key=cv2.contourArea)
    return thresh, seg


def calculate_fingers(res, drawing):
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    hull_points = cv2.convexHull(res, returnPoints=True)
    cv2.drawContours(drawing, [hull_points + (ROI_START_Y, ROI_START_X)], -1, (255, 255, 255), 2)  # Draw the hull
    defects = None
    if len(hull) > 3:
        try:
            defects = cv2.convexityDefects(res, hull)
        except:
            defects = None
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            cnt = 0
            valid_defect_points = []  # List to store valid defect points
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                far = (far[0] + ROI_START_Y, far[1] + ROI_START_X)
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    valid_defect_points.append(far)
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt, valid_defect_points, res
    return False, 0, defects, res


def get_prediction(img):
    for_gesture = cv2.resize(img, (64, 64))
    x = img_to_array(for_gesture)
    x = x / 255.0
    x = x.reshape((1,) + x.shape)
    gesture = str(actions[np.argmax(model.predict(x))])
    return gesture


while video_capture.isOpened():
    ret, frame = video_capture.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[ROI_START_X:ROI_END_X, ROI_START_Y:ROI_END_Y]  # [100:300, 300:500]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_VAL, BLUR_VAL), 0)

        if num_frames < 30:
            update_background(gray, WEIGHT)
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (ROI_START_Y, ROI_START_X)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                for cnt in contours:
                    if cv2.contourArea(cnt) > 5000:
                        gesture = get_prediction(thresholded)

                        finished, num_defects, valid_defects, res = calculate_fingers(segmented, clone)
                        if gesture == "move":
                            if num_defects >= 1:
                                if selected_defect_index >= len(valid_defects):
                                    selected_defect_index = 0  # Fall back to the first defect point
                                far = valid_defects[selected_defect_index]
                                cv2.circle(clone, far, 8, [0, 255, 0], -1)  # Draw the circle in green

                                if moving_mode:  # If already in moving mode
                                    if last_defect_point is not None:
                                        move_x = far[0] - last_defect_point[0]
                                        move_y = far[1] - last_defect_point[1]
                                        # Get the current mouse position
                                        current_mouse_x, current_mouse_y = pyautogui.position()
                                        # Calculate the new mouse position
                                        new_mouse_x = current_mouse_x + move_x * 6
                                        new_mouse_y = current_mouse_y + move_y * 6
                                        # Apply the movement to the mouse
                                        pyautogui.moveTo(new_mouse_x, new_mouse_y,
                                                         duration=0.0)  # adjust duration for speed
                                else:  # If not in moving mode
                                    moving_mode = True  # Enter moving mode
                                # Update the last defect point
                                last_defect_point = far
                                selected_defect_index = valid_defects.index(far)
                            else:
                                moving_mode = False  # Exit moving mode

                        elif gesture == "click_right":
                            pyautogui.click(button='right')
                            # time.sleep(0.5)

                        elif gesture == "click_left":
                            pyautogui.click(button='left')
                            # time.sleep(0.5)

                        elif gesture == "double_click":
                            pyautogui.doubleClick()
                            # time.sleep(0.5)

                        elif gesture == "scroll_up":
                            pyautogui.scroll(50)
                            # time.sleep(0.5)

                        elif gesture == "scroll_down":
                            pyautogui.scroll(-50)
                            # time.sleep(0.5)

                        cv2.putText(clone, gesture, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.rectangle(clone, (ROI_START_Y, ROI_START_X), (ROI_END_Y, ROI_END_X), (0, 255, 0), 2)
        num_frames += 1
        cv2.putText(clone, "Place your hand in the square", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('frame', clone)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video_capture.release()
cv2.destroyAllWindows()
