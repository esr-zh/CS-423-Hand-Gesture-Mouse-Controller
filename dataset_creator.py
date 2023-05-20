import cv2
import os

# Constants
WIDTH, HEIGHT = 1000, 480
ROI = (100, 300, 300, 500)  # Coordinates for region of interest
AREA_THRESHOLD = 5000  # Area threshold for hand detection
IMAGE_DIR = './custom_dataset/5'


def run_avg(img, weight, bg):
    """Calculate running average of the background."""
    if bg is None:
        bg = img.copy().astype('float')
    else:
        cv2.accumulateWeighted(img, bg, weight)
    return bg


def segment(img, bg, thres=25):
    """Segment the image from the background."""
    diff = cv2.absdiff(bg.astype('uint8'), img)
    _, threshold = cv2.threshold(diff, thres, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv2.contourArea)
        return threshold, segmented


def main():
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Unable to open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    weight = 0.2
    num_frames = 0
    bg = None
    i = 0

    # Check if the directory to save images exists
    if not os.path.exists(IMAGE_DIR):
        print("Creating directory to save images...")
        os.makedirs(IMAGE_DIR)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                outer_frame = frame.copy()
                (height, width) = frame.shape[:2]
                roi = frame[ROI[0]:ROI[1], ROI[2]:ROI[3]]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (41, 41), 0)

                if num_frames < 30:
                    bg = run_avg(gray, weight, bg)
                else:
                    hand = segment(gray, bg)

                    if hand is not None:
                        (threshold, segmented) = hand
                        cv2.drawContours(outer_frame, [segmented + (ROI[2], ROI[0])], -1, (0, 0, 255))
                        cv2.imshow("Thesholded", threshold)
                        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area > AREA_THRESHOLD:
                                cv2.putText(outer_frame, "Hand is detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)
                                to_save = cv2.resize(threshold, (64, 64))
                                filename = os.path.join(IMAGE_DIR, 'img' + str(i) + '.jpg')
                                if not os.path.exists(filename):
                                    cv2.imwrite(filename, to_save)
                                    i = i + 1
                                    print (i)
                cv2.rectangle(outer_frame, (ROI[2], ROI[0]), (ROI[3], ROI[1]), (0, 255, 0), 2)
                cv2.putText(outer_frame, "Place your hand in the square", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                num_frames += 1
                cv2.imshow('frame', outer_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
