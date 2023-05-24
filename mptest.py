import cv2
import mediapipe as mp
import time

def get_mediapipe_boundingbox(img):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    #success, img = cap.read()
    h, w, c = img.shape

    #while True:
        #success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape

                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

                # cx, cy = int(lm.x *w), int(lm.y*h)
                # if id ==0:
                cv2.circle(img, (x, y), 7, (255, 0, 255), cv2.FILLED)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        return x_min, y_min, x_max, y_max

def main():
    img = cv2.imread("C:/development/OCULI/testing/hand-detection-tutorial/egohands_kitti_formatted/images/CARDS_COURTYARD_B_T_frame_0011.jpg")
    get_mediapipe_boundingbox(img)
    cv2.imshow("image", img)
    cv2.waitKey(0)

main()