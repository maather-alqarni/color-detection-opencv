import cv2
import numpy as np

cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    
    def draw(mask, color, name):
        global prev_x, prev_y

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 3000:
                x, y, w, h = cv2.boundingRect(cnt)

                
                cx = x + w // 2
                cy = y + h // 2

                
                cx = int((cx + prev_x) / 2)
                cy = int((cy + prev_y) / 2)

                prev_x, prev_y = cx, cy

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, name, (x, y), 1, 1, color, 2)

    draw(mask_red, (0,0,255), "Red")
    draw(mask_green, (0,255,0), "Green")
    draw(mask_blue, (255,0,0), "Blue")

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
