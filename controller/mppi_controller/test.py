import cv2
import numpy as np
import pyautogui

# Tạo cửa sổ có khả năng thay đổi kích thước
cv2.namedWindow("RViz Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RViz Stream", 1280, 720)   # đặt kích thước cửa sổ mong muốn

while True:
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    cv2.imshow("RViz Stream", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("rviz_capture.png", frame)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
