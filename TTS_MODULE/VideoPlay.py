import os
import time

import cv2

Vid = cv2.VideoCapture('/home/mj/Desktop/mydata/04.mp4')

# if Vid.isOpened():
#     fps = Vid.get(cv2.CAP_PROP_FPS)
#     f_count = Vid.get(cv2.CAP_PROP_FRAME_COUNT)
#     f_width = Vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#     f_height = Vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
#     print('Frames per second : ', fps, 'FPS')
#     print('Frame count : ', f_count)
#     print('Frame width : ', f_width)
#     print('Frame height : ', f_height)

time.sleep(12) # 1 : 8 , 2 : 12

while Vid.isOpened():
    run, frame = Vid.read()
    
    cv2.imshow('Lecture_Video', frame)
    
    if cv2.waitKey(35) & 0xFF == ord('q'):
        break

Vid.release()
cv2.destroyAllWindows()