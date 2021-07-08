import numpy as np
import cv2
import numpy as np
import pandas as pd

cap = cv2.VideoCapture('fish_video.avi')



#data = pd.read_excel('fish_data.xlsx', index_col=0)  
data = pd.read_csv('fish_data3.csv', index_col=0) 
#data = pd.read_csv('contexual_anomaly1.csv', index_col=14) 
#print(data)
#print(data['midx1'])

#data = pd.read_csv('perfect_data_1.csv', index_col=14)
#cap = cv2.VideoCapture('fish_video.avi')


x = 0
y = 0
i = 0.00
while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)
    
    
    x1 = int(data['midx1'][round(i,2)]/2)
    y1 = int(data['midy1'][round(i,2)]/2)
    x2 = int(data['midx2'][round(i,2)]/2)
    y2 = int(data['midy2'][round(i,2)]/2)
    x3 = int(data['midx3'][round(i,2)]/2)
    y3 = int(data['midy3'][round(i,2)]/2)
    x4 = int(data['midx4'][round(i,2)]/2)
    y4 = int(data['midy4'][round(i,2)]/2)    
    x5 = int(data['midx5'][round(i,2)]/2)
    y5 = int(data['midy5'][round(i,2)]/2)
    x6 = int(data['midx6'][round(i,2)]/2)
    y6 = int(data['midy6'][round(i,2)]/2)
    x7 = int(data['midx7'][round(i,2)]/2)
    y7 = int(data['midy7'][round(i,2)]/2)    
    
    
    
    frame = cv2.circle(frame, (x1,y1), radius=10, color=(0, 0, 255), thickness=-1)
    frame = cv2.circle(frame, (x2,y2), radius=10, color=(0, 0, 255), thickness=-1)
    frame = cv2.circle(frame, (x3,y3), radius=10, color=(0, 0, 255), thickness=-1)
    frame = cv2.circle(frame, (x4,y4), radius=10, color=(0, 0, 255), thickness=-1)
    frame = cv2.circle(frame, (x5,y5), radius=10, color=(0, 0, 255), thickness=-1)
    frame = cv2.circle(frame, (x6,y6), radius=10, color=(0, 0, 255), thickness=-1)
    frame = cv2.circle(frame, (x7,y7), radius=10, color=(0, 0, 255), thickness=-1)

    i += 0.01
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
