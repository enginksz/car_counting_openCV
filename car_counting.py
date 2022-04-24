import cv2
import numpy as np
from time import sleep

#min and max values for rectangle
min_rectangle_width = 65
min_rectangle_height = 65
maks_rectangle_width = 150
maks_rectangle_height = 150

offset = 6 #pixel offset

#x border for entrance and exit. Y border for car counting
y_border = 210
x_border = 320


delay = 60 #offset

detection = [] # detect array

number_of_vehicles= 0 #number of detected cars
number_of_vehicles_entering = 0 #number of entered cars
number_of_vehicles_exits = 0 #number of departing cars

def merkez_al(x, y, w, h): # find the center of rectangle
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video_cars.mp4') #file to read

backg_sub = cv2.createBackgroundSubtractorKNN() #background subtraction
#backg_sub = cv2.createBackgroundSubtractorMOG(); you can't use mog in raspberry
#if you are going to use mog you should change the other parameters too

while True:

    ret, frame = cap.read() #read the camera or video frame
    tmp = float(1/delay)
    sleep(tmp)
    
    # You can change these parameters add new ones or delete. You have to find
    # your algorithm if you are working on another video. Since we are using low
    # profile systems we can't use machine learning models. You need to figure
    # out whats best for your case.
    
    
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey,(21,21),0)
    sub_img = backg_sub.apply(blurred)
    dilate = cv2.dilate(sub_img,np.ones((5,5)))    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    #Morphologhical operations
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel,iterations = 3)
    eroding = cv2.morphologyEx(opening, cv2.MORPH_ERODE, kernel,iterations = 2)
    closing = cv2.morphologyEx (eroding, cv2. MORPH_CLOSE , kernel,iterations = 3)

    #contour
    contours,hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25, y_border), (600, y_border), (255,255,0), 2)
    cv2.line(frame, (x_border, 25), (x_border, 340), (0,0,0), 2)
    
    for(i,c) in enumerate(contours):
        
        (x,y,w,h) = cv2.boundingRect(c)
        contour_kontrol = (w >= min_rectangle_width) and (h >= min_rectangle_height) and w<160 and h<160
        if not contour_kontrol:
            continue
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)        
        merkez = merkez_al(x, y, w, h)
        detection.append(merkez)
        cv2.circle(frame, merkez, 4, (0, 0,255), -1)
        
        
        # If you are going to use another video, you also should change these
        # border values. Find whats best for your case.
        
        for (x,y) in detection:             
            if y<(y_border+offset) and y>(y_border-offset)and x <= x_border:
                number_of_vehicles_entering +=1
                number_of_vehicles+=1
                cv2.line(frame, (25, y_border), (600, y_border), (0,0,255), 2)
                detection.clear()
                
            if y<(y_border+offset) and y>(y_border-offset)and x >= x_border:
                number_of_vehicles_exits+=1
                number_of_vehicles+=1                     
                detection.clear()
                cv2.line(frame, (25, y_border), (600, y_border), (34,233,29), 2)

            
    cv2.putText(frame, "Total Number of Vehicles : "+str(number_of_vehicles), (300, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.9, (0, 0, 0),2)
    cv2.putText(frame, "Number of Vehicles Entered: "+str(number_of_vehicles_entering), (300, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255),2)
    cv2.putText(frame, "Number of Vehicles Released: "+str(number_of_vehicles_exits), (300, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (34, 233, 29),2)

    cv2.imshow("Video Cars" , frame)

    #Press esc to quit
    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
