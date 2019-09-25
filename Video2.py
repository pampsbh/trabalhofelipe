import cv2
import numpy as np
from matplotlib import pyplot as plt
cont = 0
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('coelho.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  if ret == True:
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
      )
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]   
        cv2.imwrite(("Pessoas/face"+str(cont)+".png"), roi_color)
        #img = cv2.imread('Pessoas/face0.png',0)
        plt.hist(roi_color.ravel(),256,[0,256])
        histg = cv2.calcHist([roi_color],[0],None,[256],[0,256])
        plt.savefig(("HistFace/HistogramaFace"+str(cont)+".jpg")) 
        #HISTOGRAMA FRAMES
        plt.hist(frame.ravel(),256,[0,256])
        histg = cv2.calcHist([frame],[0],None,[256],[0,256])
        plt.savefig(("Hist/HistogramaFrame"+str(cont)+".jpg")) 
    
    cont = cont + 1
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()