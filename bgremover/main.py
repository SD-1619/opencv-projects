import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

listImg = os.listdir("images")
imglist = []
for imgPath in listImg:
    img = cv2.imread(f'images/{imgPath}')
    imglist.append(img)

indexImg = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imglist[indexImg], threshold=0.8)

    imgstacked = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgstacked = fpsReader.update(imgstacked, color=(0, 0, 255))
    print('Image no: ', indexImg + 1)
    cv2.imshow("Image", imgstacked)
    key = cv2.waitKey(1)
    if key == ord('s'):
        if indexImg>0:
            indexImg -=1
    elif key == ord('d'):
        if indexImg<len(imglist) -1:
            indexImg +=1
    elif key == ord('q'):
        break
