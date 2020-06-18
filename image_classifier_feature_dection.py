import cv2
import numpy as np
import os

path= "images_train"
# Oriented FAST and rotated BRIEF (ORB) is a fast robust local feature detector,
orb = cv2.ORB_create(nfeatures=1000) #DEFAULT IS 500 incerase to 1000 to get more features

#IMPORT IMAGES
images =[]
classname = []
mylist = os.listdir(path) #mylist contains the images in the train folder
print(mylist)
print("total classes dected is ",len(mylist))
for cl in mylist:
     imgcur = cv2.imread(f'{path}/{cl}',0)
     images.append(imgcur)
     classname.append(os.path.splitext(cl)[0])  #classname contains only the name of obect excluding the .jpg format
print(classname)



def findDes(images):
    desList=[]
    for img in images:
        kp,des =orb.detectAndCompute(img,None)   #des -contains the omages in the train folder
        desList.append(des)                      # des2 -contains the images from the webcam
    return desList

def findID(img,desList,thres=15):
    kp2,des2= orb.detectAndCompute(img,None) #detech the matching features, kp2 contains the boolean value
    bf = cv2.BFMatcher()    #use the bruteforce matching technique to match from the train list
    matchList=[]
    finalVal= -1
    try:
        for des in desList:
            matches = bf.knnMatch(des,des2,k=2)    #using knn clasifiaction approach to classify the dectected object
            good=[]
            for m,n in matches:
                if m.distance <0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
         #print(matchList)
    except:
        pass
    if len(matchList)!=0:
        if max(matchList)> thres:    # threshold val can be fixed manually also
            finalVal = matchList.index(max(matchList))  # final gets the index of mylist object which is matched with the object in front of it
    return finalVal

desList= findDes(images)
print(len(desList))

# capture vedio from webcam
cap= cv2.VideoCapture(0)
while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    id = findID(img2,desList)
    if id!=-1:
        cv2.putText(imgOriginal,classname[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    cv2.imshow("img2",imgOriginal)
    cv2.waitKey(1)            #if 0 then only runs for 1ms



