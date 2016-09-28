# Face detection in images using haar classifier from opencv
# Download first haarcascade_frontalface_alt.xml to get the code working
import os
import cv2
from PIL import Image
cascadeClassifier = '/path-to-dir/haarcascade_frontalface_alt.xml'
rootDir = "/path-to-dir/"
classDirPath = rootDir + "binaryClasses/"
faceFileDir = rootDir + 'faces/'
classDirNames = next(os.walk(classDirPath))[1]
for classDir in classDirNames:
	imageDirPath = classDirPath+classDir+'/'
	imageNames = next(os.walk(imageDirPath))[2]
	classFaceFilePath = faceFileDir + classDir+'/'
	print(classDir)
	for imageName in imageNames:
		imageFile = imageDirPath+imageName
		img = cv2.imread(imageFile)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faceCascade = cv2.CascadeClassifier(cascadeClassifier)
		faces = faceCascade.detectMultiScale(gray, 1.3, 5)
		if(len(faces)==1):
			for (x,y,w,h) in faces:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				sub_face = img[y:y+h, x:x+w]
				face_file_name = classFaceFilePath + imageName
				cv2.imwrite(face_file_name, sub_face)
