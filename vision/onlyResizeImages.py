import os
from os.path import join
import numpy
import PIL
from PIL import Image

rootDir = '/path-to-root-dir/' # where multiple type of datasets are stored

dataset = sys.argv[1]
rootDir = rootDir+dataset+'/'
notInTargetDataset = ""
dirLayer1Names = os.walk(rootDir).next()[1]
for dir1 in dirLayer1Names:
	if dir1 == notInTargetDataset:
		continue
	dirLayer1 = rootDir+dir1+'/'
	dirLayer2Names = os.walk(dirLayer1).next()[1]
	for dir2 in dirLayer2Names:
		dirLayer2 = dirLayer1+dir2+'/'
		print dirLayer2
		fileNames = os.walk(dirLayer2).next()[2]
		for files in fileNames:
			imageFile = dirLayer2+files
			try:
				img = Image.open(imageFile)
				img=img.resize((256,256), PIL.Image.ANTIALIAS)
				img.save(imageFile)
			except :
				pass
