from PIL import Image
import os

os.chdir(".//_Assets")
for file in os.listdir():
	fname, ext = os.path.splitext(file)
	if ext == ".png" or ext == ".jpg" or ext == ".jpeg":
		img = Image.open(file)
		img.save(fname+".bmp")


		 
