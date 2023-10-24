import func as f
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
from skimage import color
from skimage import io


pathImg = "/home/santilopez/Documentos/uni/Q7/VA2/imagenprueba.jpeg"


###################
#### FUNCIONES ####
###################

def adjustIntensity(inImage, inRange, outRange):
    
    if inRange == []:
        inRange = [np.amin(inImage), np.amax(inImage)] 

    if outRange is None:
        outRange = [0, 1]
 
    outImage = outRange[0] + ((outRange[1] - outRange[0]) * (inImage - inRange[0])) / (inRange[1] - inRange[0])

    return outImage



##############
#### MAIN ####
##############

#src = f.cargarImg(pathImg)
img = io.imread(pathImg, as_gray=True)
print(img)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title("Original")
ax3.hist(img.ravel(), bins=256, range=(0,1))

inRange = []
outRange = [0, 1]

imgOut = adjustIntensity(img, inRange, outRange)

ax2.imshow(imgOut, cmap=plt.cm.gray)
ax2.set_title('Ajustada')
ax4.hist(imgOut.ravel(), bins=256, range=(0,1))
plt.show()

