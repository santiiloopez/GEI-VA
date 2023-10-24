import func as f
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys


pathImg = "/home/santilopez/Documentos/uni/Q7/VA2/imagenprueba.jpeg"


###################
#### FUNCIONES ####
#### AUXILIARES ###
###################


###################
#### FUNCIONES ####
###################


def adjustIntensity(inImage, inRange, outRange):
    
    if inRange is None:
        inRange = [np.amin(inImage), np.amax(inImage)]   

    if outRange is None:
        outRange = [0, 1]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
 
    ax1.imshow(inImage, cmap='gray')
    ax3.hist(inImage.ravel(), bins=256, range=(0,1))
 

    outImage = outRange[0] + ((outRange[1] - outRange[0]) * (inImage - inRange[0])) / (inRange[1] - inRange[0])
    #outImage = np.clip(outImage, outRange[0], outRange[1])

    
    ax2.imshow(outImage, cmap='gray')
    ax4.hist(outImage.ravel(), bins=256, range=(0,1))
    plt.show()

    return outImage


def equalizeIntensity(inImage, nBins = 256):
    src = np.zeros(nBins)
    histAcum = np.zeros(nBins)
    histNorm = np.zeros(nBins)

    img = np.round(inImage * (nBins - 1))

    #for i in img.ravel(): 
    

    return #outImage


##############
#### MAIN ####
##############


src = f.cargarImg(pathImg)
#f.mostrarImg(src)

inRange = None
outRange = [0.7, 1]

img = adjustIntensity(src, inRange, outRange)




