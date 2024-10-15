import numpy as np
import cv2 as cv
import sys
import math

##############
### FunAux ###
##############

def cargarImg(pathImg):
    src = cv.imread(pathImg, cv.IMREAD_GRAYSCALE)
    if src is None:
        sys.exit("No se pudo leer la imagen.\n")
    img = src/256   #Convertir el rango [0..255] a [0..1]


    return img

def guardarImg(img, nombre):
    
    cv.imwrite(nombre, img)

    return

def histograma(img, bins):
    
    hist = np.zeros(bins)

    M, N = img.shape

    for i in range(M):
        for j in range(N):
            aux = int(img[i,j]*100)
            hist[aux] += 1
    
    return hist

def sum(array):
    
    M, N = array.shape

    result = 0

    for i in range(M):
        for j in range(N):
            result += array[i,j]
    
    return result

def convert_binario(img):

    imgBinaria = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] >= 0.5:
                imgBinaria[i,j] = 1
            else:
                imgBinaria[i,j] = 0

    return imgBinaria

def interseccion(img1, img2):
    
    M, N = img1.shape
    outImage = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            if img1[i,j] == 1 and img2[i,j] == 1:
                outImage[i,j] = 1

    return outImage


            
            
