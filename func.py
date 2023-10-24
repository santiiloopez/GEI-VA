#Cargar imagen, guardar y visualizar
#Formalizar y normalizar a [0,1]

import cv2 as cv
import sys

def cargarImg(pathImg):
    src = cv.imread(pathImg, cv.IMREAD_GRAYSCALE)
    if src is None:
        sys.exit("No se pudo leer la imagen.\n")
    img = src/256   #Convertir el rango [0..255] a [0..1]

    return img

def guardarImg(img):
    cv.imwrite("imgPrueba.png", img)
    return

def mostrarImg(img):
    cv.imshow("Imagen", img)
    k = cv.waitKey(0)
    if k == '\x1b':
        cv.destroyAllWindows()	
    