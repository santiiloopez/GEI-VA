import funciones as f
import funAux as fa
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation, filters

pathImg1 = "/home/santilopez/Documentos/uni/Q7/VAPracFinal/Prac2/MaterialPulmones/im1.png"
pathImg2 = "/home/santilopez/Documentos/uni/Q7/VAPracFinal/Prac2/MaterialPulmones/im2.png"
pathImg3 = "/home/santilopez/Documentos/uni/Q7/VAPracFinal/Prac2/MaterialPulmones/im3.png"
pathImg4 = "/home/santilopez/Documentos/uni/Q7/VAPracFinal/Prac2/MaterialPulmones/im4.png"
pathImg5 = "/home/santilopez/Documentos/uni/Q7/VAPracFinal/Prac2/MaterialPulmones/im5.png"
pathImg6 = "/home/santilopez/Documentos/uni/Q7/VAPracFinal/Prac2/MaterialPulmones/im6.png"
pathImg7 = "/home/santilopez/Documentos/uni/Q7/VAPracFinal/Prac2/MaterialPulmones/im7.png"

sigma = 2

##############
#### MAIN ####
##############


img1 = fa.cargarImg(pathImg1)
img2 = fa.cargarImg(pathImg2)
img3 = fa.cargarImg(pathImg3)
img4 = fa.cargarImg(pathImg4)
img5 = fa.cargarImg(pathImg5)
img6 = fa.cargarImg(pathImg6) 
img7 = fa.cargarImg(pathImg7)

img = img7


#Aplicamos un filtro gausiano para suavizar la imagen

img_suavizada = f.gaussianFilter(img, sigma)

#Aplicamos umbralizacion de otsu para segmentar las regiones pulmonares

otsu = filters.threshold_otsu(img_suavizada[img_suavizada>0])       #Calcular Umbral de otsu

_, img_binaria = cv.threshold(img_suavizada, otsu, 1, cv.THRESH_BINARY_INV)

#Aplicamos apertura para eliminar componentes no pulmonares

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))     #Calcular EE

img_op = f.opening(img_binaria, kernel)

#Componentes conexas

etiquetas, num_etiquetas = ndi.label(img_op)     #Etiquetar regiones

areas = ndi.sum(img_op, etiquetas, range(num_etiquetas + 1))        #Eliminar la region que no es pulmon

img_conect = img_op
img_conect[etiquetas == areas.argmax()] = 0

img_conect = ndi.binary_fill_holes(img_conect)      #Rellenar huecos

#Eliminar traquea y bronquios

etiquetasTraq, _ = ndi.label(img_conect)

propiedades = measure.regionprops(etiquetasTraq)    #Calcular las propiedades de las regiones

img_pulm = np.zeros_like(img)

for prop in propiedades:
    #Comprobamos que los pulmones tiene un area significativa y no se encuentran donde se suele encontrar la traquea
    if prop.area > 500 and prop.bbox[0] > 0.1 * img.shape[0]:
        img_pulm[etiquetasTraq == prop.label] = 1

#Extraer contornos

contornos, _ = cv.findContours(img_pulm.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

img_cont = np.zeros_like(img)
cv.drawContours(img_cont, contornos, -1, (1), thickness = 2)

#Dibujar los contornos sobre la imagen original

imgOut = img.copy()

cv.drawContours(imgOut, contornos, -1, (255, 0, 0), thickness=2)

cv.imshow("4", imgOut)

k = cv.waitKey(0)
if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
    cv.destroyAllWindows()	

#Subplot imagen original
plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")

#Subplot imagen suavizada
plt.subplot(2, 3, 2)
plt.imshow(img_suavizada, cmap="gray")
plt.title("Imagen suavizada")

#Subplot imagen umbralizada
plt.subplot(2, 3, 3)
plt.imshow(img_binaria, cmap="gray")
plt.title("Imagen umbralizada (otsu)")

#Subplot imagen apertura
plt.subplot(2, 3, 4)
plt.imshow(img_op, cmap="gray")
plt.title("Apertura")

#Subplot imagen Componentes Conexas
plt.subplot(2, 3, 5)
plt.imshow(img_conect, cmap="gray")
plt.title("Comp Conexas")

#Subplot imagen contornos
plt.subplot(2, 3, 6)
plt.imshow(img_cont, cmap="gray")
plt.title("Contornos")

plt.tight_layout()
plt.show()

"""
print(otsu)
cv.imshow("1", img_binaria)
cv.imshow("2", img_op)
cv.imshow("3", img_cont)
cv.imshow("4", imgOut)

k = cv.waitKey(0)
if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
    cv.destroyAllWindows()	
"""





