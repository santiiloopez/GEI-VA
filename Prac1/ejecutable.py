import funciones as f
import funAux as fa
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

pathImg1 = "/home/santilopez/Documentos/uni/Q7/VAPractica/Prac1/grays.png"
pathImg2 = "/home/santilopez/Documentos/uni/Q7/VAPractica/Prac1/grid.png"
pathImg3 = "/home/santilopez/Documentos/uni/Q7/VAPractica/Prac1/circles.png"
pathImg4 = "/home/santilopez/Documentos/uni/Q7/VAPractica/Prac1/morph.png"


###################
### Ejecutables ###
###################

###################
#### FUNCIONES ####
### HISTOGRAMAS ###
###################

def ejecutarAdjInten(img, inRange, outRange):

    imgOut = f.adjustIntensity(img, inRange, outRange)

    cv.imshow("Imagen original", img)
    cv.imshow("Imagen ajustada", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax1.plot(fa.histograma(img, 101))
    ax1.set_title("Original")
    ax2.plot(fa.histograma(imgOut, 101))
    ax2.set_title("Ajustada")

    plt.show()

    return

def ejecutarEqualizeInten(img, nBins=256):

    imgOut = f.equalizeIntensity(img, nBins)

    cv.imshow("Imagen original", img)
    cv.imshow("Imagen ecualizada", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax1.plot(fa.histograma(img, 101))
    ax1.set_title("Original")
    ax2.plot(fa.histograma(imgOut, 101))
    ax2.set_title("Ecualizada")

    plt.show()

    return

####################
#### FUNCIONES #####
#### FILTRADO ######
####################

def ejecutarFilter(img, kernel):

    imgOut = f.filterImage(img, kernel)

    cv.imshow("Imagen original", img)
    cv.imshow("Imagen filtrada", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    return

def ejecutarGausskernel(sigma):

    kernel = f.gaussKernel1D(sigma)
    print(kernel)

    return

def ejecutarGaussianFilter(img, sigma):

    imgOut = f.gaussianFilter(img, sigma)

    cv.imshow("Imagen original", img)
    cv.imshow("Imagen filtrada", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    return

def ejecutarMedianFilter(img, filterSize):

    imgOut = f.medianFilter(img, filterSize)

    cv.imshow("Imagen original", img)
    cv.imshow("Imagen filtrada", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    return

###################
### OPERADORES ####
### MORFOLOGICOS ##
###################

def ejecutarErode(img, SE, center=[]):

    imgBinaria = fa.convert_binario(img)
    
    imgOut = f.erode(imgBinaria, SE, center)

    cv.imshow("Imagen original", img)
    cv.imshow("Imagen binaria", imgBinaria)
    cv.imshow("Imagen erosionada", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    return

def ejecutarDilate(img, SE, center=[]):

    imgBinaria = fa.convert_binario(img)

    imgOut = f.dilate(imgBinaria, SE, center)
    
    cv.imshow("Imagen original", img)
    cv.imshow("Imagen binaria", imgBinaria)
    cv.imshow("Imagen dilatada", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    return

def ejecutarOpening(img, SE, center=[]):

    imgBinaria = fa.convert_binario(img)

    imgOut = f.opening(imgBinaria, SE, center)

    cv.imshow("Imagen original", img)
    cv.imshow("Imagen binaria", imgBinaria)
    cv.imshow("Imagen apertura", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    return

def ejecutarClosing(img, SE, center=[]):

    imgBinaria = fa.convert_binario(img)

    imgOut = f.closing(imgBinaria, SE, center)

    cv.imshow("Imagen original", img)
    cv.imshow("Imagen binaria", imgBinaria)
    cv.imshow("Imagen cierre", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    return

def ejecutarHitOrMiss(img, objSEj, bgSE, center=[]):

    imgBinaria = fa.convert_binario(img)

    imgOut = f.hit_or_miss(imgBinaria, objSEj, bgSE, center)

    cv.imshow("Imagen original", img)
    cv.imshow("Imagen binaria", imgBinaria)
    cv.imshow("Imagen hit_or_miss", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    return

###################
#### DETECCIÓN ####
#### DE BORDES ####
###################

def ejecutarGradientImg(img, operator):

    
    grad = f.gradientImage(img, operator)

    cv.imshow("Imagen original", img)
    cv.imshow("Gx", grad[0])
    cv.imshow("Gy", grad[1])

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	


    return

def ejecutarLoG(img, sigma):

    imgOut = f.LoG(img, sigma)

    cv.imshow("Imagen original", img)
    cv.imshow("LoG", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    return

def ejecutarEdgeCanny(img, sigma, tlow, thigh):

    imgOut = f.edgeCanny(img, sigma, tlow, thigh)

    cv.imshow("Imagen original", img)
    cv.imshow("Bordes", imgOut)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	

    return


##############
#### MAIN ####
##############

img = fa.cargarImg(pathImg1) #grays.png
img2 = fa.cargarImg(pathImg2) #grid.png
img3 = fa.cargarImg(pathImg3) #circles.png
img4 = fa.cargarImg(pathImg4) #morph.png


inRange = []
outRange = [0, 1]

nBins = 256

sigma = 1


filterSize = 5

#Roberts / CentralDiff / Prewitt / Sobel
operator = "Sobel" 

kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])

SE = np.array([[0, 0, 0],
               [0, 1, 1],
               [0, 0, 0]])

objSEj = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]])


bgSE = np.array([[0, 1, 0],
                 [0, 0, 1],
                 [0, 0, 0]])

tlow = 20/255

thigh = 50/255


    
print("Opciones:")
print("\t1: Histogramas.")
print("\t2: Filtrado de imagenes.")
print("\t3: Operadores morfologicos.")
print("\t4: Detección de bordes.")

select = input("Seleccione la opcion: ")

if(select == "1"):
    
    print("Opciones:")
    print("\t1: adjustIntensity.")
    print("\t2: equalizeIntensity.")

    select = input("Seleccione la funcion: ")

    if(select == "1"):
        ejecutarAdjInten(img, inRange, outRange)
    elif(select == "2"):
        ejecutarEqualizeInten(img, nBins)
    else:
        print("Funcion no reconocida.")

elif(select == "2"):
    
    print("Opciones:")
    print("\t1: filterImage.")
    print("\t2: gaussKernel1D.")
    print("\t3: gaussianFilter.")
    print("\t4: medianFilter.")
    
    select = input("Seleccione la funcion: ")

    if(select == "1"):
        ejecutarFilter(img2, kernel)
    elif(select == "2"):
        ejecutarGausskernel(sigma)
    elif(select == "3"):
        ejecutarGaussianFilter(img2, sigma)
    elif(select == "4"):
        ejecutarMedianFilter(img2, filterSize)
    else:
        print("Funcion no reconocida.")

elif(select == "3"):
    
    print("Opciones:")
    print("\t1: erode.")
    print("\t2: dilate.")
    print("\t3: opening.")
    print("\t4: closing.")
    print("\t5: hit_or_miss.")

    select = input("Seleccione la funcion: ")

    if(select == "1"):
        ejecutarErode(img4, SE, [])
    elif(select == "2"):
        ejecutarDilate(img4, SE, [])
    elif(select == "3"):
        ejecutarOpening(img4, SE, [])
    elif(select == "4"):
        ejecutarClosing(img4, SE, [])
    elif(select == "5"):
        ejecutarHitOrMiss(img4, objSEj, bgSE, [])
    else:
        print("Funcion no reconocida.")


elif(select == "4"):
    
    print("Opciones:")
    print("\t1: gradientImage.")
    print("\t2: LoG.")
    print("\t3: edgeCanny.")

    select = input("Seleccione la funcion: ")

    if(select == "1"):
        ejecutarGradientImg(img3, operator)
    elif(select == "2"):
        ejecutarLoG(img2, sigma)
    elif(select == "3"):
        ejecutarEdgeCanny(img3, sigma, tlow, thigh)
    else:
        print("Funcion no reconocida.")

else:
    print("Funcion no reconocida.")







