import funAux as fa
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import math


###################
#### FUNCIONES ####
### HISTOGRAMAS ###
###################

def adjustIntensity(inImage, inRange, outRange):
    
    if inRange == []:
        inRange = [np.amin(inImage), np.amax(inImage)] 


    if outRange is None:
        outRange = [0, 1]
 
    outImage = outRange[0] + ((outRange[1] - outRange[0]) * (inImage - inRange[0])) / (inRange[1] - inRange[0])

    return outImage

def equalizeIntensity(inImage, nBins=256):
    
    hist = fa.histograma(inImage, nBins)
    histAcum = np.zeros(nBins)
    M, N = inImage.shape
     
    histAcum[0] = hist[0]
    
    for i in range(1, hist.size):
        histAcum[i] = histAcum[i-1] + hist[i]
            
    outImage = np.zeros((M, N))
    
    aux = 1 / (M*N)

    for i in range(M): 
        for j in range(N):
            outImage[i,j] = aux * histAcum[int(inImage[i,j]*100)]
    
    return outImage


####################
#### FUNCIONES #####
#### FILTRADO ######
####################

def filterImage(inImage, kernel):
    
    M, N = inImage.shape        
    P, Q = kernel.shape         

    outImage = np.zeros((M,N))

    Pcentro = math.floor(P/2)
    Qcentro = math.floor(Q/2)
    

    for i in range(M):
        for j in range(N):
            if i > Pcentro and i < M - Pcentro and j > Qcentro and j < N - Qcentro:
            
                vecindario = inImage[i - Pcentro:i + Pcentro + 1,
                                     j - Qcentro:j + Qcentro + 1]
                
                outImage[i,j] = fa.sum(vecindario * kernel)

    return outImage

def gaussKernel1D(sigma):

    N = 2 * math.ceil(3*sigma) + 1

    if N % 2 == 0:      #Comprobamos que el tamaño del kernel sea impar
        N += 1

    centro = N // 2

    array_aux = np.arange(-centro, centro + 1)      #Iniciamos el kernel con tamaño N

    exp = 2 * (sigma ** 2)
    aux = np.sqrt(2 * np.pi) * sigma
    
    kernel = (1 / aux) * np.exp(-(array_aux**2)/exp)

    kernel /= np.sum(kernel)        #Normalizamos el kernel para que la suma de todos sus valores sea 1

    kernel = kernel.reshape(1, -1)      #Convertimos el kernel en una matriz 1xN

    return kernel

def gaussianFilter(inImage, sigma):
    
    kernel = gaussKernel1D(sigma)

    auxImage = filterImage(inImage, kernel)

    kernelT = np.transpose(kernel)

    outImage = filterImage(auxImage, kernelT)

    return outImage

def medianFilter(inImage, filterSize):

    centro = math.floor(filterSize/2) 

    M, N = inImage.shape        

    outImage = np.zeros((M,N))

    for i in range(M):
        for j in range(N):
            vecindario = inImage[max(0, i - centro):min(i + centro + 1, M), 
                                 max(0, j - centro):min(j + centro + 1, N)]

            outImage[i,j] = np.median(vecindario)

    return outImage


###################
### OPERADORES ####
### MORFOLOGICOS ##
###################

def erode(inImage, SE, center=[]):
    
    M, N = inImage.shape
    outImage = np.zeros((M, N))

    P, Q = SE.shape
    
    if center == []:
        center = [P//2, Q//2]

    pad_img = np.pad(inImage, ((center[0], center[0]), (center[1], center[1])), mode='constant', constant_values=0)     #Aplicar padding

    for i in range(M):
        for j in range(N):
            vecindario = pad_img[i:i+P, j:j+Q]      #Extraer el vecindario de la posicion del tamaño del SE

            if np.all(vecindario[SE == 1] == 1):    #Comprobar si el SE encaja
                outImage[i, j] = 1

    return outImage

def dilate(inImage, SE, center=[]):
    
    M, N = inImage.shape
    outImage = np.zeros((M, N))

    P, Q = SE.shape

    if center == []:
        center = [P//2, Q//2]

    pad_img = np.pad(inImage, ((center[0], center[0]), (center[1], center[1])), mode='constant', constant_values=0)     #Aplicar padding


    for i in range(M):
        for j in range(N):
            vecindario = pad_img[i:i+P, j:j+Q]      #Extraer el vecindario de la posicion del tamaño del SE

            if np.any(vecindario[SE == 1] == 1):    #Comprobar si algun elemento del SE coincide con el vecindario
                outImage[i, j] = 1

    return outImage

def opening(inImage, SE, center=[]):
    
    auxImage = erode(inImage, SE, center)

    outImage = dilate(auxImage, SE, center)

    return outImage

def closing(inImage, SE, center=[]):
    
    auxImage = dilate(inImage, SE, center)

    outImage = erode(auxImage, SE, center)
    
    return outImage

def hit_or_miss(inImage, objSEj, bgSE, center=[]):

    if np.count_nonzero(fa.interseccion(objSEj, bgSE) != 0):
        sys.exit("Error: elementos estructurantes incoherentes.")
    
    hitImage = erode(inImage, objSEj, center)

    inImageComp = 1 - inImage

    missImage = erode(inImageComp, bgSE, center)   

    cv.imshow("hit", hitImage)
    cv.imshow("miss", missImage)

    k = cv.waitKey(0)
    if k == '\x1b':     #Presionar la tecla ESC para cerrar las ventanas
        cv.destroyAllWindows()	
    
    outImage = fa.interseccion(hitImage, missImage)

    return outImage


###################
#### DETECCIÓN ####
#### DE BORDES ####
###################

def gradientImage(inImage, operator):
    
    if operator == "Roberts":
        
        kernel1 = np.array([[-1, 0],
                           [0, 1]])
        
        kernel2 = np.array([[0, -1],
                           [1, 0]])
    
    elif operator == "CentralDiff":
    
        kernel1 = np.array([[-1, 0, 1]])    
        
        kernel2 = np.array([[-1],
                            [ 0],
                            [ 1]])
        
    elif operator == "Prewitt":
        
        kernel1 = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        
        kernel2 = np.array([[-1, -1, -1],
                            [ 0,  0,  0],
                            [ 1,  1,  1]])
    
    elif operator == "Sobel":
        
        kernel1 = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        
        kernel2 = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])
    
    else:
        sys.exit("Operador no valido.")

    gx = filterImage(inImage, kernel1)
    gy = filterImage(inImage, kernel2)

    return [gx, gy]

def LoG(inImage, sigma):

    img_suavizada = gaussianFilter(inImage, sigma)

    kernel = np.array([
        [0, 0, -1, 0, 0], 
        [0, -1, -2, -1, 0], 
        [-1, -2, 16, -2, -1], 
        [0, 0, -1, 0, 0], 
        [0, -1, -2, -1, 0]
    ])


    outImage = filterImage(img_suavizada, kernel)

    return outImage

def edgeCanny(inImage, sigma, tlow, thigh):

    #Suavizar la imagen mediante gauss

    img_suavizada = gaussianFilter(inImage, sigma)

    #Calcular magnitud y direccion

    operator = "Sobel"

    gradient = gradientImage(img_suavizada, operator)

    magnitud = np.hypot(gradient[0], gradient[1])
    direccion = np.arctan2(gradient[1], gradient[0])
    
    #Supresion No Maxima

    M, N = magnitud.shape
    
    supp_array = np.zeros((M, N))

    angulo = direccion * 180 / np.pi
    angulo[angulo < 0] +=  180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255
                
                #0 grados
                if (0 <= angulo[i,j] < 22.5) or (157.5 <= angulo[i,j] <= 180):
                    q = magnitud[i, j+1]
                    r = magnitud[i, j-1]
                #45 grados
                elif (22.5 <= angulo[i,j] < 67.5):
                    q = magnitud[i+1, j-1]
                    r = magnitud[i-1, j+1]
                #90 grados
                elif (67.5 <= angulo[i,j] < 112.5):
                    q = magnitud[i+1, j]
                    r = magnitud[i-1, j]
                #135 grados
                elif (112.5 <= angulo[i,j] < 157.5):
                    q = magnitud[i-1, j-1]
                    r = magnitud[i+1, j+1]

                if (magnitud[i,j] >= q) and (magnitud[i,j] >= r):
                    supp_array[i,j] = magnitud[i,j]
                else:
                    supp_array[i,j] = 0


            except IndexError as e:
                pass
        
    #Umbralizacion con histeresis

    high = supp_array > thigh
    low = supp_array > tlow
    
    outImage = np.zeros((M, N))
    outImage[high] = 1
    
    while True:
        aux_array = np.zeros((M, N))
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if outImage[i, j] == 1:
                    vecindario = outImage[i-1:i+2, j-1:j+2]
                    aux_array[i-1:i+2, j-1:j+2] = np.logical_or(vecindario, low[i-1:i+2, j-1:j+2])
        
        if np.array_equal(outImage, aux_array):
            break
        outImage = aux_array
    
    return outImage

