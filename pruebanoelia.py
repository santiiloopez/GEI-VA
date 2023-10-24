import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io

def adjustIntensity(inImage, inRange=[], outRange=(0, 1)):
    if inRange == []:
        imin = np.min(inImage)
        imax = np.max(inImage)
    else:
        imin = inRange[0]
        imax = inRange[1]

    GminNorm = outRange[0]
    GmaxNorm = outRange[1]

    outImage = GminNorm + (((GmaxNorm - GminNorm) * (inImage - imin)) / (imax - imin))
    outImage = np.clip(outImage, outRange[0], outRange[1])
    return outImage
"""
https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=equalizehist


The function equalizes the histogram of the input image using the following algorithm:

    Calculate the histogram H for src .

    Normalize the histogram so that the sum of histogram bins is 255.

    Compute the integral of the histogram:

            H'_i = \sum _{0 \le j < i} H(j)

    Transform the image using H' as a look-up table: \texttt{dst}(x,y) = H'(\texttt{src}(x,y))

The algorithm normalizes the brightness and increases the contrast of the image.
"""
def equalizeIntensity(inImage, nBins=256):
    hist = np.zeros(nBins)
    histAcum = np.zeros(nBins)
    histAcumNorm = np.zeros(nBins)

    img_scl = np.round(inImage * (nBins - 1))

    for elemento in img_scl.ravel():
        aux = int(elemento)
        hist[aux] += 1

    histAcum[0] = hist[0]
    for valor in range(1, nBins):
        histAcum[valor] = histAcum[valor - 1] + hist[valor]

    histAcumNorm = histAcum / histAcum.max()
    outImage = histAcumNorm[img_scl.astype(int)]

    return outImage



img = io.imread("/home/santilopez/Documentos/uni/Q7/VA2/imagenprueba.jpeg")
img_gray = color.rgb2gray(img)
img_esc = img_gray / 256.0

# Se crea la figura para visualizar las cuatro imagenes
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 10))

#Visualizar 
ax1.imshow(img_gray, cmap='gray')
ax1.set_title('Imagen Original')

ax3.hist(img.ravel(), bins=256, range=(0,256), color='blue', alpha=0.7)
ax3.set_title('Original')

# Se muestra la imagen ajustada
img_salida = adjustIntensity(img_esc, [], [0, 1])
ax2.imshow(img_salida, cmap='gray')
ax2.set_title('Imagen Transformada')

ax4.hist(img_salida.ravel(), bins=256, range=(0, 1), color='blue', alpha=0.7)
ax4.set_title('Transformada')

img_ecualizada = equalizeIntensity(img_esc)
ax5.imshow(img_ecualizada, cmap='gray')
ax5.set_title('Ecualizada')

ax6.hist(img_ecualizada.ravel(), bins=256, range=(0, 1), color='blue', alpha=0.7)
ax6.set_title('Ecualizada')

plt.tight_layout()
plt.show()