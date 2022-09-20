import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

#Função que cria pontos para cortar frequências
def circle(a, b, IMs, a1=0.005, a2=0.005):
    N = IMs.shape[0]
    M = IMs.shape[1]

    x, y = np.meshgrid(np.arange(M), np.arange(N))

    filter = 1 - np.exp(-a1*(x-a)**2 - a2*(y-b)**2)
    return filter

filename = sys.argv[1]
img = cv2.imread(filename,0)

#espectro de frequência da imagem
img_fft = np.fft.fft2(img)
IMs = np.fft.fftshift(img_fft)

#gerando pontos para limpar a imagem (obs: feito para a imagem halftone.png)

listX = [100, 300, 500, 700, 900]
listY = [65, 200, 330, 465, 600]
Z = 1

for x in listX:
    for y in listY:
        dot = circle(x, y, IMs)
        Z = Z * dot

listX_2 = [200, 400, 600, 800]
listY_2 = [130, 265, 400, 530]

for x in listX_2:
    for y in listY_2:
        dot = circle(x, y, IMs)
        Z = Z * dot

#aplica no espectro
IMFs = IMs * Z

#recupera a imagem com o filtro aplicado
IMFr = np.fft.ifftshift(IMFs)
imfr = np.fft.ifft2(IMFr)

#image plot
plt.subplot(221),plt.imshow(np.log(1+np.absolute(IMs)), cmap = 'gray')
plt.title('Espectro'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(np.log(1+np.absolute(IMFs)), cmap = 'gray')
plt.title('Filtro no espectro'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(np.real(imfr), cmap = 'gray')
plt.title('Resultado'), plt.xticks([]), plt.yticks([])

plt.show()
