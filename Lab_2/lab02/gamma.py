import sys
import cv2
import numpy as np

from cv_utils import waitKey


def gamma_correction(img, gamma,c=1.0):
   i = img.copy()
   i[:,:,:] = 255*(c*(img[:,:,:]/255.0)**(1.0 / gamma))
   return i


def gamma_correction_LUT(img, gamma,c=1.0):

	#cria uma Lookup Table (LUT)
	GAMMA_LUT = np.array([c*((i / 255.0) ** (1.0 / gamma)) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# aplica a transformação usando LUT
	return cv2.LUT(img, GAMMA_LUT)


def callback_trackbar(x):
    try:
        gamma = cv2.getTrackbarPos('gamma','image')
        im_gamma = gamma_correction_LUT(im, gamma*0.01)
        #pega os canais da foto com alteração no gamma
        b2, g2, r2 = cv2.split(im_gamma)

        #cria uma nova imagem com alteração apenas no azul
        new_image = cv2.merge((b2, g, r))

        #diminuindo o gamma para o azul temos o efeito amarelado
        cv2.imshow('image',new_image)
    except:
        return




#abre imagem
im = cv2.imread('./imagens/jato.jpg')

#guarda os canais da foto original
b,g,r = cv2.split(im)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.createTrackbar('gamma','image',0,100,callback_trackbar)


cv2.imshow('image',im)
waitKey('image', 27) #27 = ESC	













