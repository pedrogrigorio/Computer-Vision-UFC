import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

filename = sys.argv[1]
img = cv2.imread(filename)

#remove o texto e aumenta a imagem para conseguir detectar todas as linhas
img[90:,:] = (255,255,255)
img = cv2.resize(img, (0,0), fx=1.5, fy=1.5)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#pega as bordas e aplica a transformada de Hough, pegando linhas de no minimo 80 (para pegar apenas verticais) com Gap pequeno para não deixar passar as barras finas
edges = cv2.Canny(gray, 100, 200, apertureSize=7)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=1)

#lista para guardar a coordenada x de cada reta
xCords = []
for line in lines:
	x1,y1,x2,y2 = line[0]
	xCords.append(x1)
	cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 1)

#ordena a lista
xCords = sorted(xCords)
print(xCords, "\n")

#medindo a distancia de uma linha para outra
distance = []
for i in range(len(xCords)-1):
	d = xCords[i+1] - xCords[i]
	distance.append(d)
print(distance, "\n")

#dividindo por 3 e convertendo para string
distance = (np.array(distance)/3).astype(int).astype(str)
print(distance, "\n")

#dicionario com a traducao do codigo de barras 128 (aqui estao listados apenas os que serao usados)
barcode_dict = {'211214':'INICIO(COD B)', '213131':'U', '132311':'F', '131321':'C', '122132':'-', '211331':'Q', '331121':'X', '112313':'D', '113141':'0', '222122':'1', '121241':'8', '231113':'2', '2331112':'PARADA'}

#lista para concatenar 6 digitos (tamanho de 3 barras e 3 espaços) que formam um caracter no código
code = []
for i in range(0, len(distance)-1,6):
	character = ''
	if(i + 7 == len(distance)):
		for j in range(7):
			character += distance[i+j]
	else:
		for j in range(6):
			character += distance[i+j]
	code.append(character)

#Primeiro caracter é o código de caracteres (A, B ou C)
COD = barcode_dict[code[0]]

#Ultimo caracter é o ponto de parada e contem 7 dígitos (4 barras e 3 espaços)
end_char = barcode_dict[code[-1]]

#Informação útil
code = code[1:-1]

#Concatenando caracteres para formar a palavra
word = ''
for char in code:
	word += barcode_dict[char]

print(word)

plt.subplot(311),plt.imshow(gray,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(312),plt.imshow(edges,cmap = 'gray'),plt.title('Edges')
plt.xticks([]), plt.yticks([])
plt.subplot(313),plt.imshow(img),plt.title('Linhas')
plt.xticks([]), plt.yticks([])

plt.show()
