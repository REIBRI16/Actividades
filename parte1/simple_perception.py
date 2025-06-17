# Importamos las librerías necesarias
import cv2 
import numpy as np
import math

# Importamos la clase Image de la librería PIL
from PIL import Image

# Abre la cámara
vid = cv2.VideoCapture(0) 

while(True): 
	
    # Obtenemos un único frame
    ret, img = vid.read()

    # Transformamos la imagen a HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definimos los rangos de los colores, en este caso el verde
    # OJO: estos rangos dependen de la cámara y de la iluminación, y deben ser ajustados.
    low_green = np.array([40, 100, 100])
    high_green = np.array([80, 255, 255])

    # Creamos la máscara para el color verde con los rangos definidos
    green_mask = cv2.inRange(img_hsv, low_green, high_green)

    # Aplicamos la máscara a la imagen original
    img_masked = cv2.bitwise_and(img, img, mask=green_mask)

    # Convertimos la máscara a un objeto de la clase Image de la librería PIL
    # OJO: esta máscara está en formato distinto al obtenido en la línea 26.
    green_mask_ = Image.fromarray(green_mask)

    # Obtenemos el bounding box de la máscara
    bounding_box = green_mask_.getbbox()

    # Dibujamos el bounding box en la imagen original y en la imagen con la máscara
    # Recomendación: Para dibujar siempre usen opencv!
    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        cv2.rectangle(img_masked, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_masked, "green", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "green", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Mostramos las imágenes
    cv2.imshow('masked', img_masked)
    cv2.imshow('original', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
