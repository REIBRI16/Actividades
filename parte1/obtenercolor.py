import cv2
import numpy as np
import random
from PIL import Image

# Función para convertir el valor HSV a un valor CV
def hsv_to_cv(hue, saturation, v):
    huecv = (179/355) * hue
    saturationcv = (255/100) * saturation
    vcv = (255/100) * v
    return np.array([huecv, saturationcv, vcv])

# Reproducción del video y obtención de fotograma aleatorio
path = "PerceptionDataset.mp4"
vid = cv2.VideoCapture(path)
total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Función para calcular el promedio de color en el espacio HSV en una región seleccionada
def get_avg_color_hsv(image, roi):
    x, y, w, h = roi
    roi_image = image[y:y+h, x:x+w]
    hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    avg_color_per_row = np.average(hsv_image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color

while(True):
    # Seleccionamos un fotograma aleatorio del video
    random_frame = random.randint(0, total_frames-1)
    vid.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
    ret, img = vid.read()
    
    if not ret:
        break
    
    img = cv2.resize(img, (700, 600))  # Redimensionamos la imagen
    cv2.imshow('Video', img)  # Mostramos el fotograma en pantalla
    
    # Te dejamos seleccionar el área de interés en la imagen
    roi = cv2.selectROI("Seleccionar área de interés", img, fromCenter=False, showCrosshair=True)
    
    # Cuando se presiona 'ENTER' o 'ESC' después de seleccionar el ROI, se calcula el color promedio
    if roi != (0, 0, 0, 0):  # Verificamos que se haya seleccionado un área
        avg_hsv = get_avg_color_hsv(img, roi)
        print(f"Promedio HSV en la región seleccionada: {avg_hsv}")
    
    # Cerrar la ventana de la selección ROI después de hacer la selección
    cv2.destroyWindow("Seleccionar área de interés")
    
    # Esperamos a que presiones la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerramos todas las ventanas de OpenCV
vid.release()
cv2.destroyAllWindows()
