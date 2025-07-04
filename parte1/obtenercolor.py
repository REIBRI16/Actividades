import cv2
import numpy as np

#CreadoConChatGPT

def circular_mean_hue(hues):
    angles = hues * 2 * np.pi / 180.0  
    sin_sum = np.mean(np.sin(angles))
    cos_sum = np.mean(np.cos(angles))
    mean_ang = np.arctan2(sin_sum, cos_sum)
    if mean_ang < 0:
        mean_ang += 2*np.pi
    mean_deg = mean_ang * 180 / np.pi
    return mean_deg / 2

def get_avg_color_hsv(image, roi):
    x, y, w, h = roi
    roi_img = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_vals = h.flatten().astype(float)
    s_vals = s.flatten().astype(float)
    v_vals = v.flatten().astype(float)
    mean_h = circular_mean_hue(h_vals)
    mean_s = np.mean(s_vals)
    mean_v = np.mean(v_vals)
    return np.array([mean_h, mean_s, mean_v])

vid = cv2.VideoCapture(0)
if not vid.isOpened():
    print("No se pudo abrir la cámara.")
    exit(1)

while True:
    ret, img = vid.read()
    if not ret:
        break

    img = cv2.resize(img, (700, 600))
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == 27:
        print("Programa detenido por usuario.")
        break

    roi = cv2.selectROI("Seleccionar área de interés", img,
                        fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Seleccionar área de interés")

    if roi == (0, 0, 0, 0):
        print("Selección cancelada. Saliendo.")
        break

    avg_hsv = get_avg_color_hsv(img, roi)
    print(f"Promedio HSV circular en la región seleccionada: {avg_hsv}")

vid.release()
cv2.destroyAllWindows()
