import numpy as np
import cv2
from PIL import Image
import time
import serial
from simple_pid import PID
import threading

accion_actual = None

def input_thread():
    global accion_actual
    acciones = ['orientar', 'acercar', 'avanzar', 'orientarcentro', 'acercarcentro', 'None']
    accion = input("Ingrese acción (orientar/acercar/avanzar/orientarcentro/acercarcentro/None): ").strip()
    if accion in acciones:
        accion_actual = accion
        print(f"→ Acción: {accion_actual}")
    else:
        print("Acción invalida")

def range_to_mask(image, lower_bound, higher_bound):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower_bound, higher_bound)
    img_masked = cv2.bitwise_and(image, image, mask=mask)
    return img_masked

def range_to_bb(image, lower_bound, higher_bound):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower_bound, higher_bound)
    mask_ = Image.fromarray(mask)
    bounding_box = mask_.getbbox()
    return bounding_box

def graph_bb(image, image_masked, bounding_box, text):
    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        cv2.rectangle(image_masked, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_masked, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

def bb_center(bounding_box):
    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        center = np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2])
        return center
    else:
        return np.array([0, 0])
    
def graph_vector(image, image_masked, coord1, coord0):
    x0, y0 = map(int, coord0)
    x1, y1 = map(int, coord1)
    cv2.line(image_masked, (x1, y1), (x0, y0), (255,255, 255), 2)
    cv2.line(image, (x1, y1), (x0, y0), (0, 0, 0), 2)

def hsv_to_cv(hue, saturation, v):
    huecv = (179/355) * hue
    saturationcv = (255/100) * saturation
    vcv = (255/100) * v
    return np.array([huecv, saturationcv, vcv])


ser = serial.Serial("COM7", baudrate=38400, timeout=1)
velocidad = 0

def enviar_velocidad(rpml, rpmr):
    velocidad = f"a{int(rpml)}b{int(rpmr)};"
    encoded = velocidad.encode()
    ser.write(encoded)

vid = cv2.VideoCapture(1) 

pid_dist = PID(0.5, 0.001, 0.5, setpoint=32)
pid_a = PID(1, 0, 0, setpoint=0)
no_llego = True
contador = 0

def avanzar():
    enviar_velocidad(-20, 20)

def orientarse(angle):
    ctrl_a = pid_a(angle)
    rpml = -ctrl_a
    rpmr = ctrl_a
    enviar_velocidad(rpml, rpmr)

def acercar(dist, angle):
    contrl = pid_dist(dist)
    ctrl_a = pid_a(angle)
    rpml = - contrl - ctrl_a
    rpmr = contrl + ctrl_a
    enviar_velocidad(rpml, rpmr)

threading.Thread(target=input_thread, daemon=True).start()
while(True):
          
    ret, img = vid.read()

    img_arco_der = img[200:450, 600:700]
    img_arco_izq = img[200:450, 0:100]


    arco_lb = np.array([106, 110, 81])
    arco_ub = np.array([111, 160, 95])

    rojo_lb1 = np.array([0, 120, 100])
    rojo_ub1 = np.array([10, 255, 255])
    rojo_lb2 = np.array([170, 120, 100])
    rojo_ub2 = np.array([179, 255, 255])

    azul_lb = np.array([105, 150, 80])
    azul_ub = np.array([115, 200, 150])

    pelota_lb = np.array([15, 150, 120])
    pelota_ub = np.array([25, 200, 200])


    img_masked_blue = range_to_mask(img, azul_lb, azul_ub)
    img_masked_red1 = range_to_mask(img, rojo_lb1, rojo_ub1)
    img_masked_red2 = range_to_mask(img, rojo_lb2, rojo_ub2)
    img_masked_red = img_masked_red1 + img_masked_red2
    img_masked_pelota = range_to_mask(img, pelota_lb, pelota_ub)
    img_masked_arco = range_to_mask(img, arco_lb, arco_ub)
    img_masked = img_masked_blue + img_masked_red + img_masked_pelota + img_masked_arco


    bounding_box_blue = range_to_bb(img, azul_lb, azul_ub)
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_ = Image.fromarray(img_masked_red)
    bounding_box_red = mask_.getbbox()
    bounding_box_pelota = range_to_bb(img, pelota_lb, pelota_ub)
    bounding_box_arco = range_to_bb(img, arco_lb, arco_ub)


    graph_bb(img, img_masked, bounding_box_blue, "azul")
    graph_bb(img, img_masked, bounding_box_red, "rojo")
    graph_bb(img, img_masked, bounding_box_pelota, "pelota")


    centro_azul = bb_center(bounding_box_blue)
    centro_rojo = bb_center(bounding_box_red)
    centro_pelota = bb_center(bounding_box_pelota)


    vector_direccion = centro_azul - centro_rojo
    mag_dir = np.linalg.norm(vector_direccion)
    vector_a_pelota = centro_pelota - centro_rojo
    mag_pelota = np.linalg.norm(vector_a_pelota)


    theta = np.arctan2(vector_direccion[1], vector_direccion[0])
    fi = np.arctan2(vector_a_pelota[1], vector_a_pelota[0])
    angulo = theta - fi

    
    if (angulo > np.pi):
        angulo -= 2 * np.pi
    elif (angulo <= -np.pi):
        angulo += 2 * np.pi
    angulo = np.rad2deg(angulo)

    print(f"Distancia(px) y angulo(deg) : {round(mag_pelota, 3)},  {round(angulo, 3)}")

    graph_vector(img, img_masked, centro_pelota, centro_rojo)
    graph_vector(img, img_masked, centro_rojo, centro_azul)


    cv2.imshow("recorte", img_arco_izq)
    cv2.imshow("recorte2", img_arco_der)
    cv2.imshow("mask", img_masked)
    cv2.imshow('original', img)
    contador += 1
    
    if accion_actual == "None":
        enviar_velocidad(0, 0)
    if accion_actual == "orientar":
        orientarse(angulo)
    if accion_actual == "acercar":
        acercar(mag_pelota, angulo)
    if accion_actual == "avanzar":
        avanzar()
    if accion_actual == "orientarcentro":
        orientarse(angulo_centro)
    if accion_actual == "acercarcentro":
        acercar(distancia_centro, angulo_centro)

    # contrl = pid_dist(mag_pelota)
    # if abs(angulo)> 0 and no_llego and contador>=100:

    #     ctrl_a = pid_a(angulo)
    #     rpml = -ctrl_a
    #     rpmr = ctrl_a

    #     velocidad = f"a{int(rpml)}b{int(rpmr)};"
    #     encoded = velocidad.encode()
    #     ser.write(encoded)

    # elif contador >= 100:
    #     no_llego = False
    #     ctrl_a = pid_a(angulo)
    #     rpml = contrl - ctrl_a
    #     rpmr = contrl + ctrl_a

    #     velocidad = f"a{int(rpml)}b{int(rpmr)};"
    #     encoded = velocidad.encode()
    #     ser.write(encoded)

    if cv2.waitKey(1) & 0xFF == 27:
        velocidad = f"a{0}b{0};"
        encoded = velocidad.encode()
        ser.write(encoded)
        time.sleep(1)
        break
ser.close()
cv2.destroyAllWindows()