import numpy as np
import cv2
from PIL import Image
import time
import serial
from simple_pid import PID
import threading

accion_actual = None
fin = False
def input_thread():
    global accion_actual, fin, first
    acciones = ["orientar", "acercar", "avanzar", "orientarcentro", "acercarcentro", "None", "fin", "golpe", "arco"]
    while not fin:
        accion = input("Ingrese acción (orientar/acercar/avanzar/orientarcentro/acercarcentro/None/fin/golpe/arco): ").strip()
        if accion in acciones:
            accion_anterior = accion_actual
            accion_actual = accion
            if accion_anterior != accion_actual:
                first = True
            print(f"Acción: {accion_actual}")
            if accion == "fin":
                enviar_velocidad(0, 0)
                fin = True
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

def magyang(centro_azul, centro_rojo, centro):
    vector_direccion = centro_azul - centro_rojo
    mag_dir = np.linalg.norm(vector_direccion)
    vector_a_pelota = centro - centro_rojo
    mag_pelota = np.linalg.norm(vector_a_pelota)


    theta = np.arctan2(vector_direccion[1], vector_direccion[0])
    fi = np.arctan2(vector_a_pelota[1], vector_a_pelota[0])
    angulo = theta - fi

    
    if (angulo > np.pi):
        angulo -= 2 * np.pi
    elif (angulo <= -np.pi):
        angulo += 2 * np.pi
    angulo = np.rad2deg(angulo)
    return (mag_pelota, angulo)


ser = serial.Serial("COM8", baudrate=38400, timeout=1)
velocidad = 0

def enviar_velocidad(rpml, rpmr):
    time.sleep(0.25)
    velocidad = f"a{int(rpml)}b{int(rpmr)};"
    encoded = velocidad.encode()
    ser.write(encoded)

vid = cv2.VideoCapture(1, cv2.CAP_DSHOW) 

pid_dist = PID(0.5, 0.0001, 0.05, setpoint=25)
pid_a = PID(0.5, 0, 0, setpoint=0)
no_llego = True
contador = 0

def avanzar():
    enviar_velocidad(50, 50)

def orientarse(angle):
    ctrl_a = pid_a(angle)
    rpml = -ctrl_a
    rpmr = ctrl_a
    enviar_velocidad(rpml, rpmr)

def acercar(dist, angle):
    contrl = pid_dist(dist)
    ctrl_a = pid_a(angle)
    rpml = contrl - ctrl_a
    rpmr = contrl + ctrl_a
    enviar_velocidad(rpml, rpmr)

threading.Thread(target=input_thread, daemon=True).start()
while(True):
          
    ret, img = vid.read()
    height = img.shape[0]
    width = img.shape[1]
    centro = np.array([width/2, height/2])
    arcoizq = np.array([0, height/2])
    arcoder = np.array([width, height/2])


    arco_lb = np.array([106, 110, 81])
    arco_ub = np.array([111, 160, 95])

    rojo_lb1 = np.array([174, 180, 129])
    rojo_ub1 = np.array([177, 198, 144])

    rojo_lb2 = np.array([174, 180, 129])
    rojo_ub2 = np.array([177, 198, 144])

    azul_lb = np.array([105, 163, 123])
    azul_ub = np.array([108, 184, 136])

    pelota_lb = np.array([20, 199, 175])
    pelota_ub = np.array([24, 215, 220])


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


    mag_pelota, angulo = magyang(centro_azul, centro_rojo, centro_pelota)
    distancia_centro, angulo_centro = magyang(centro_azul, centro_rojo, centro)
    distancia_arco, angulo_arco = magyang(centro_azul, centro_rojo, arcoizq)

    #print(f"Distancia(px) y angulo(deg) : {round(mag_pelota, 3)},  {round(angulo, 3)}")

    graph_vector(img, img_masked, centro_pelota, centro_rojo)
    graph_vector(img, img_masked, arcoizq, centro_rojo)
    graph_vector(img, img_masked, arcoder, centro_rojo)
    graph_vector(img, img_masked, centro, centro_rojo)
    graph_vector(img, img_masked, centro_rojo, centro_azul)


    #cv2.imshow("recorte", img_arco_izq)
    #cv2.imshow("recorte2", img_arco_der)
    cv2.imshow("mask", img_masked)
    cv2.imshow('original', img)
    contador += 1
    
    if np.abs(angulo) >= 10:
        angulobien = False
    else:
        angulobien = True
    
    if np.abs(angulo_centro) >= 10:
        angulocentrobien = False
    else:
        angulocentrobien = True

    if np.abs(angulo_arco) >= 10:
        anguloarcobien = False
    else:
        anguloarcobien = True

    
    if accion_actual == "None":
        enviar_velocidad(0, 0)
    if accion_actual == "orientar":
        orientarse(angulo)
    if accion_actual == "acercar":
        if angulobien:
            if mag_pelota > 100:
                acercar(mag_pelota, angulo)
            else:
                orientarse(angulo)
        else:
            orientarse(angulo)
    if accion_actual == "avanzar":
        enviar_velocidad(50, 50)
    if accion_actual == "orientarcentro":
        orientarse(angulo_centro)
    if accion_actual == "acercarcentro":
        if angulocentrobien:
            if distancia_centro > 50:
                acercar(distancia_centro, angulo_centro)
            else:
                orientarse(angulo_centro)
        else:
            orientarse(angulo_centro)
    if accion_actual == "golpe":
        while first:
            if angulobien:
                if mag_pelota > 15:
                    acercar(mag_pelota, angulo)
                else:
                    first = False
                    orientarse(angulo)
            else:
                orientarse(angulo)
    if accion_actual == "arco":
        if angulobien:
            if mag_pelota > 15:
                acercar(mag_pelota, angulo)
            else:
                if anguloarcobien:
                    if distancia_arco > 20:
                        acercar(distancia_arco, angulo_arco)
                    else:
                        orientarse(angulo_arco)
                else:
                    orientarse(angulo)
        else:
            orientarse(angulo)



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