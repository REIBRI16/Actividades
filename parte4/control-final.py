import numpy as np
import cv2
from PIL import Image
import time
import serial
from simple_pid import PID

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

#path1 = "gameExample_1.png"
# path1 = "capturavideo.png"
# path2 = "PerceptionDataset.mp4"
# vid = cv2.VideoCapture(path2)
vid = cv2.VideoCapture(0) 

pid_dist = PID(0.5, 0.001, 0.5, setpoint=32)
pid_a = PID(1.2, 0.01, 0.1, setpoint=0)
no_llego = True

while(True):
          
    ret, img = vid.read()
    # img = cv2.resize(img, (700, 600))
    #img = cv2.imread(path1)

    img_arco_der = img[200:450, 600:700]
    img_arco_izq = img[200:450, 0:100]



    arco_lb = np.array([106, 110, 81])
    arco_ub = np.array([111, 160, 95])

    # rojo_lb = np.array([0, 120, 100])
    # rojo_ub = np.array([10, 255, 200])

    verde_lb = np.array([0, 120, 100])
    verde_ub = np.array([10, 255, 200])


    azul_lb = np.array([100, 175, 119])
    azul_ub = np.array([106, 200, 140])

    pelota_lb = np.array([20, 150, 150])
    pelota_ub = np.array([30, 255, 255])




    img_masked_blue = range_to_mask(img, azul_lb, azul_ub)
    img_masked_red = range_to_mask(img, rojo_lb, rojo_ub)
    img_masked_pelota = range_to_mask(img, pelota_lb, pelota_ub)
    img_masked_arco = range_to_mask(img, arco_lb, arco_ub)
    img_masked = img_masked_blue + img_masked_red + img_masked_pelota + img_masked_arco


    bounding_box_blue = range_to_bb(img, azul_lb, azul_ub)
    bounding_box_red = range_to_bb(img, rojo_lb, rojo_ub)
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

    if angulo > np.pi:
        angulo = 2 * np.pi - angulo

    print(f"Distancia(px) y angulo(rad) : {round(mag_pelota, 3)},  {round(angulo, 3)}")

    graph_vector(img, img_masked, centro_pelota, centro_rojo)
    graph_vector(img, img_masked, centro_rojo, centro_azul)


    # cv2.imshow("recorte", img_arco_izq)
    # cv2.imshow("recorte2", img_arco_der)
    cv2.imshow("mask", img_masked)
    cv2.imshow('original', img)

    contrl = pid_dist(mag_pelota)

    while(abs(angulo)> 2 and no_llego):

        ctrl_a = pid_a(angulo)
        rpml = ctrl_a
        rpmr = -ctrl_a
    no_llego = False
    ctrl_a = pid_a(angulo)
    rpml = ctrl_a
    rpmr = -ctrl_a
    rpml = -contrl + ctrl_a
    rpmr = -contrl - ctrl_a

    velocidad = f"a{rpml}b{rpmr}"
    encoded = velocidad.encode()
    ser.write(encoded)

    if cv2.waitKey(1) & 0xFF == 27:
        break
ser.close()
cv2.destroyAllWindows()