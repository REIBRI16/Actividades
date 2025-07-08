import time

import cv2
import serial


message_example = "a100b50;"
message = input("Ingresa una velocidad(rpm): ")
ser = serial.Serial("COM8", baudrate=38400, timeout=1)
time.sleep(1)

not_salir = True

while not_salir:
    encoded = message.encode()
    ser.write(encoded)
    message = input("Ingresa una velocidad(rpm): ")

    if message == "END":
        not_salir = False
        break

ser.close()
