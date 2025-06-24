import numpy as np
import cv2
import time
import math
import _thread
from simple_pid import PID
#import serial

class irb_sim:
	robot1_pos = np.array([0,0,0])
	robot2_pos = np.array([0,0,0])
	ball_pos = np.array([0,0,0])
	game_field = cv2.imread('field.jpg')
 
	def draw(self):
		self.game_field = cv2.imread('field.jpg')
		r1_fx = int(self.robot1_pos[0] + 30*math.cos(self.robot1_pos[2]))
		r1_fy = int(self.robot1_pos[1] + 30*math.sin(self.robot1_pos[2]))
		
		r1_bx = int(self.robot1_pos[0] + 0*math.cos(self.robot1_pos[2]))
		r1_by = int(self.robot1_pos[1] + 0*math.sin(self.robot1_pos[2]))

		cv2.circle(self.game_field,(r1_fx, r1_fy), 10, (0,0,255), -1)
		cv2.circle(self.game_field,(r1_bx, r1_by), 10, (255,0,0), -1)
		# cv2.circle(self.game_field,(200, 200), 10, (150,180,255), -1)
		# cv2.circle(self.game_field,(170, 200), 10, (0,255,0), -1)
		cv2.circle(self.game_field,(self.ball_pos[0],self.ball_pos[0]), 10, (0, 255, 255), -1)
		
global frame
global nClick

global Robot1hsvFrontColor
global Robot1hsvBackColor
global Robot2hsvFrontColor
global Robot2hsvBackColor
global BallhsvBallColor 

global Robot1goalx
global Robot1goaly
global Robot2goalx
global Robot2goaly

global xcr_Robot1ColorBack
global ycr_Robot1ColorBack	
global xcr_Robot1ColorFront
global ycr_Robot1ColorFront
global xcr_BallColor
global ycr_BallColor
global xcr_Robot2ColorBack
global ycr_Robot2ColorBack 
global xcr_Robot2ColorFront
global ycr_Robot2ColorFront

global theta
global dist
global cmd
global r1Ar
global r1Al

Robot1hsvFrontColor = np.array([0,0,0])
Robot1hsvBackColor = np.array([0,0,0])
Robot2hsvFrontColor = np.array([0,0,0])
Robot2hsvBackColor = np.array([0,0,0])
BallhsvBallColor = np.array([0,0,0]) 

Robot1goalx = 0
Robot1goaly = 0
Robot2goalx = 0
Robot2goaly = 0

xcr_Robot1ColorBack = 0
ycr_Robot1ColorBack = 0	
xcr_Robot1ColorFront = 0
ycr_Robot1ColorFront = 0
xcr_BallColor = 0
ycr_BallColor = 0	
xcr_Robot2ColorBack = 0
ycr_Robot2ColorBack = 0
xcr_Robot2ColorFront = 0
ycr_Robot2ColorFront = 0

theta = 0
dist = 0
cmd = "0,0"
r1Al = 0
r1Ar = 0

LowerColorError = np.array([-10,-35,-35])
UpperColorError = np.array([10,35,35])


global grupo1
grupo1 = irb_sim()

def start(delay):
	global r1Ar
	global r1Al
	#r1Ar: PWM rueda derecha
	#r1Al: PWM rueda izquierda
	#dit: distancia robot-pelota
	#theta: angulo robot-pelota
	time.sleep(2)
	# r1Al = 20
	# r1Ar = 20
	# time.sleep(5)
	# r1Al = 10
	# r1Ar = 3
	# time.sleep(5)
	# r1Al = -8.85
	# r1Ar = 8.85
	# time.sleep(5)
	# r1Al = 0
	# r1Ar = 0
	print(theta)
	print(dist)
	pid_dist = PID(0.5, 0.001, 0.5, setpoint=32)
	pid_a = PID(1.2, 0.01, 0.1, setpoint=0)
	no_llego = True
	while(True):
		contrl = pid_dist(dist)

		while(abs(theta)> 2 and no_llego):

			ctrl_a = pid_a(theta)
			r1Al = ctrl_a
			r1Ar = -ctrl_a
			print("Angulo: " + str(theta))
			print("Distancia: " + str(dist))
		no_llego = False
		ctrl_a = pid_a(theta)
		r1Al = ctrl_a
		r1Ar = -ctrl_a
		r1Al = -contrl + ctrl_a
		r1Ar = -contrl - ctrl_a
		#Ciclo de programacion
		print("Angulo: " + str(theta))
		print("Distancia: " + str(dist))
	

cv2.namedWindow('realidad', cv2.WINDOW_AUTOSIZE)
		

def sim_run(delay):
	global grupo1
	global r1Ar
	global r1Al
	r1x = 100
	r1y = 100
	r1a = 0  #ANGULOOOOO
	rbx = 350
	rby = 350
	
	
	grupo1.robot1_pos = np.array([r1x,r1y,r1a])
	grupo1.ball_pos = np.array([rbx,rby,0])
	grupo1.draw()
	t = 0
	t_step = 0.027
	m = 0.5
	
	r1Vl = 0
	r1Vr = 0
	r1Al = 0
	r1Ar = 0

	while(True):
		r1Vr = r1Vr + r1Ar*t_step*t_step/2 - r1Vr*0.02
		r1Vl = r1Vl + r1Al*t_step*t_step/2 - r1Vl*0.02
		
		r1x_p = 0.5*(r1Vr + r1Vl)*math.cos(r1a)
		r1y_p = 0.5*(r1Vr + r1Vl)*math.sin(r1a)
		r1a_p = 0.5*(r1Vr - r1Vl)/15
		
		r1x = r1x + r1x_p*t_step
		r1y = r1y + r1y_p*t_step
		r1a = r1a + r1a_p*t_step
		
		grupo1.robot1_pos = np.array([r1x,r1y,r1a])
		grupo1.draw()
			
		
		

cv2.namedWindow('res',  cv2.WINDOW_AUTOSIZE )	
cv2.moveWindow('res', 700, 100)

_thread.start_new_thread(sim_run, (1,))
_thread.start_new_thread ( start, (1,) )

frame = grupo1.game_field
frame = grupo1.game_field
Ypx, Xpx, ch = frame.shape
Ypx = int(Ypx/2)
Xpx = int(Xpx/2)


while(True):
	frame = grupo1.game_field	
	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	Robot1LowerColorBack = np.array([120,255,255]) + LowerColorError
	Robot1UpperColorBack = np.array([120,255,255]) + UpperColorError	
	Robot1LowerColorFront = np.array([0,255,255]) + LowerColorError
	Robot1UpperColorFront = np.array([0,255,255]) + UpperColorError	

	BallLowerColor = np.array([30,255,255]) + LowerColorError
	BallUpperColor = np.array([30,255,255]) + UpperColorError	

	Robot2LowerColorBack = Robot2hsvBackColor + LowerColorError
	Robot2UpperColorBack = Robot2hsvBackColor + UpperColorError		
	Robot2LowerColorFront = Robot2hsvFrontColor + LowerColorError
	Robot2UpperColorFront = Robot2hsvFrontColor + UpperColorError
	
    # Threshold for HSV image
	Robot1ColorBackMask = cv2.inRange(hsv, Robot1LowerColorBack, Robot1UpperColorBack)
	Robot1ColorBackBlur = cv2.GaussianBlur(Robot1ColorBackMask,(31,31),0)
	Robot1ColorBackRet, Robot1ColorBackOTSUMask = cv2.threshold(Robot1ColorBackBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	Robot1ColorFrontMask = cv2.inRange(hsv, Robot1LowerColorFront, Robot1UpperColorFront)
	Robot1ColorFrontBlur = cv2.GaussianBlur(Robot1ColorFrontMask,(31,31),0)
	Robot1ColorFrontRet, Robot1ColorFrontOTSUMask = cv2.threshold(Robot1ColorFrontBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	BallColorMask = cv2.inRange(hsv, BallLowerColor, BallUpperColor)
	BallColorBlur = cv2.GaussianBlur(BallColorMask,(31,31),0)
	BallColorRet, BallColorOTSUMask = cv2.threshold(BallColorBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	Robot2ColorBackMask = cv2.inRange(hsv, Robot2LowerColorBack, Robot2UpperColorBack)
	Robot2ColorBackBlur = cv2.GaussianBlur(Robot2ColorBackMask,(31,31),0)
	Robot2ColorBackRet, Robot2ColorBackOTSUMask = cv2.threshold(Robot2ColorBackBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	Robot2ColorFrontMask = cv2.inRange(hsv, Robot2LowerColorFront, Robot2UpperColorFront)
	Robot2ColorFrontBlur = cv2.GaussianBlur(Robot2ColorFrontMask,(31,31),0)
	Robot2ColorFrontRet, Robot2ColorFrontOTSUMask = cv2.threshold(Robot2ColorFrontBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	
 
	# Bitwise-AND mask and original image
	Robot1ColorBackRes = cv2.bitwise_and(frame,frame, mask= Robot1ColorBackOTSUMask)
	Robot1ColorFrontRes = cv2.bitwise_and(frame,frame, mask= Robot1ColorFrontOTSUMask)	
	BallColorRes = cv2.bitwise_and(frame,frame, mask= BallColorOTSUMask)	
	Robot2ColorBackRes = cv2.bitwise_and(frame,frame, mask= Robot2ColorBackOTSUMask)
	Robot2ColorFrontRes = cv2.bitwise_and(frame,frame, mask= Robot2ColorFrontOTSUMask)
		
	res = Robot1ColorBackRes + Robot1ColorFrontRes + Robot2ColorBackRes + Robot2ColorFrontRes + BallColorRes

	Robot1ColorBackMoments = cv2.moments(Robot1ColorBackOTSUMask, 1)
	m00_Robot1ColorBack = int(Robot1ColorBackMoments['m00'])
	m01_Robot1ColorBack = int(Robot1ColorBackMoments['m01'])
	m10_Robot1ColorBack = int(Robot1ColorBackMoments['m10'])	
	Robot1ColorFrontMoments = cv2.moments(Robot1ColorFrontOTSUMask, 1)
	m00_Robot1ColorFront = int(Robot1ColorFrontMoments['m00'])
	m01_Robot1ColorFront = int(Robot1ColorFrontMoments['m01'])
	m10_Robot1ColorFront = int(Robot1ColorFrontMoments['m10'])
	
	BallColorMoments = cv2.moments(BallColorOTSUMask, 1)
	m00_BallColor = int(BallColorMoments['m00'])
	m01_BallColor = int(BallColorMoments['m01'])
	m10_BallColor = int(BallColorMoments['m10'])	
	
	Robot2ColorBackMoments = cv2.moments(Robot2ColorBackOTSUMask, 1)
	m00_Robot2ColorBack = int(Robot2ColorBackMoments['m00'])
	m01_Robot2ColorBack = int(Robot2ColorBackMoments['m01'])
	m10_Robot2ColorBack = int(Robot2ColorBackMoments['m10'])	
	Robot2ColorFrontMoments = cv2.moments(Robot2ColorFrontOTSUMask, 1)
	m00_Robot2ColorFront = int(Robot2ColorFrontMoments['m00'])
	m01_Robot2ColorFront = int(Robot2ColorFrontMoments['m01'])
	m10_Robot2ColorFront = int(Robot2ColorFrontMoments['m10'])
	

	if(m00_BallColor*m00_Robot1ColorBack*m00_Robot1ColorFront*m00_Robot2ColorBack*m00_Robot2ColorFront > 0):
		
		xc_Robot1ColorBack = int(m10_Robot1ColorBack/m00_Robot1ColorBack)
		yc_Robot1ColorBack = int(m01_Robot1ColorBack/m00_Robot1ColorBack)	
		xc_Robot1ColorFront = int(m10_Robot1ColorFront/m00_Robot1ColorFront)
		yc_Robot1ColorFront = int(m01_Robot1ColorFront/m00_Robot1ColorFront) 
		
		xc_BallColor = int(m10_BallColor/m00_BallColor)
		yc_BallColor = int(m01_BallColor/m00_BallColor)	
		
		xc_Robot2ColorBack = int(m10_Robot2ColorBack/m00_Robot2ColorBack)
		yc_Robot2ColorBack = int(m01_Robot2ColorBack/m00_Robot2ColorBack) 
		xc_Robot2ColorFront = int(m10_Robot2ColorFront/m00_Robot2ColorFront)
		yc_Robot2ColorFront = int(m01_Robot2ColorFront/m00_Robot2ColorFront) 

		xcr_Robot1ColorBack = xc_Robot1ColorBack - Xpx
		ycr_Robot1ColorBack = yc_Robot1ColorBack - Ypx	
		xcr_Robot1ColorFront = xc_Robot1ColorFront - Xpx
		ycr_Robot1ColorFront = yc_Robot1ColorFront - Ypx
		
		xcr_BallColor = xc_BallColor - Xpx
		ycr_BallColor = yc_BallColor - Ypx	
		
		xcr_Robot2ColorBack = xc_Robot2ColorBack - Xpx
		ycr_Robot2ColorBack = yc_Robot2ColorBack - Ypx
		xcr_Robot2ColorFront = xc_Robot2ColorFront - Xpx
		ycr_Robot2ColorFront = yc_Robot2ColorFront - Ypx		
		
		ax = xcr_Robot1ColorFront - xcr_Robot1ColorBack
		ay = ycr_Robot1ColorFront - ycr_Robot1ColorBack
		bx = xcr_BallColor - xcr_Robot1ColorBack
		by = ycr_BallColor - ycr_Robot1ColorBack
		
		r1mx = int((xcr_Robot1ColorFront + xcr_Robot1ColorBack)/2)
		r1my = int((ycr_Robot1ColorFront + ycr_Robot1ColorBack)/2)
		if(ay != 0):
			m = -ax/ay
			n = r1my - m*r1mx
			if(ycr_Robot1ColorFront < ycr_Robot1ColorBack):
				p1x = int(r1mx + 10) + Xpx
				p1y = int(m*(p1x-Xpx)+ n) + Ypx
				p2x = int(r1mx - 10) + Xpx
				p2y = int(m*(p2x-Xpx)+ n) + Ypx
			else:
				p1x = int(r1mx - 10) + Xpx
				p1y = int(m*(p1x-Xpx)+ n) + Ypx
				p2x = int(r1mx + 10) + Xpx
				p2y = int(m*(p2x-Xpx)+ n) + Ypx
		else:
			m = 0
			p1x = r1mx + Xpx
			p1y = r1my + 10 + Ypx
			p2x = r1mx + Xpx
			p2y = r1my - 10 + Ypx
		
		am = math.sqrt(math.pow(ax, 2) + math.pow(ay, 2))
		bm = math.sqrt(math.pow(bx, 2) + math.pow(by, 2)) 
		ab = ax*bx + ay*by
		
		d1 = int(math.sqrt(math.pow(xcr_BallColor - p1x + Xpx, 2) + math.pow(ycr_BallColor - p1y + Ypx, 2)))
		d2 = int(math.sqrt(math.pow(xcr_BallColor - p2x + Xpx, 2) + math.pow(ycr_BallColor - p2y + Ypx, 2)))
		dist = int(math.sqrt(math.pow(xcr_BallColor - xcr_Robot1ColorFront, 2) + math.pow(ycr_BallColor - ycr_Robot1ColorFront, 2)))
		
		theta = 0
		if(am*bm > 0) and ab/(am*bm)<=1:  ##agregué por error por arcocoseno de 1.0000000000002 no existe.
			if(d1 < d2):
				theta = int(math.acos(ab/(am*bm))*180/math.pi)
			else:
				# print(ab/(am*bm))
				theta = -int(math.acos(ab/(am*bm))*180/math.pi)
		else:
			theta = 0

			
		cv2.line(res, (xc_Robot1ColorBack, yc_Robot1ColorBack), (xc_Robot1ColorFront, yc_Robot1ColorFront), (255,0,0),3)
		cv2.line(res, (xc_Robot1ColorBack, yc_Robot1ColorBack), (xc_BallColor, yc_BallColor), (255,0,0),3)
		
		cv2.line(res, (Xpx-int(Xpx*3/2), Ypx), (Xpx+int(Xpx*3/2), Ypx), (0,255,0),1)
		cv2.line(res, (Xpx, Ypx-int(Ypx*3/2)), (Xpx, Ypx+int(Ypx*3/2)), (0,255,0),1)
		
		# cv2.line(res, (p1x, p1y), (p2x, p2y), (0,0,255),3)
		
		cv2.circle(res,(xc_Robot1ColorBack,yc_Robot1ColorBack), 3, (0,0,255), -1)
		cv2.circle(res,(xc_Robot1ColorFront,yc_Robot1ColorFront), 3, (0,0,255), -1)
		cv2.circle(res,(xc_BallColor,yc_BallColor), 3, (0,0,255), -1)
		cv2.circle(res,(xc_Robot2ColorBack,yc_Robot2ColorBack), 3, (0,0,255), -1)
		cv2.circle(res,(xc_Robot2ColorFront,yc_Robot2ColorFront), 3, (0,0,255), -1)
		cv2.circle(res,(r1mx + Xpx,r1my + Ypx), 3, (0,0,255), -1)
		
		# cv2.circle(res,(Robot1goalx,Robot1goaly), 10, (0,0,255), -1)
		# cv2.circle(res,(Robot2goalx,Robot2goaly), 10, (0,0,255), -1)
		# cv2.circle(res,(p1x,p1y), 2, (0,0,255), -1)
		# cv2.circle(res,(p2x,p2y), 2, (0,0,255), -1)
				
		cv2.putText(res, "(" + str(xcr_Robot1ColorBack) + "," + str(ycr_Robot1ColorBack) + ")",(xc_Robot1ColorBack,yc_Robot1ColorBack), 5, 1, (255,255,255),1,8,False)
		cv2.putText(res, "(" + str(xcr_Robot1ColorFront) + "," + str(ycr_Robot1ColorFront) + ")",(xc_Robot1ColorFront,yc_Robot1ColorFront), 5, 1, (255,255,255),1,8,False)
		cv2.putText(res, "(" + str(xcr_BallColor) + "," + str(ycr_BallColor) + ")",(xc_BallColor,yc_BallColor), 5, 1, (255,255,255),1,8,False)
		# cv2.putText(res, "(" + str(xcr_Robot2ColorBack) + "," + str(ycr_Robot2ColorBack) + ")",(xc_Robot2ColorBack,yc_Robot2ColorBack), 5, 1, (255,255,255),1,8,False)
		# cv2.putText(res, "(" + str(xcr_Robot2ColorFront) + "," + str(ycr_Robot2ColorFront) + ")",(xc_Robot2ColorFront,yc_Robot2ColorFront), 5, 1, (255,255,255),1,8,False)
		
		cv2.putText(res, "<" + str(theta) + " |" + str(dist),(xc_BallColor+30,yc_BallColor+30), 5, 1, (255,255,255),1,8,False)
		# cv2.putText(res, "P1(" + str(d1) + ")",(p1x,p1y), 5, 1, (0,255,0),1,8,False)
		# cv2.putText(res, "P2(" + str(d2) + ")",(p2x,p2y), 5, 1, (0,255,0),1,8,False)		
		
		
	#cv2.imshow('frame',frame)	
	cv2.imshow('res',res)
	cv2.imshow('realidad', grupo1.game_field)


	if cv2.waitKey(1) & 0xFF == 27:
		break


cv2.destroyAllWindows()
