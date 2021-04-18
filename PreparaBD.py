import cv2
import os
import imutils

# Creando carpeta con emocion de felicidad
emotionName = 'Felicidad'

dataPath = os.path.dirname(__file__)
emotionsPath = dataPath + '\\' +'Data' + '\\' + emotionName


if not os.path.exists(emotionsPath):
    print('Carpeta creada: ',emotionsPath)
    os.makedirs(emotionsPath)
# Grabando y guardando la expresion de felicidad.
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# Se utiliza el clasificador facial pre entrenado haarcascade
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0
while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    # Con  el modulo detectMultiScale se identifican los rostros
    faces = faceClassif.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(emotionsPath + '/rotro_{}.jpg'.format(count),rostro)
        count = count + 1
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    # Se detiene la grabacion si se presiona esc = ascii 27 o si se llegan a 300 rostros
    if k == 27 or count >= 300:
        break
cap.release()
cv2.destroyAllWindows()