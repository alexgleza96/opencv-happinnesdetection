import cv2
import os
import numpy as np
import pyaudio
import wave

def emotionImage(emotion):
    if emotion == 'Felicidad':
        image = cv2.imread('fel_res.jpeg')
        pathaudio = r"C:\Users\Alex\PycharmProjects\opencv\felicidad.wav"
        diraudio = os.path.dirname(__file__)
        pathaudio = os.path.join(diraudio,r"C:\Users\Alex\PycharmProjects\opencv\felicidad.wav")
        f = wave.open(pathaudio,"rb")
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(f.getsampwidth()), channels=f.getnchannels(),
                        rate=f.getframerate(), output=True)
        datos = f.readframes(1024)
        while datos:
            stream.write(datos)
            datos = f.readframes(1024)
        stream.stop_stream()
        stream.close()
        p.terminate()
    else:
        image = cv2.imread('no_iden_res.jpeg')
    return image


# ----------- Método usado para el entrenamiento y lectura del modelo ----------
method = 'LBPH'
emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
emotion_recognizer.read('modelo' + method + '.xml')
# --------------------------------------------------------------------------------
dataPath = r"C:\Users\Alex\PycharmProjects\opencv\Data"
dirname = os.path.dirname(__file__)
dataPath = os.path.join(dirname,r"C:\Users\Alex\PycharmProjects\opencv\Data")

imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# ------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)
        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        if result[1] < 60:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image = emotionImage(imagePaths[result[0]])
            nFrame = cv2.hconcat([frame, image])
        else:
            cv2.putText(frame, 'No identificado', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])
    cv2.imshow('nFrame', nFrame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()