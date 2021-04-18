import cv2
import os
import numpy as np
import time


def obtenerModelo(method, facesData, labels):
    emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Entrenando el reconocedor de rostros
    print("Entrenando ( " + method + " )...")
    inicio = time.time()
    emotion_recognizer.train(facesData, np.array(labels))
    tiempoEntrenamiento = time.time() - inicio
    print("Tiempo de entrenamiento ( " + method + " ): ", tiempoEntrenamiento)
    # Almacenando el modelo obtenido
    emotion_recognizer.write("modelo" + method + ".xml")

dataPath = r"C:\Users\Alex\PycharmProjects\opencv\Data"
dirname = os.path.dirname(__file__)
dataPath = os.path.join(dirname,r"C:\Users\Alex\PycharmProjects\opencv\Data")
emotionsList = os.listdir(dataPath)

print('Lista de personas: ', emotionsList)
# labels son las etiquetas de cada uno de los rostros correspondientes a cada emoción
labels = []
# en facesdata se almacenarán todos los rostros con sus diferentes emociones
facesData = []
label = 0
for nameDir in emotionsList:
    emotionsPath = dataPath + '/' + nameDir
    for fileName in os.listdir(emotionsPath):
        labels.append(label)
        facesData.append(cv2.imread(emotionsPath + '/' + fileName, 0))

    label = label + 1
obtenerModelo('LBPH', facesData, labels)