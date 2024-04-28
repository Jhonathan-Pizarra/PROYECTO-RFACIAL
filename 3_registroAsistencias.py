import cv2
import os
import numpy as np
from datetime import datetime

# Ruta donde se almacenan los datos de entrenamiento
dataPath = 'imagenes' 
# Índice de la cámara a utilizar (0 para la cámara predeterminada)
numCamara = 0

# Cargar los nombres de las imágenes (etiquetas)
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# Cargar el modelo de reconocimiento facial previamente entrenado
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(numCamara, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Registro de asistencia
asistencia = {}

try:
    while True:
        # Leer un fotograma del video
        ret, frame = cap.read()
        if ret == False:
            break
        
        # Convertir el fotograma a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        # Detectar rostros en el fotograma
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:  # Si no se detecta ningún rostro
            cv2.putText(frame, 'No está en el puesto', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                # Recortar la región de interés (rostro)
                rostro = auxFrame[y:y+h, x:x+w]
                # Redimensionar el rostro para que coincida con el tamaño utilizado durante el entrenamiento
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                
                # Realizar la predicción utilizando el modelo de reconocimiento facial
                label, confidence = face_recognizer.predict(rostro)

                # Mostrar el resultado de la predicción en la imagen
                if confidence < 70: # Umbral de confianza ajustable
                    nombre_persona = imagePaths[label]
                    # Registrar la hora de entrada si es una persona reconocida
                    hora_entrada = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if nombre_persona not in asistencia:
                        asistencia[nombre_persona] = [hora_entrada]
                    else:
                        asistencia[nombre_persona].append(hora_entrada)
                    print("Registro de asistencia actualizado para {} a las {}".format(nombre_persona, hora_entrada))  # Mensaje con la hora de entrada
                    # Mostrar el nombre de la persona reconocida
                    cv2.putText(frame, '{}'.format(nombre_persona), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                    # Dibujar un rectángulo alrededor del rostro reconocido
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    # Mostrar "Desconocido" si la confianza es baja
                    cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    # Dibujar un rectángulo alrededor del rostro no reconocido
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Mostrar el fotograma con los rostros detectados y reconocidos
        cv2.imshow('frame', frame)
        # Esperar 1 milisegundo para la próxima iteración del bucle
        k = cv2.waitKey(1)
        # Si se presiona la tecla 'Esc', salir del bucle
        if k == 27:
            break
finally:
    # Liberar los recursos de la cámara
    cap.release()
    cv2.destroyAllWindows()

    # Obtener la ruta absoluta del directorio actual
    current_directory = os.getcwd()
    # Definir la ruta completa del archivo de registro de asistencia
    registro_filepath = os.path.join(current_directory, "registro_asistencia.txt")

    # Guardar el registro de asistencia en el archivo de texto
    with open(registro_filepath, "w") as registro_file:
        registro_file.write("Registro de Asistencia:\n")
        for nombre_persona, horas in asistencia.items():
            registro_file.write("{}: {}\n".format(nombre_persona, ", ".join(horas)))
