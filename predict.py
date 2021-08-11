from tensorflow.keras.models import model_from_json
import operator
import cv2
from numpy import unicode
from text_to_speech import speak
from gtts import gTTS
import playsound
import os
import winsound


json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])

    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    roi = frame[y1:y2, x1:x2]

    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    prediction = {'ZERO': result[0][0],
                  'ONE': result[0][1],
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5],
                  'SIX': result[0][6],
                  'SEVEN': result[0][7],
                  'EIGHT': result[0][8],
                  'NINE': result[0][9]}
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    cv2.putText(frame,'PREDICTED AS:' ,(70,70) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0))
    cv2.imshow("Frame", frame)

    if prediction[0][0] == 'ZERO':
        img=cv2.imread('G:\\Draft\\labels\\zero.png')
        #speak("শূন্য",lang="bn")
        #os.system("digit 0.wav")
    elif prediction[0][0] == 'ONE':
        img = cv2.imread('G:\\Draft\\labels\\one.png')
        #speak("এক",lang="bn")
        #os.system("digit 1.wav")
    elif prediction[0][0] == 'TWO':
        img = cv2.imread('G:\\Draft\\labels\\two.png')
        speak("দুই",lang="bn")
        os.system("digit 2.wav")
    elif prediction[0][0] == 'THREE':
        img = cv2.imread('G:\\Draft\\labels\\three.png')
        speak("তিন",lang="bn")
        os.system("digit 3.wav")
    elif prediction[0][0] == 'FOUR':
        img = cv2.imread('G:\\Draft\\labels\\four.png')
        speak("চার",lang="bn")
        #os.system("digit 4.wav")
    elif prediction[0][0] == 'FIVE':
        img = cv2.imread('G:\\Draft\\labels\\five.png')
        speak("পাঁচ",lang="bn")
        #os.system("digit 5.wav")
    elif prediction[0][0] == 'SIX':
        img = cv2.imread('G:\\Draft\\labels\\six.png')
        speak("ছয়",lang="bn")
        #os.system("digit 6.wav")
    elif prediction[0][0] == 'SEVEN':
        img = cv2.imread('G:\\Draft\\labels\\seven.png')
        speak("সাত",lang="bn")
        #os.system("digit 7.wav")
    elif prediction[0][0] == 'EIGHT':
        img = cv2.imread('G:\\Draft\\labels\\eight.png')
        speak("আট",lang="bn")
        #os.system("digit 8.wav")
    elif prediction[0][0] == 'NINE':
        img = cv2.imread('G:\\Draft\\labels\\nine.png')
        speak("নয়",lang="bn")
        #os.system("digit 9.wav")

    img_height, img_width, _ = img.shape()
    x = 100
    y = 100
    frame[y:y + img_height, x:x + img_width] = img
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()