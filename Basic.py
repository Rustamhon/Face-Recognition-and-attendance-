import cv2
import numpy as np
import face_recognition_models
import face_recognition

#reading pic 1
imgRustamhon = face_recognition.load_image_file('C:\\work\\FaceRecognitionProject\\imagesBasic\\Rustamhon.jpg')
imgRustamhon = cv2.cvtColor(imgRustamhon, cv2.COLOR_BGR2RGB)

#reading test pic
imgTest = face_recognition.load_image_file('C:\\work\\FaceRecognitionProject\\imagesBasic\\RustamhonTest.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#putting rectangle on face
faceLoc = face_recognition.face_locations(imgRustamhon)[0]
encodeRustamhon = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgRustamhon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255), 2)

#putting rectangle on TestPic face
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255), 2)

#comparing pictures
results = face_recognition.compare_faces([encodeRustamhon], encodeTest)
faceDis = face_recognition.face_distance([encodeRustamhon], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),2)

cv2.imshow('Rustamhon Ismailov', imgRustamhon)
cv2.imshow('Test', imgTest)
cv2.waitKey(0)