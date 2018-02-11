import cv2
import numpy as np
import os
import imutils
import matplotlib.pyplot as plt

from keras.models import model_from_json

def test(model, img_path):
    img = cv2.imread(img_path)
    img.resize(64,64,3)
    print(img.shape)
    k = model.predict(x=np.array([img]))[0][0]
    print(k)
    if k>0.66:
        print("Happy in the picture")
    elif k<0.33:
        print("sad in the picture")
    else:
        print("normal")

if __name__ == "__main__":

    model = None
    with open('model.json') as json_file:
        model = model_from_json(json_file.read())
        json_file.close()
        
    model.load_weights('weights.h5')
        
    model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])

    test(model,'test/pp.jpeg')
    test(model, 'test/my_image.jpg')


    cam = cv2.VideoCapture(0)
    while True:
        ret_value, frame = cam.read()
        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)
        img = frame.copy()
        img.resize(64,64,3)
        (ht, wd) = frame.shape[:2]
        cv2.imshow("Video Feed", frame)

        state = model.predict(x=np.array([img]))
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break


        print(state, img.shape)




    cam.release()
    cv2.destroyAllWindows()
