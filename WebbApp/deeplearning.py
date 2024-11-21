import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt
import torch
from PIL import Image

model = torch.hub.load(
    'yolov5',
    'custom',
    path='yolov5/runs/train/Model2/weights/best.pt',
    source="local")


def object_detection(path, filename):
    # Read image
    image = Image.open(path)  # PIL object
    image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)

    result = model(image, size=640)

    result.save()
    coords = result.xyxy[0].tolist() #return a matrix of coordinates and classes

    new_coords = []
    for obj_class in coords:
      obj_class = obj_class[:4]
      new_coords.append([int(x) for x in obj_class])
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(
        'WebbApp/static/predict/{}'.format(filename), image_bgr)
    return new_coords


def save_text(filename, text):
    name, ext = os.path.splitext(filename)
    with open('WebbApp/static/predict/{}.txt'.format(name), mode='w') as f:
        f.write(text)
    f.close()


def OCR(path, filename):
    img = np.array(load_img(path))
    cods = object_detection(path, filename)
    final_text = ""

    for i in range(len(cods)):
      xmin, ymin, xmax, ymax = cods[i].copy()
      
      roi = img[ymin:ymax, xmin:xmax]
      roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
      gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)


      blur = cv2.GaussianBlur(gray, (3,3), 0)
      thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

      # Morph open to remove noise and invert image
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
      opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
      invert = 255 - opening

      cv2.imwrite(
          'WebbApp/static/roi/{}'.format(filename), roi_bgr)

      text = pt.image_to_string(invert, lang='eng', config='--psm 6')
      final_text += text.strip() + ";" +"\n"
    print(final_text)
    save_text(filename, final_text)
    return final_text
