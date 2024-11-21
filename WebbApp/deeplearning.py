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
    # image1 = load_img(path, target_size=(224, 224))
    # # Data preprocessing
    # # Convert into array and get the normalized output
    # image_arr_224 = img_to_array(image1)/255.0
    # h, w, d = image.shape
    # test_arr = image_arr_224.reshape(1, 224, 224, 3)
    # # Make predictions
    # coords = model.predict(test_arr)
    # # Denormalize the values
    # denorm = np.array([w, w, h, h])
    # coords = coords * denorm
    # coords = coords.astype(np.int32)
    # # Draw bounding on top the image
    # xmin, xmax, ymin, ymax = coords[0]
    # pt1 = (xmin, ymin)
    # pt2 = (xmax, ymax)
    # print(pt1, pt2)
    # cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    # # Convert into bgr
    

    result = model(image, size=640)

    result.save()
    coords = result.xyxy[0].tolist()

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
      # magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)

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


def apply_brightness_contrast(input_img, brightness=0, contrast=0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
