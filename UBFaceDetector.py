
from mimetypes import encodings_map
from helper import show_image

import cv2
import numpy as np
import os
import sys

import face_recognition
from sklearn.cluster import KMeans


def detect_faces(input_path: str) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    facedetector = cv2.dnn.readNetFromCaffe(
        "dnn_facedetector/deploy.prototxt",
        "dnn_facedetector/res10_300x300_ssd_iter_140000_fp16.caffemodel")
    for file in os.listdir(input_path):
        print("Detecting Face : " , file)
        img = cv2.imread(os.path.join(input_path, file))
        (h, w) = img.shape[:2]
        img_preprocessed = cv2.dnn.blobFromImage(
            cv2.resize(
                img, (300, 300)),
                1.0,(300, 300),
                (104.0, 177.0, 123.0))
        facedetector.setInput(img_preprocessed)
        faces = facedetector.forward()
        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.8:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                # box = np.floor(box)
                label = {"iname": file,"bbox":[int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])]}
                result_list.append(label)

                # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(0, 0, 255), 2)
        # show_image(img)
    return result_list


'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    K = int(K)
    data = detect_faces(input_path=input_path)
    encodings = np.zeros((len(data), 128))
    data_2 = []
    for i, d in enumerate(data):
        iname = d["iname"]
        box = d["bbox"]
        box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        img = cv2.imread(os.path.join(input_path, iname))

        print("Encoding face : ", iname, " , bbox : ", box)
        face = face_recognition.face_encodings(img, [box])[0]
        dd = [
            {"image":iname,
            "bbox": box,
            "encoding": face}]
        encodings[i, :] = face
        data_2.extend(dd)
    encodings = encodings.astype(np.float32)

    print("K means Clustering...")
    kmeans = KMeans(n_clusters=K, random_state=0).fit(encodings)
    labels = kmeans.labels_

    unique_labels = np.unique(labels)
    print(len(unique_labels), unique_labels)
    for label in unique_labels:
        print("Extracting Class label : ", label)
        idxs = np.where(labels==label)[0]
        elements = []
        faces = []
    
        for i, idx in enumerate(idxs):
            img = cv2.imread(os.path.join(input_path, data_2[idx]["image"]))
            elements.append(str(data_2[idx]["image"]))
            box = data_2[idx]["bbox"]
            crop_face = img[box[1]:box[3], box[0]:box[2], :]
            face = cv2.resize(crop_face, (96, 96))
            faces.append(face)

        faces = np.hstack(faces)
        show_image(faces)
        cluster = {"cluster_no": int(label), "elements":elements}
        result_list.append(cluster)
    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""
