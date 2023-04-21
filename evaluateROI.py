# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import importlib
from prettytable import PrettyTable
import boe.main as run  # <- Change the module folder
from os import listdir
from os.path import isfile, join
from lxml import etree

eps = 0.00000001

# Default parameters (the only code you can change)
def getROI(f):
    tree = etree.parse(f)

    d = tree.xpath("size")[0]
    w = int(d.xpath("width")[0].text)
    h = int(d.xpath("height")[0].text)

    d = tree.xpath("object/bndbox")[0]
    xMin = int(d.xpath("xmin")[0].text)
    xMax = int(d.xpath("xmax")[0].text)
    yMin = int(d.xpath("ymin")[0].text)
    yMax = int(d.xpath("ymax")[0].text)

    img = np.zeros((h, w), dtype=np.uint8)
    img = cv2.rectangle(img, (xMin, yMin), (xMax, yMax), (1), -1)

    return img


input_dir = "../dataset"
annotations_dir = "annotations"

numImages = 400
onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
onlyfiles.sort(key=lambda f: int(f.split(".")[0]))
files = onlyfiles[0:numImages]

importlib.reload(run)
error = np.zeros(numImages)
precision = np.zeros(numImages)
recall = np.zeros(numImages)
iou = np.zeros(numImages)

for i, name in enumerate(files):
    input_img = cv2.imread(input_dir + "/" + name)

    gtPart = getROI(annotations_dir + "/" + name.split(".")[0] + ".xml")
    outputPart = run.detect_ROI(input_img)

    gtPart = gtPart.astype("float32")
    outputPart = outputPart.astype("float32")

    precision[i] = sum(sum(gtPart * outputPart)) / (sum(sum(outputPart)) + eps)
    recall[i] = sum(sum(gtPart * outputPart)) / sum(sum(gtPart))
    error[i] = 1 - ((2 * precision[i] * recall[i]) / (precision[i] + recall[i] + eps))
    iou[i] = sum(sum(gtPart * outputPart)) / sum(sum(np.clip(gtPart + outputPart, 0, 1)))

# Print performance scores
print("####  IMAGE ROI RESULTS  ####")
t = PrettyTable(["Image", "Error", "Precision", "Recall", "IoU"])
avg_error = np.mean(error)
avg_precision = np.mean(precision)
avg_recall = np.mean(recall)
avg_iou = np.mean(iou)

for i in range(len(files)):
    t.add_row(
        [
            files[i].split(".")[0],
            str(round(error[i], 4)),
            str(round(precision[i], 4)),
            str(round(recall[i], 4)),
            str(round(iou[i], 4)),
        ]
    )

print(t)
print(f"Average Error: {avg_error}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average IoU: {avg_iou}")

# END OF EVALUATION CODE####################################################
