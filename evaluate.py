# -*- coding: utf-8 -*-
import numpy as np
import cv2
import importlib
from prettytable import PrettyTable
from os import listdir
from os.path import isfile, join
import json
from functools import lru_cache

import boe.main as run  # <- Change the module folder


def lev_dist(a, b):
    """
    This function will calculate the levenshtein distance between two input
    strings a and b

    params:
        a (String) : The first string you want to compare
        b (String) : The second string you want to compare

    returns:
        This function will return the distnace between string a and b.

    example:
        a = 'stamp'
        b = 'stomp'
        lev_dist(a,b)
        >> 1.0
    """

    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),  # insert character
            min_dist(s1 + 1, s2),  # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)


def character_level_accuracy(pred, gt):
    n_correct = 0
    if len(pred) != 0:
        for i, char in enumerate(gt):
            if i < len(pred):
                if pred[i] == char:
                    n_correct += 1
            else:
                break

    return round((n_correct / len(gt)), 4)


input_dir = "../dataset"
ground_truth_filename = "../ground_truth.json"
numImages = 400
onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
onlyfiles.sort(key=lambda f: int(f.split(".")[0]))
files = onlyfiles[0:numImages]
gt_file = open(ground_truth_filename)
gt_dict = json.load(gt_file)

importlib.reload(run)

acc_list = np.zeros(numImages)
char_level_acc_list = np.zeros(numImages)
lev_dist_list = np.zeros(numImages)
gt_list = []
pred_list = []

for i, name in enumerate(files):
    input_img = cv2.imread(input_dir + "/" + name)
    output_plate = run.detect_plate(input_img)
    img_key = name.split(".")[0]
    gt_plate = gt_dict[str(img_key)]
    gt_list.append(gt_plate)
    pred_list.append(output_plate)
    acc_list[i] = 1 if output_plate == gt_plate else 0
    char_level_acc_list[i] = character_level_accuracy(output_plate, gt_plate)
    lev_dist_list[i] = lev_dist(output_plate, gt_plate)


# Print performance scores
print("####  Plate Recognition Results  ####")
t = PrettyTable(
    [
        "Image",
        "Ground Truth",
        "Predicted",
        "Is Completely Accurate?",
        "Character Level Accuracy",
        "Levenshtein Distance",
    ]
)
acc_percent = round(np.mean(acc_list), 4) * 100
avg_char_level_acc = round(np.mean(char_level_acc_list), 4) * 100
avg_lev_dist = np.mean(lev_dist_list)


for i in range(len(files)):
    t.add_row(
        [
            files[i].split(".")[0],
            gt_list[i],
            pred_list[i],
            "Yes" if acc_list[i] == 1 else "No",
            char_level_acc_list[i],
            lev_dist_list[i],
        ]
    )

print(t)
print(f"Accuracy: {acc_percent}%")
print(f"Average Character Level Accuracy: {avg_char_level_acc}%")
print(f"Average Levenshtein Distance: {avg_lev_dist}")
# END OF EVALUATION CODE####################################################
