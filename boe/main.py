import cv2
import numpy as np
from os.path import splitext, basename
from keras.models import model_from_json

from sklearn.preprocessing import LabelEncoder

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# CHOOSE PIPELINE #########################################
# PIPELINE = 1 : WPOD-NET + MOBILENETS
# PIPELINE = 2 : WPOD-NET + PYTESSERACT
pipeline = 2

###########################################################

# START OF UTILS CODE####################################

# utils
# Honor code from https://github.com/quangnhat185/Plate_detect_and_recognize
class Label:
    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top left(x: %f, y: %f), bottom right(x: %f, y: %f)' % (
            self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl

    def cc(self): return self.__tl + self.wh() / 2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    def bl(self): return np.array([self.__tl[0], self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob


class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, axis=1)
        br = np.amax(pts, axis=1)
        Label.__init__(self, cl, tl, br, prob)


def getWH(shape):
    return np.array(shape[1::-1]).astype(float)


def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert ((wh1 >= 0).all() and (wh2 >= 0).all())

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area


def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())


def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)
    return SelectedLabels


def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T

        A[i * 2, 3:6] = -xil[2] * xi
        A[i * 2, 6:] = xil[1] * xi
        A[i * 2 + 1, :3] = xil[2] * xi
        A[i * 2 + 1, 6:] = -xil[0] * xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H


def getRectPts(tlx, tly, brx, bry):
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1, 1, 1, 1]], dtype=float)


def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop


# Reconstruction function from predict value into plate cropped from image
def reconstruct(I, Iresized, Yr, lp_threshold):

    # 4 max-pooling layers, stride = 2
    net_stride = 2 ** 4
    side = ((208 + 40) / 2) / net_stride

    # one line and two lines license plate size
    one_line = (470, 110)
    two_lines = (280, 200)

    Probs = Yr[..., 0]
    Affines = Yr[..., 2:]

    xx, yy = np.where(Probs > lp_threshold)

    # CNN input image size
    WH = getWH(Iresized.shape)

    # output feature map size
    MN = WH / net_stride

    vxx = vyy = 0.5  # alpha
    base = lambda vx, vy: np.matrix([[-vx, -vy, 1], [vx, -vy, 1], [vx, vy, 1], [-vx, vy, 1]]).T
    labels = []
    labels_frontal = []

    for i in range(len(xx)):
        x, y = xx[i], yy[i]
        affine = Affines[x, y]
        prob = Probs[x, y]

        mn = np.array([float(y) + 0.5, float(x) + 0.5])

        # affine transformation matrix
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0)
        A[1, 1] = max(A[1, 1], 0)

        # identity transformation
        B = np.zeros((2, 3))
        B[0, 0] = max(A[0, 0], 0)
        B[1, 1] = max(A[1, 1], 0)

        pts = np.array(A * base(vxx, vyy))
        pts_frontal = np.array(B * base(vxx, vyy))

        pts_prop = normal(pts, side, mn, MN)
        frontal = normal(pts_frontal, side, mn, MN)

        labels.append(DLabel(0, pts_prop, prob))
        labels_frontal.append(DLabel(0, frontal, prob))

    final_labels = nms(labels, 0.1)
    final_labels_frontal = nms(labels_frontal, 0.1)

    # print(final_labels_frontal)
    assert final_labels_frontal, "No License plate is detected"

    # LP size and type
    out_size, lp_type = (two_lines, 2) if (
            (final_labels_frontal[0].wh()[0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)

    # apparently, forcing the output to always (one_line, 1) will decrease the accuracy even on 1line plates
    # thus, i proceed with using aspect ratio normalization to test the accuracy

    TLp = []
    Cor = []
    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for _, label in enumerate(final_labels):
            t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
            ptsh = np.concatenate((label.pts * getWH(I.shape).reshape((2, 1)), np.ones((1, 4))))
            H = find_T_matrix(ptsh, t_ptsh)
            Ilp = cv2.warpPerspective(I, H, out_size, borderValue=0)
            TLp.append(Ilp)
            Cor.append(ptsh)
    return final_labels, TLp, lp_type, Cor


def detect_lp(model, I, max_dim, lp_threshold):
    min_dim_img = min(I.shape[:2])
    factor = float(max_dim) / min_dim_img
    w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()
    Iresized = cv2.resize(I, (w, h))
    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    Yr = model.predict(T, verbose=0)
    Yr = np.squeeze(Yr)
    # print(Yr.shape)
    L, TLp, lp_type, Cor = reconstruct(I, Iresized, Yr, lp_threshold)
    return L, TLp, lp_type, Cor


# END OF UTILS CODE ######################################

# START OF HELPER FUNCTION ################################

def load_model(path, json_file_type='.json', h5_file_type='.h5'):
    try:
        path = splitext(path)[0]
        with open('%s%s' % (path, json_file_type), 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s%s' % (path, h5_file_type))
        print("Model Loaded")
        return model

    except Exception as e:
        print(e)


def preprocess_image(img, resize=False):
    if isinstance(img, str):
        img = cv2.imread(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


# detect plates from images and return plate location
# if no plates detected, adjust min_dim value
def get_plate(img_path, max_dim=608, min_dim=256):
    img = preprocess_image(img_path)
    aspect_ratio = float(max(img.shape[:2])) / min(img.shape[:2])
    new_side = int(aspect_ratio * min_dim)
    final_dim = min(new_side, max_dim)
    _, plate_img, plate_type, coords = detect_lp(wpod_net, img, final_dim, lp_threshold=0.5)
    return plate_img, plate_type, coords


def sort_contours(cnts, reverse=False):
    # Find the bounding box for each contour
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]

    # Sort the contours and their bounding boxes from left to right
    sorted_contours_bounding_boxes = sorted(zip(cnts, bounding_boxes),
                                            key=lambda b: b[1][0],
                                            reverse=reverse)
    if sorted_contours_bounding_boxes:
        # Unzip the sorted contours and their bounding boxes
        sorted_cnts, _ = zip(*sorted_contours_bounding_boxes)
        return sorted_cnts
    else:
        return []


def predict_image_with_model(image, model, labels):
    # Resize image to 80 x 80
    image = cv2.resize(image, (80, 80))

    # Stack image for 3 channel input
    image = np.stack((image,) * 3, axis=-1)

    # Predict with the model
    prediction = labels.inverse_transform(
        [np.argmax(model.predict(image[np.newaxis, :], verbose=0))]
    )

    # Return the prediction
    return prediction


# END OF HELPER FUNCTION ################################

# START OF EVALUATION FUNCTION ##############################

path = 'boe/'

wpod_net_path = path + "wpod-net/wpod-net.json"
wpod_net = load_model(wpod_net_path)

model_architecture = open(f'{path}wpod-net/MobileNets_character_recognition.json', 'r')
model = model_from_json(model_architecture.read())
model_architecture.close()

model.load_weights(f'{path}wpod-net/License_character_recognition_weight.h5')

labels = np.load(f'{path}wpod-net/license_character_classes.npy')
label_encoder = LabelEncoder()
label_encoder.classes_ = labels


# WHOLE PIPELINE ######################################################

def detect_plate(input_img):
    # detection
    # get plate coordinate from input
    try:
        plate_img, plate_type, cor = get_plate(input_img)
    except AssertionError:
        return ""

    # image preprocessing
    # scales the detected license plate and converts it to 8-bit image
    plate_image = cv2.convertScaleAbs(plate_img[0], alpha=255.0)

    # converts the image to grayscale and blurs it
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # applies an inverse binary threshold
    binary = cv2.threshold(blurred, 180, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # applies a morphological dilation using a rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)

    # MOBILE-NETS MODEL PIPELINE ###############################################################
    if pipeline == 1:
        # Find contours of the binary image
        contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_plate = plate_image.copy()

        cropped_characters = []

        # Standard width and height of a character
        standard_char_width, standard_char_height = 30, 60

        # Loop through the sorted contours
        for c in sort_contours(contours):
            # Find the bounding box for each contour
            x, y, w, h = cv2.boundingRect(c)

            # Check if the contour meets the aspect ratio criteria
            aspect_ratio = h / w
            if 1 <= aspect_ratio <= 3.5:
                # Check if the contour meets the height criteria
                height_criteria = h / plate_image.shape[0] >= 0.5
                if height_criteria:

                    # Crop the character from the image and resize it
                    current_char = erode[y:y + h, x:x + w]
                    current_char = cv2.resize(current_char, dsize=(standard_char_width, standard_char_height))

                    # Threshold the character to get a binary image
                    _, current_char = cv2.threshold(current_char, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Add the character to the list of cropped characters
                    cropped_characters.append(current_char)

        # recognition
        final_string = ''

        for i, character in enumerate(cropped_characters):
            # Predict the character with the model
            contour = np.array2string(predict_image_with_model(character, model, label_encoder))

            final_string += contour.strip("'[]")

    # PYTESSERACT MODEL PIPELINE ######################################################
    elif pipeline == 2:

        erode = erode * 255
        img_final = Image.fromarray(erode)

        # plt.imshow(erode)
        # print(img_final)

        # recognition

        predicted_result = pytesseract.image_to_string(img_final, lang='eng',
                                                       config='--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHJKLMNPQRSTUVWXY0123456789')
        final_string = "".join(predicted_result.split()).replace(":", "").replace("-", "")

        if final_string == "":
            predicted_result = pytesseract.image_to_string(input_img, lang='eng',
                                                           config='--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHJKLMNPQRSTUVWXY0123456789')
            final_string = "".join(predicted_result.split()).replace(":", "").replace("-", "")

    return final_string


# END OF EVALUATION FUNCTION ##############################

# START OF DETECTION EVALUATION ##############################
def detect_ROI(input_img):
    blank_image = np.zeros(input_img.shape[:-1], np.uint8)

    # detection
    # get plate coordinate from input
    try:
        plate_img, plate_type, coordinates = get_plate(input_img)
    except AssertionError:
        return blank_image

    points = []
    x_points = coordinates[0][0]
    y_points = coordinates[0][1]
    # stores the top-left, top-right, bottom-left, and bottom-right
    # corners of the license plate
    for i in range(4):
        points.append([int(x_points[i]), int(y_points[i])])

    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))

    cv2.fillPoly(blank_image, pts=[points], color=(1))

    return blank_image
# END OF DETECTION EVALUATION ##############################
