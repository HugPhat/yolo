import random
import cv2 
import numpy as np 


class RandomShear(object):

    def __init__(self, shear_factor=0.2):
        self.shear_factor = shear_factor

        if type(self.shear_factor) == tuple:
            assert len(
                self.shear_factor) == 2, "Invalid range for scaling factor"
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)

        shear_factor = random.uniform(*self.shear_factor)

    def __call__(self, img, bboxes):

        shear_factor = random.uniform(*self.shear_factor)

        w, h = img.shape[1], img.shape[0]

        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

        nW = img.shape[1] + abs(shear_factor*img.shape[0])

        bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) *
                              abs(shear_factor)).astype(int)

        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        img = cv2.resize(img, (w, h))

        scale_factor_x = nW / w

        bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]

        return img, bboxes


class Shear(object):

    def __init__(self, shear_factor=0.2):
        self.shear_factor = shear_factor

    def __call__(self, img, bboxes):
        w, h = img.shape[:2]
        shear_factor = self.shear_factor
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

        nW = img.shape[1] + int(abs(shear_factor*img.shape[0]))

        bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) *
                              abs(shear_factor)).astype(int)

        img = cv2.warpAffine(img, M, (w, img.shape[0]))

        if shear_factor < 0:
             img, bboxes = HorizontalFlip()(img, bboxes)

        return img, bboxes


class HorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))

        img = img[:, ::-1, :]
        bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])

        box_w = abs(bboxes[:, 0] - bboxes[:, 2])

        bboxes[:, 0] -= box_w
        bboxes[:, 2] += box_w

        return img, bboxes


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0])*(bbox[:, 3] - bbox[:, 1])


def clip_box(bbox, clip_box, alpha):
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    delta_area = ((ar_ - bbox_area(bbox))/(ar_ + 1e-8))

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1, :]
    return bbox


def get_enclosing_box(corners):
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def rotate_im(image, angle):

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    image = cv2.warpAffine(image, M, (nW, nH))
    return image


def get_corners(bboxes):

    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners, angle,  cx, cy, h, w):
    corners = corners.reshape(-1, 2)
    corners = np.hstack(
        (corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


class Rotate(object):
    def __init__(self, angle):
        '''
            Input: angle-> list/tuple: range of angle [num, num2]
        '''
        self.angle = angle

    def __call__(self, img, bboxes):
        '''
            * Input: 
                + img-> np.ndarray
                + bboxes-> np.ndarray
        '''
        angle = self.angle

        w, h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2

        corners = get_corners(bboxes)

        corners = np.hstack((corners, bboxes[:, 4:]))

        img = rotate_im(img, angle)

        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
        new_bbox = get_enclosing_box(corners)

        scale_factor_x = img.shape[1] / w

        scale_factor_y = img.shape[0] / h

        img = cv2.resize(img, (w, h))

        new_bbox[:, :4] /= [scale_factor_x,
                            scale_factor_y, scale_factor_x, scale_factor_y]

        bboxes = new_bbox

        bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

        return img, bboxes


def scale(img, boxes, scale):
    h, w, _ = img.shape
    img = cv2.resize(img, dsize=(
        int(w * scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    for i in range(2):
        boxes[:, i] = boxes[:, i] * scale
        boxes[:, i+2] = boxes[:, i+2] * scale
    return img, boxes

def random_rotate(img, boxes):
    a = random.uniform(-30, 30)
    return Rotate(a)(img, boxes)

def random_shear(img, boxes):
    a = random.uniform(-0.2, 0.2)
    return Shear(a)(img, boxes)

def random_scale(img, boxes):
    scale = [random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)]
    h, w, _ = img.shape
    img = cv2.resize(img, dsize=(
        int(w * scale[0]), int(h*scale[1])), interpolation=cv2.INTER_LINEAR)
    for i in range(2):
        boxes[:, i] = boxes[:, i] * scale[i]
        boxes[:, i+2] = boxes[:, i+2] * scale[i]
    return img, boxes

def random_blur(bgr):
    ksize = random.choice([2, 3, 4, 5])
    bgr = cv2.blur(bgr, (ksize, ksize))
    return bgr

def random_brightness(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    adjust = random.uniform(0.8, 1.2)
    v = v * adjust
    v = np.clip(v, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def random_hue(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    adjust = random.uniform(0.8, 1.2)
    h = h * adjust
    h = np.clip(h, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def random_saturation(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    adjust = random.uniform(0.5, 1.5)
    s = s * adjust
    s = np.clip(s, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def random_HFlip(img, bbox):
    center = np.array(img.shape[:2])[::-1] / 2
    center = np.hstack((center, center))
    img = img.copy()[:, ::-1, :]
    bbox[:, [0, 2]] = 2*center[[0, 2]] - bbox[:, [0, 2]]
    box_w = abs(bbox[:, 0] - bbox[:, 2])
    bbox[:, 0] -= box_w
    bbox[:, 2] += box_w
    return img, bbox

def random_VFlip(img, bbox):
    center = np.array(img.shape[:2])[::-1] / 2
    center = np.hstack((center, center))
    img = img.copy()[::-1, :, :]
    bbox[:, [1, 3]] = 2*center[[1, 3]] - bbox[:, [1, 3]]
    box_w = abs(bbox[:, 1] - bbox[:, 3])
    bbox[:, 1] -= box_w
    bbox[:, 3] += box_w
    return img, bbox
