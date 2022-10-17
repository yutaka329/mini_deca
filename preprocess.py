import numpy as np
from skimage.transform import estimate_transform, warp, resize, rescale
import cv2

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
class FaceDetectModule:
    def __init__(self):
        self.face_detector = RetinaFacePredictor(threshold=0.8, device='cpu', model=RetinaFacePredictor.get_model('mobilenet0.25'))

    def run(self, image_data):
        detected_faces = self.face_detector(image_data, rgb = True)
        select = 0
        if detected_faces.shape[0] == 0:
            print('detect no face')
            return None
        else:
            maxsize = 0
            for i in range(detected_faces.shape[0]):
                x1,y1,x2,y2 = detected_faces[i, :4]
                size = (x2-x1) * (y2-y1)
                if size > maxsize:
                    select = i
                    maxsize = size
        box = detected_faces[select, :4]
        return box

class FaceLandmarkModule:
    def __init__(self):
        self.landmark_detector = FANPredictor(device='cpu', model=FANPredictor.get_model('2DFAN2_ALT'))
    def run(self, image_data, face):
        landmarks, scores = self.landmark_detector(image_data, face, rgb = True)

        return landmarks.squeeze()

class FAN(object):
    def __init__(self):
        #import face_alignment
        #self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device = 'cpu')
        self.face_model = FaceDetectModule()
        self.faceland_model = FaceLandmarkModule()
    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        # out = self.model.get_landmarks(image)
        # if out is None:
        #     return [0], 'kpt68'
        # else:
        #     kpt = out[0].squeeze()
        #     left = np.min(kpt[:,0]); right = np.max(kpt[:,0]);
        #     top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
        #     bbox = [left,top, right, bottom]
        #     return bbox, 'kpt68'

        faces = self.face_model.run(image)
        if faces is None:
            return [0], 'kpt68'
        else:
            kpt = self.faceland_model.run(image, faces)
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0])
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'

class PreProcess:
    def __init__(self):
        self.crop_size = 224
        self.scale = 1.25
        self.iscrop = True
        self.resolution_inp = 224
        self.face_detector = FAN()

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def process(self, image):
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]
        image = image[..., ::-1] #for RGB

        h, w, _ = image.shape
        if self.iscrop:
            bbox, bbox_type = self.face_detector.run(image)
            if len(bbox) < 4:
                print('no face detected! run original image')
                left = 0;right = h - 1;top = 0;bottom = w - 1
            else:
                left = bbox[0]; right = bbox[2]
                top = bbox[1]; bottom = bbox[3]
            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size * self.scale)
            src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],[center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.
        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2, 0, 1)

        return dst_image, tform #cxhxw, _


