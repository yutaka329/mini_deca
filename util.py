import cv2
from PyQt5.QtGui import QImage
import numpy as np

def cv2qimage(cv_image):
    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2BGRA)
    return QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB32).copy()



def write_obj(obj_name,
              vertices,
              faces=None,
              ):

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices
        for i in range(vertices.shape[0]):
            f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))

        # write uv coords
        if faces is not None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))


class MeshClean():
    def __init__(self, verts, vt, fs, vt_tuple):
        '''
        :param verts:
        :param vt: 纹理坐标
        :param fs: 顶点面片
        :param vt_tuple: 纹理面片
        '''
        print("Input Verts shape={}, Vts shape={}, Faecs shape={}, Vt Faces shape={}".format(verts.shape, vt.shape, fs.shape, vt_tuple.shape))
        self.vt = vt.copy()
        self.fs = fs.copy()
        self.vt_tuple = vt_tuple.copy()
        self.build(verts, vt, fs, vt_tuple)

    def build(self, verts, vt, fs, vt_tuple):
        # find out the verts lying on the seam (if a vert owning 2 corresponding vt coordinates)
        # establish the mapping between verts and vts
        self.vert_vt_map = {}
        init = -np.ones(verts.shape[0])
        for fi in np.arange(fs.shape[0]):
            v_id = fs[fi, :]
            vt_id = vt_tuple[fi, :]
            for ids in np.arange(3):
                if init[v_id[ids]] < 0:
                    init[v_id[ids]] = vt_id[ids]
                    self.vert_vt_map[v_id[ids]] = [vt_id[ids]]
                elif len(self.vert_vt_map[v_id[ids]]) == 1 and self.vert_vt_map[v_id[ids]] != vt_id[ids]:
                    tmp = self.vert_vt_map[v_id[ids]][0]
                    self.vert_vt_map[v_id[ids]] = [tmp, vt_id[ids]]

        # # add additional verts by duplicating these verts on the seam
        # verts_add = []
        # verts_add_id = {}
        # cnt = 0
        # verts_num = verts.shape[0]
        # self.vert_vt_map_seam = {}
        # for i in np.arange(verts.shape[0]):
        #     if len(vert_vt_map[i]) == 2:
        #         self.vert_vt_map_seam[i] = vert_vt_map[i]
        #         verts_add_id[i] = verts_num + cnt
        #         verts_add.append(verts[i])
        #         cnt += 1
        # verts_new = np.vstack((verts, np.asarray(verts_add)))
        #
        # # calculate the re-ordered faces ids
        # fs_new = fs
        # for fi in np.arange(fs.shape[0]):
        #     v_id = fs[fi, :]
        #     vt_id = vt_tuple[fi, :]
        #     for ids in np.arange(3):
        #         if len(vert_vt_map[v_id[ids]]) == 2 and vt_id[ids] == vert_vt_map[v_id[ids]][1]:
        #             fs_new[fi, ids] = verts_add_id[v_id[ids]]
        #
        # # re-order vts, such that verts_new and vts_new are 1-to-1 corresponding
        # vts_new = np.zeros(vt.shape)
        # for fi in np.arange(fs.shape[0]):
        #     v_id = fs[fi, :]
        #     vt_id = vt_tuple[fi, :]
        #     for ids in np.arange(3):
        #         vts_new[v_id[ids]] = vt[vt_id[ids]]
        #
        # print("New Verts shape={}, Vts shape={}, Faecs shape={}".format(verts_new.shape,
        #         vts_new.shape, fs_new.shape))
        #
        # return verts_new, vts_new, fs_new


    def ReMap(self, verts):
        # add additional verts by duplicating these verts on the seam
        verts_add = []
        verts_add_id = {}
        cnt = 0
        verts_num = verts.shape[0]
        for i in np.arange(verts.shape[0]):
            if len(self.vert_vt_map[i]) == 2:
                verts_add_id[i] = verts_num + cnt
                verts_add.append(verts[i])
                cnt += 1
        verts_new = np.vstack((verts, np.asarray(verts_add)))

        # calculate the re-ordered faces ids
        fs_new = self.fs.copy()
        for fi in np.arange(self.fs.shape[0]):
            v_id = self.fs[fi, :]
            vt_id = self.vt_tuple[fi, :]
            for ids in np.arange(3):
                if len(self.vert_vt_map[v_id[ids]]) == 2 and vt_id[ids] == self.vert_vt_map[v_id[ids]][1]:
                    fs_new[fi, ids] = verts_add_id[v_id[ids]]

        # re-order vts, such that verts_new and vts_new are 1-to-1 corresponding
        vts_new = np.zeros(self.vt.shape)
        for fi in np.arange(self.fs.shape[0]):
            v_id = self.fs[fi, :]
            vt_id = self.vt_tuple[fi, :]
            for ids in np.arange(3):
                vts_new[v_id[ids]] = self.vt[vt_id[ids]]

        print("New Verts shape={}, Vts shape={}, Faecs shape={}".format(verts_new.shape, vts_new.shape, fs_new.shape))

        return verts_new, vts_new, fs_new


