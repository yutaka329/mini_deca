# def script_method(fn, _rcb=None):
#     return fn
# def script(obj, optimize=True, _frames_up=0, _rcb=None):
#     return obj
# import torch.jit
# script_method1 = torch.jit.script_method
# script1 = torch.jit.script
# torch.jit.script_method = script_method
# torch.jit.script = script
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import torch
from decalib.utils.config import cfg as deca_cfg
from decalib.deca import DECA
from preprocess import PreProcess
import numpy as np
import pyvista as pv
from util import write_obj, MeshClean
from decalib.utils.util import load_obj_numpy
from decalib.utils.util import write_obj as write_obj_SP

class deca_agent:
    def __init__(self):
        self.device = 'cuda'#'cpu'
        self.preprocess = PreProcess()
        self.build()

    def build(self):
        deca_cfg.model.use_tex = True
        deca_cfg.rasterizer_type = "pytorch3d" #cpu 不支持
        deca_cfg.model.extract_tex = True
        self.deca = DECA(config = deca_cfg, device=self.device)
        self.codedict = None

    def run(self, image, savefolder,  name, saveObj=True):
        image_numpy = self.preprocess.process(image)  # cxhxw
        image_tensor = torch.tensor(image_numpy).float()

        images = image_tensor.to(self.device)[None, ...]
        with torch.no_grad():
            self.codedict = self.deca.encode(images)
            opdict, visdict = self.deca.decode(self.codedict, rendering=True, iddict=None,
                                               vis_lmk=False, return_vis=True,
                                               use_detail=True, render_orig=False)  # tensor
            if saveObj:
                self.deca.save_obj(os.path.join(savefolder, name + '.obj'), opdict)

    def run_hand(self, id_dict):
        shape_array = self.codedict['shape'].cpu().numpy()
        for key, value in id_dict.items():
            shape_array[:, key] = value
        self.codedict['shape'] = torch.from_numpy(shape_array).to(self.codedict['shape'].device)
        verts = self.deca.decode_hand(self.codedict).detach().cpu().numpy().squeeze() #5023x3

        return verts

class DECAInterface():
    def __init__(self, config=None):
        self.config= config
        self.deca_agent = deca_agent()
        self.last_actor = None
        self.mesh = None

        verts, self.uvcoords, self.faces, self.uvfaces  = load_obj_numpy(os.path.join('data', 'head_template.obj'))

        self.mesh_clean = MeshClean(verts, self.uvcoords, self.faces, self.uvfaces)


    def __call__(self, image_file):
        print(image_file)
        self.savefolder = os.path.splitext(image_file)[0]
        self.name = os.path.splitext(os.path.split(image_file)[-1])[0]
        os.makedirs(self.savefolder, exist_ok=True)

        image = cv2.imread(image_file)
        self.deca_agent.run(image, self.savefolder, self.name)

    def getShapeParameter(self):
        return self.deca_agent.codedict['shape'].cpu().numpy()

    def plot_with_pyvista(self, plot):
        obj_file = os.path.join(self.savefolder, self.name + '.obj')
        self.tex_file = os.path.join(self.savefolder, self.name + '.png')
        print(obj_file)
        print(self.tex_file)
        if os.path.exists(obj_file) and os.path.exists(self.tex_file):
            self.mesh = pv.read(obj_file)
            self.tex_img = pv.read_texture(self.tex_file)
            self.mesh = self.mesh.clean()
            self.t_coords = self.mesh.t_coords.copy()
            if self.last_actor is not None:
                re = plot.remove_actor(self.last_actor, reset_camera=True)
                print('remove {}'.format(re))
            if self.config is not None and self.config['disTex']:
                self.last_actor = plot.add_mesh(self.mesh, texture=self.tex_img, smooth_shading=True, show_edges=False)
            else:
                self.last_actor = plot.add_mesh(self.mesh, smooth_shading=True, show_edges=False)
        else:
            print('obj & png not exists')

    def plot_with_hand(self, id_dict, plot):
        '''
        :param id_dict: id : value changed
        :return:
        '''
        self.verts_new = self.deca_agent.run_hand(id_dict)
        #print("Debug hand xyz min = {}, max= {}".format(np.min(verts, axis=0), np.max(verts, axis=0)))

        #write_obj('test.obj', verts, faces=self.faces + 1)
        #print('New verts: ', verts.shape, self.faces.shape)
        if self.mesh is not None:
            # print('Debug1, ', self.mesh.n_points, self.mesh.number_of_points)
            # print('Debug2, ', self.mesh.points.shape, type(self.mesh.points), self.mesh.faces.shape, self.mesh.n_faces)
            # print("Debug xyz min = {}, max = {}".format(np.min(self.mesh.points, axis=0),np.max(self.mesh.points, axis=0)))
            print('Debug : t coord ', self.mesh.t_coords.shape)

            print('Debug : obj t coord', self.uvcoords.shape, self.t_coords.shape)

            #self.mesh.points[:,:] = verts[:,:]
            #self.mesh.compute_normals()

            #plot.update_coordinates(verts)

            #self.mesh.points[:, 2] += 1.0
            # for i in range(10):
            #     print(i, self.mesh.points[i])

            # faces_ = np.zeros((self.faces.shape[0], 4), dtype=np.long)
            # faces_[:, 0] = 3
            # faces_[:, 1:4] = self.faces[:,:]
            # faces_ = faces_.flatten()
            # mesh_updated = pv.PolyData(verts, faces_)
            # mesh_updated.t_coords = self.t_coords #self.uvcoords

            verts_new, vts_new, fs_new = self.mesh_clean.ReMap(self.verts_new)
            cells = np.c_[np.full(len(fs_new), 3), fs_new]
            mesh_updated = pv.PolyData(verts_new, cells)
            mesh_updated.t_coords = vts_new

            mesh_updated = mesh_updated.clean()
            if self.config is not None and self.config['disTex']:
                actor = plot.add_mesh(mesh_updated, texture=self.tex_img, smooth_shading=True, show_edges=False)
            else:
                actor = plot.add_mesh(mesh_updated, smooth_shading=True, show_edges=False)
            if self.last_actor is not None:
                plot.remove_actor(self.last_actor)
            self.last_actor = actor

    def saveMesh(self, image_file):
        savefolder = os.path.splitext(image_file)[0] + "_1"
        name = os.path.splitext(os.path.split(image_file)[-1])[0]
        os.makedirs(savefolder, exist_ok=True)

        verts = self.verts_new
        faces = self.faces
        uvcoords = self.uvcoords
        uvfaces = self.uvfaces
        texture = cv2.imread(self.tex_file)

        write_obj_SP(os.path.join(savefolder, name + '.obj'),
                     verts, faces,
                     texture = texture,
                     uvcoords = uvcoords,
                     uvfaces = uvfaces)
        print("Save New Mesh Success, output dir = {}".format(savefolder))









