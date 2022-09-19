import cv2
import torch
from decalib.utils.config import cfg as deca_cfg
from decalib.deca import DECA
from preprocess import PreProcess
import os


class deca_agent:
    def __init__(self):
        self.device = 'cpu'
        self.preprocess = PreProcess()
        self.build()

    def build(self):
        deca_cfg.model.use_tex = True
        deca_cfg.rasterizer_type = "pytorch3d" #cpu 不支持
        deca_cfg.model.extract_tex = False
        self.deca = DECA(config = deca_cfg, device=self.device)

    def run(self, image, savefolder,  name, saveObj=True):
        image_numpy = self.preprocess.process(image)  # cxhxw
        image_tensor = torch.tensor(image_numpy).float()

        images = image_tensor.to(self.device)[None, ...]
        with torch.no_grad():
            codedict = self.deca.encode(images)
            opdict, visdict = self.deca.decode(codedict, rendering=True, iddict=None,
                                               vis_lmk=False, return_vis=True,
                                               use_detail=True, render_orig=False)  # tensor
            if saveObj:
                self.deca.save_obj(os.path.join(savefolder, name + '.obj'), opdict)


def main():
    test_image = "Images/IMG_0392_inputs.jpg"
    savefolder = os.path.splitext(test_image)[0]
    name = os.path.splitext(os.path.split(test_image)[-1])[0]
    print(savefolder, name)
    os.makedirs(savefolder, exist_ok=True)

    deca = deca_agent()
    image = cv2.imread(test_image)

    deca.run(image, savefolder, name)

if __name__ == "__main__":
    main()