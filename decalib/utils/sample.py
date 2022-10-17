import cv2
import numpy as np
import torch

def float_to_uint8(data):
    data = data*255.
    data = np.maximum(np.minimum(data, 255), 0)
    return data.astype(np.uint8)

def grid_sample(image, grid_map):
    '''

    :param image:  1xCxHxW  RGB
    :param grid_map:  1,H_out,W_out,2
    :return: N,C,H_out,w_out
    '''

    im = image.detach().cpu().numpy()[0].transpose((1,2,0))[:,:,::-1]
    height, width, _ = im.shape

    print("Debugxxx", im.shape)
    cv2.imwrite('debug1.jpg', float_to_uint8(im))


    grid = grid_map.detach().cpu().numpy()[0]

    grid_mapx = (grid[:,:, 0] + 1)/2 * (width - 1)
    grid_mapy = (grid[:,:, 1] + 1)/2 * (height - 1)

    grid_image = cv2.remap(im, grid_mapx, grid_mapy, cv2.INTER_LINEAR)
    print('debugxxx, grid_image shape = ', grid_image.shape)

    grid_image = grid_image*255.
    grid_image = np.maximum(np.minimum(grid_image, 255), 0)

    out = grid_image.astype(np.uint8)

    cv2.imwrite("debug2.jpg", out)

def grid_sample_for_orimage(image, grid_map, transform):
    '''

    :param image:  H,W,C, BGR
    :param grid_map:  1,H_out,W_out,2
    :return: N,C,H_out,w_out
    '''

    im = image
    height, width, _ = im.shape

    #cv2.imwrite('debug1.jpg', float_to_uint8(im))

    grid = grid_map.detach().cpu().numpy()[0]
    grid_size = grid.shape[0]

    grid = (grid + 1) / 2 * (224 - 1) #[-1,1]范围变换到图像域224x224
    grid = grid.reshape((-1, 2))
    grid_trans = transform.inverse(grid)
    grid_trans = grid_trans.reshape((grid_size, grid_size, 2)).astype(np.float32)

    grid_mapx = grid_trans[:, :, 0]
    grid_mapy = grid_trans[:, :, 1]
    grid_image = cv2.remap(im, grid_mapx, grid_mapy, cv2.INTER_LINEAR)


    #cv2.imwrite("debug2.jpg", grid_image)

    grid_image = grid_image[:,:,::-1].transpose(2, 0, 1) / 255.

    grid_image = torch.tensor(grid_image).to('cuda')[None, ...]
    return grid_image



