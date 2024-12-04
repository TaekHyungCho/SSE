import numpy as np
import torch
import torch.nn.functional as F
import math
import cv2
import os

'''
======================================================
This source code is based on 
1) "Scale-Equivariant Steerable Networks",  
paper url : https://arxiv.org/abs/1910.11093 ICLR 2019
github : https://github.com/ISosnovik/sesn
2) "DISCO: accurate Discrete Scale Convolutions", 
paper url : https://arxiv.org/abs/2106.02733 BMVC 2021
github : https://github.com/ISosnovik/disco

======================================================

We've added and modifed some codes according to its purpose.
'''

def hermite_poly(X, n):
    """Hermite polynomial of order n calculated at X
    Args:
        n: int >= 0
        X: np.array

    Output:
        Y: array of shape X.shape
    """
    coeff = [0] * n + [1]
    func = np.polynomial.hermite_e.hermeval(X, coeff)
    return func

def heremite_2DGaussian_basis(size, scales, effective_size,mult,max_order):

    num_funcs = effective_size**2
    basis_tensors= []
    for scale in scales:
        # Get Basis without padding
        basis= one_scale_hermite_gaussian(size,
                                            base_scale=scale,
                                            max_order=max_order,
                                            mult=mult,
                                            num_funcs=num_funcs)
        basis = basis[None, :, :, :]
        basis_tensors.append(basis)
    return torch.stack(basis_tensors, 1)

def one_scale_hermite_gaussian(size, base_scale, max_order=4, mult=2, num_funcs=None):
    
    max_order = max_order
    X = np.linspace(-(size // 2), size // 2, size)
    Y = np.linspace(-(size // 2), size // 2, size)
    order_y, order_x = np.indices([max_order + 1, max_order + 1])
    scale = base_scale
    G = np.exp(-X**2 / (2 * scale**2)) / (scale)

    basis_x = [G * hermite_poly(X / scale, n) for n in order_x.ravel()]
    basis_y = [G * hermite_poly(Y / scale, n) for n in order_y.ravel()]
    basis_x = torch.Tensor(np.stack(basis_x))
    basis_y = torch.Tensor(np.stack(basis_y))
    basis = torch.bmm(basis_x[:, :, None], basis_y[:, None, :])
    return basis

def hermite_basis_varying_sigma(size, scales, max_order=2, mult=2, num_funcs=None):

    num_funcs = num_funcs or size ** 2
    basis_x = []
    basis_y = []

    X = torch.linspace(-(size // 2), size // 2, size)
    Y = torch.linspace(-(size // 2), size // 2, size)

    for scale in scales:
        G = torch.exp(-X**2 / (2 * scale**2)) / scale

        order_y, order_x = np.indices([max_order + 1, max_order + 1])
        mask = order_y + order_x <= max_order            
        bx = [G * hermite_poly(X / scale, n) for n in order_x[mask]]
        by = [G * hermite_poly(Y / scale, n) for n in order_y[mask]]

        basis_x.extend(bx)
        basis_y.extend(by)

    basis_x = torch.stack(basis_x)[:num_funcs]
    basis_y = torch.stack(basis_y)[:num_funcs]
    return torch.bmm(basis_x[:, :, None], basis_y[:, None, :])

def normalize_basis_by_min_scale(basis):
    norm = basis.pow(2).sum([2, 3], keepdim=True).sqrt()[:, [0]]
    return basis / norm

def get_basis(dir,filename,permute=True,pure=False):
    fpath = os.path.join(dir,filename)
    basis = torch.load(fpath,map_location='cpu').contiguous()
    if 'disco' in filename:
        if not pure:
            nb,ns,k,k = basis.size()
            W = hermite_basis_varying_sigma(size=3,scales=[1.0,1.4,2.0])
            W = W.view(nb,-1)
            basis = (W@basis.view(nb,-1)).view(nb,ns,k,k)            
    if not permute:
        basis = normalize_basis_by_min_scale(basis)
        basis = basis.contiguous()
    else:
        #Intergrating diverse scales into each scale dimensions. Didn't apply normalization on basis function in our project.
        #Denote low scale-equivariance errors.
        #basis = normalize_basis_by_min_scale(basis)
        basis = basis.permute(1,0,2,3).unsqueeze(dim=0).contiguous() 
    return basis
