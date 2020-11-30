import numpy as np
import cv2
import math
from scipy.interpolate import griddata, RectBivariateSpline, interp2d
from scipy import ndimage


step = 10 # 生成单元网格的step
sigma = 2 # determines how wide the range of interfaction between points
lambdaPara = 10 # contorls the trade-off between the closeness to the data and the smoothness of the solution
alpha = 1 # controls the weight of each control point

def MRSL(image, sourcePoints, destPoints):
    """
    @description :
    @param :
        image: np.narray, input image
        sourcePoints: np.array, (n, 2), (x, y)
        destPoints: np.array, (n, 2), (x, y)
    @Returns :
        warpedImage:

    """
    
    h,w = image.shape[0], image.shape[1]

    # grid
    sp, grid = getGrid(step, h, w) 

    # normalization
    p = np.concatenate((sp, sourcePoints.T), axis=1)
    normP, normGrid = normalization(p, grid)

    # pre computation
    PV = MRLS_Precompute(normP, normGrid['nX'], normGrid['nY'], alpha, lambdaPara, sigma)

    # normalization
    q = np.concatenate((sp, destPoints.T), axis=1)
    norm_mean_Q, norm_std_Q, norm_Q = norm_ind1(q.T)

    # warp
    warpedImage = MRLS_Warp(image, norm_Q, grid['TX'], grid['TY'], normGrid, norm_mean_Q, norm_std_Q, PV)

    return warpedImage

def getGrid(step, h, w):
    """
    @description :
    @param :
        step: int
        h: int
        w: int
    @Returns :
        sp: np.array, (2,n), ([0,:]:x, [1,:]:y), boundary points
        grid: dict
    """
    h_1 = h-1
    w_1 = w-1

    hp = [i for i in range(0, h, step*10)]
    if hp[-1] != h_1: 
        hp.append(h_1)
    wp = [i for i in range(0, w, step*10)]
    if wp[-1] != w_1:
        wp.append(w_1) 

    hs1 = [1]*len(hp)
    ws1 = [1]*len(wp) # ?

    hsh = [i*h_1 for i in ws1]
    wsw = [i*w_1 for i in hs1]

    sp1 = np.array([hs1, hp], dtype=np.float32).T
    sp1 = sp1 - np.array([1, 0])
    sp2 = np.array([wp, ws1], dtype=np.float32).T
    sp2 = sp2 - np.array([0, 1])
    sp3 = np.array([wp, hsh], dtype=np.float32).T
    sp4 = np.array([wsw, hp], dtype=np.float32).T

    sp = np.concatenate((sp1, sp2[1:-1, :], sp3[1:-1, :], sp4), axis=0).T

    grid_w = np.array([i for i in range(0, w, step)] + [w_1],  dtype=np.float32)
    grid_h = np.array([i for i in range(0, h, step)] + [h_1], dtype=np.float32)
    X, Y = np.meshgrid(grid_w, grid_h)

    TX, TY = np.meshgrid(np.array([i for i in range(w)],  dtype=np.float32), np.array([i for i in range(h)],  dtype=np.float32))

    grid = {}
    grid['X'] = X
    grid['Y'] = Y
    grid['TX'] = TX
    grid['TY'] = TY

    # sp是图片边界采样所得的关键点
    return sp, grid


def normalization(p, grid, p_u=None):
    """
    @description :
    @param :
        p: np.array
        p_u: np.array
        ngrid: dict
    @Returns :
    """
    
    l = p.shape[1]
    
    if p_u!=None:
        p = np.concatenate((p, p_u), axis=0)
    
    norm_mean, norm_std, norm_p = norm_ind1(p.T)

    ngrid = {}
    ngrid['nX'] = (grid['X'] - norm_mean[0])/norm_std
    ngrid['nY'] = (grid['Y'] - norm_mean[1])/norm_std
    ngrid['nTX'] = (grid['TX'] - norm_mean[0])/norm_std
    ngrid['nTY'] = (grid['TY'] - norm_mean[1])/norm_std

    return norm_p, ngrid

def norm_ind1(x):
    """
    @description :
    @param :
        x: np.array
    @Returns :
    """

    x = x.astype(np.float64)
    n = x.shape[0]

    # NORM2 nomalizes the data to have zero means and unit covariance
    norm_mean = np.mean(x, axis=0)
    x = x - norm_mean
    norm_std = math.sqrt(np.sum(np.square(x))/n)
    x = x/norm_std

    return norm_mean, norm_std, x

def MRLS_Precompute(normP, normX, normY, alpha, lambdaPara, sigma):

    # generating kernel grid
    v = np.concatenate([normX.reshape((-1, 1), order='F'), normY.reshape((-1, 1), order='F')], axis=1)

    # construcing kernel matrix
    Gamma = con_k(normP, normP, sigma)
    Gamma_v = con_k(v, normP, sigma)

    # computing warp
    V = np.zeros(v.shape)
    C = np.zeros((v.shape[0], normP.shape[0]))
    # import ipdb; ipdb.set_trace()
    for i in range(v.shape[0]):
        
        W_1 = np.diag(np.sum(np.power(np.abs(normP-np.tile(v[i, :], (normP.shape[0], 1))), 2*alpha), axis=1))
        C[i, :] = np.dot(Gamma_v[i, :], np.linalg.inv(Gamma+lambdaPara*W_1))
        V[i, :] = v[i, :] - np.dot(C[i, :].reshape(1, -1), normP)

    PV = {}
    PV['C'] = C
    PV['V'] = V

    return PV

def con_k(x, y, sigma):
    n = x.shape[0]
    m = y.shape[0]

    k = np.tile(x[:,:, np.newaxis], (1, 1, m)) - (np.tile(y[:,:, np.newaxis], (1, 1, n)).transpose(2,1,0))
    k = np.sum(np.square(k), axis=1)
    k = -k/(sigma*sigma)
    k = np.exp(k)

    return k

def MRLS_Warp(img, normQ, TX, TY, normGrid, mean, std, PV):
    
    nX = normGrid['nX']
    nY = normGrid['nY']
    nTX = normGrid['nTX']
    nTY = normGrid['nTY']
    C = PV['C']
    V = PV['V']

    # import ipdb; ipdb.set_trace()
    # generating the grid
    v = np.concatenate([nX.reshape((-1, 1), order='F'), nY.reshape((-1, 1), order='F')], axis=1)

    # computing warp
    sfv = np.zeros(v.shape)
    for i in range(sfv.shape[0]):
        sfv[i, :] = V[i, :] + np.dot(C[i, :], normQ)

    # computing the displacements  
    dxy = sfv - v
    dxy[np.where(np.abs(dxy) < 1.0e-15)] = 0.0
    import scipy.io as io

    dxT = griddata((nX.ravel(order='F'), nY.ravel(order='F')), dxy[:, 0], (nTX, nTY), method='linear')
    dyT = griddata((nX.ravel(order='F'), nY.ravel(order='F')), dxy[:, 1], (nTX, nTY), method='linear')
    
    ifXT = (nTX+dxT)*std + mean[0]
    ifYT = (nTY+dyT)*std + mean[1]

    mapX = 2*TX-ifXT
    mapY = 2*TY-ifYT
    
    temp = np.concatenate((mapX[:,:,np.newaxis], mapY[:,:,np.newaxis]), axis=2).astype(np.float32)
    warpedImg = cv2.remap(img, temp, None, cv2.INTER_LINEAR)

    return warpedImg


