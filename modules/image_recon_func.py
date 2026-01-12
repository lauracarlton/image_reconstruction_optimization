import cedalion
import cedalion.datasets as datasets
import cedalion.imagereco.forward_model as fw
import cedalion.io as io
import cedalion.nirs as nirs
import xarray as xr
from cedalion import units
import cedalion.dataclasses as cdc 
import numpy as np
import os.path
import pickle
from cedalion.imagereco.solver import pseudo_inverse_stacked
import cedalion.dataclasses.geometry as geo 
import cedalion.geometry.landmarks as cgeolm

import cedalion.xrutils as xrutils
from cedalion.io.forward_model import load_Adot

import matplotlib.pyplot as p
import pyvista as pv
from matplotlib.colors import ListedColormap

import gzip

import sys


sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import spatial_basis_func as sbf
import pdb

#%% DATA LOADING

def load_head_model(head_model='ICBM152', with_parcels=True):
    
    if head_model == 'ICBM152':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_icbm152_segmentation()
        if with_parcels:
            PARCEL_DIR = datasets.get_icbm152_parcel_file()
        else :
            PARCEL_DIR = None
            
    elif head_model == 'colin27':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_colin27_segmentation()
        if with_parcels:
            PARCEL_DIR = datasets.get_colin27_parcel_file()
        else :
            PARCEL_DIR = None
            
    masks, t_ijk2ras = io.read_segmentation_masks(SEG_DATADIR, mask_files)

    
    head = fw.TwoSurfaceHeadModel.from_surfaces(
        segmentation_dir=SEG_DATADIR,
        mask_files = mask_files,
        brain_surface_file= os.path.join(SEG_DATADIR, "mask_brain.obj"),
        scalp_surface_file= os.path.join(SEG_DATADIR, "mask_scalp.obj"),
        landmarks_ras_file=landmarks_file,
        smoothing=0.5,
        fill_holes=True,
        parcel_file=PARCEL_DIR
    ) 
    head.scalp.units = units.mm
    head.brain.units = units.mm
    
    return head, PARCEL_DIR


def load_probe(probe_path, snirf_name ='fullhead_56x144_System2.snirf', head_model='ICBM152'):
        
    # with open(os.path.join(probe_path, 'fw',  head_model, 'Adot.pkl'), 'rb') as f:
    #     Adot = pickle.load(f)

    Adot = load_Adot(os.path.join(probe_path, 'fw',  head_model, 'Adot.nc'))

    recordings = io.read_snirf(os.path.join(probe_path, 'fw',  head_model, snirf_name))
    rec = recordings[0]
    geo3d = rec.geo3d
    amp = rec['amp']
    meas_list = rec._measurement_lists['amp']

    return Adot, meas_list, geo3d, amp


#%% MATRIX CALCULATIONS
def compute_lambda_R_indirect(Adot,
    lambda_R,
    alpha_spatial,
    wavelengths,
):
    """
    Returns:
        lambda_R_indirect
    """

    # # -----------------------------
    # # DIRECT PRIOR (Hb space)
    # # -----------------------------
    ec  = nirs.get_extinction_coefficients('prahl', wavelengths)

    A_stacked = get_Adot_scaled(Adot, wavelengths)
    nV_brain = Adot.is_brain.sum().values
    nV_head = Adot.shape[1]

    R_direct = _calculate_prior_R(A_stacked, alpha_spatial=alpha_spatial)
    R_direct = R_direct * lambda_R

    R_direct_max =[ R_direct[:nV_brain].max().values, R_direct[nV_head:nV_head+nV_brain].max().values ]

    # -----------------------------
    # INDIRECT PRIOR (OD space)
    # -----------------------------
    R_indirect_wl1 = _calculate_prior_R(Adot.isel(wavelength=0), alpha_spatial = alpha_spatial)
    R_indirect_wl2 = _calculate_prior_R(Adot.isel(wavelength=1), alpha_spatial = alpha_spatial)
    R_indirect_converted = ec.values**2 @ R_direct_max

    lambda_wl1 = R_indirect_converted[0]/R_indirect_wl1[:nV_brain].max()
    lambda_wl2 = R_indirect_converted[1]/R_indirect_wl2[:nV_brain].max()

    # -----------------------------
    # RETURN LAMBDAS 
    # -----------------------------
    lambda_R_indirect = xr.DataArray([lambda_wl1, lambda_wl2], 
                                dims = ['wavelength'],
                                coords = {'wavelength': wavelengths})

    return lambda_R_indirect
    
def get_Adot_scaled(Adot, wavelengths):
    
    nchannel = Adot.shape[0]
    nvertices = Adot.shape[1]
    E = nirs.get_extinction_coefficients('prahl', Adot.wavelength)

    A = np.zeros((2 * nchannel, 2 * nvertices))
    wl1 = wavelengths[0]
    wl2 = wavelengths[1]
    A[:nchannel, :nvertices] = E.sel(chromo="HbO", wavelength=wl1).values * Adot.sel(wavelength=wl1) # noqa: E501
    A[:nchannel, nvertices:] = E.sel(chromo="HbR", wavelength=wl1).values * Adot.sel(wavelength=wl1) # noqa: E501
    A[nchannel:, :nvertices] = E.sel(chromo="HbO", wavelength=wl2).values * Adot.sel(wavelength=wl2) # noqa: E501
    A[nchannel:, nvertices:] = E.sel(chromo="HbR", wavelength=wl2).values * Adot.sel(wavelength=wl2) # noqa: E501

    A = xr.DataArray(A, dims=("measurement", "flat_vertex"))

    if "parcel" in Adot.coords:
        A = A.assign_coords({"parcel" : ("flat_vertex", np.concatenate((Adot.coords['parcel'].values, Adot.coords['parcel'].values)))})
    if "is_brain" in Adot.coords:
        A = A.assign_coords({"is_brain" : ("flat_vertex", np.concatenate((Adot.coords['is_brain'].values, Adot.coords['is_brain'].values)))})
    return A

def calculate_W(A, 
                lambda_R=1e-6, 
                alpha_meas=1e3, 
                alpha_spatial=1e-3,
                DIRECT=True, 
                C_meas_flag=False, 
                C_meas=None, 
                D=None, 
                F=None, 
                max_eig=None):
    
    
    if DIRECT:
        if C_meas_flag:
            C_meas = np.diag(C_meas)

        W_xr, D, F, max_eig = _calculate_W_direct(A, 
                                                lambda_R=lambda_R, 
                                                alpha_meas=alpha_meas, 
                                                alpha_spatial=alpha_spatial, 
                                                C_meas_flag=C_meas_flag, 
                                                C_meas=C_meas,
                                                D=D, 
                                                F=F, 
                                                max_eig=max_eig)
    else:
        W_xr, D, F, max_eig = _calculate_W_indirect(A, 
                                                    lambda_R=lambda_R, 
                                                    alpha_meas=alpha_meas, 
                                                    alpha_spatial=alpha_spatial, 
                                                    C_meas_flag=C_meas_flag, 
                                                    C_meas=C_meas, 
                                                    D=D, 
                                                    F=F, 
                                                    max_eig=max_eig)
                    
    return W_xr, D, F, max_eig

def _calculate_prior_R(A, alpha_spatial):

    B = np.sum((A ** 2), axis=0)
    b = B.max()
    
    lambda_spatial = alpha_spatial * b
    
    L = np.sqrt(B + lambda_spatial)
    Linv = 1/L
    R = Linv**2         

    return R

def _calculate_W_direct(A, 
                        alpha_spatial=1e-3, 
                        alpha_meas=1e4,
                        lambda_R=1e-6,
                        C_meas_flag=False, 
                        C_meas=None, 
                        D=None, 
                        F=None, 
                        max_eig=None):
    
    A_coords = A.coords
    A = A.pint.dequantify().values
                     
    if D is None and F is None:
        nV_hbo = A.shape[1]//2

        # # define R using only the HbO vertices 
        R = _calculate_prior_R(A, alpha_spatial)
        Linv = np.sqrt(R)
        R = R * lambda_R

        AR = A * R 
        F = AR @ A.T # ARA'

        # #% GET F without the scaling of R to define the max eigenvalue
        A_hat = A * Linv
        F_unscaled = A_hat @ A_hat.T 
        max_eig = np.max(np.linalg.eigvals(F_unscaled)) 

        D = R[:, np.newaxis] * A.T

    else:
        D = D.values
        F = F.values
        
    lambda_meas = alpha_meas * max_eig * lambda_R
    
    if C_meas_flag:  
        assert  len(C_meas.shape) == 2                 
        W = D @ np.linalg.inv(F  + lambda_meas * C_meas )
    else:
        W = D @ np.linalg.inv(F  + lambda_meas * np.eye(A.shape[0]) )
    
    W_xr = xr.DataArray(W, dims=("flat_vertex", "measurement"))
    D_xr = xr.DataArray(D, dims=("flat_vertex", "measurement"))

    if 'parcel' in A_coords:
        W_xr = W_xr.assign_coords({"parcel" : ("flat_vertex", A_coords['parcel'].values)})
        D_xr = D_xr.assign_coords({"parcel" : ("flat_vertex", A_coords['parcel'].values)})
    if 'is_brain' in A_coords:
        W_xr = W_xr.assign_coords({"is_brain": ("flat_vertex", A_coords['is_brain'].values)}) 
        D_xr = D_xr.assign_coords({"is_brain": ("flat_vertex", A_coords['is_brain'].values)})
    
    F_xr = xr.DataArray(F, dims=("measurement1", "measurement2"))

    return W_xr, D_xr, F_xr, max_eig

def _calculate_W_indirect(A, 
                        lambda_R=1e-6, 
                        alpha_meas=1e3, 
                        alpha_spatial=1e-3, 
                        C_meas_flag=False, 
                        C_meas=None, 
                        D=None, 
                        F=None, 
                        max_eig=None):
    
    lambda_R_indirect = compute_lambda_R_indirect(A, lambda_R, alpha_spatial, A.wavelength)

    W = []
    D_lst = []
    F_lst = []
    max_eig_lst = []

    for wavelength in A.wavelength:
        
        if C_meas_flag:
            C_meas_wl = C_meas.sel(wavelength=wavelength).values
            C_meas_wl = np.diag(C_meas_wl)
        else:
            C_meas_wl = None
            
        lambda_R_wl = lambda_R_indirect.sel(wavelength=wavelength).values

        if F is None and D is None:

            A_wl = A.sel(wavelength=wavelength).values

            R = _calculate_prior_R(A_wl, alpha_spatial)
            Linv = np.sqrt(R)

            R = R * lambda_R_wl

            A_tmp = A_wl * R
            F_wl = A_tmp @ A_wl.T

            A_hat = A_wl * Linv
            F_unscaled = A_hat @ A_hat.T

            D_wl = R[:, np.newaxis] * A_wl.T

            max_eig_wl = np.max(np.linalg.eigvals(F_unscaled)) 

        else:
            F_wl = F.sel(wavelength=wavelength).values
            D_wl = D.sel(wavelength=wavelength).values
            max_eig_wl = max_eig.sel(wavelength=wavelength).values

        lambda_meas = alpha_meas * lambda_R_wl * max_eig_wl

        W_wl = D_wl @ np.linalg.inv(F_wl  + lambda_meas * C_meas_wl )
        
        W.append(W_wl)
        D_lst.append(D_wl)
        F_lst.append(F_wl)
        max_eig_lst.append(max_eig_wl)

    W_xr = xr.DataArray(W, dims=( "wavelength", "vertex", "channel",),
                        coords = {'wavelength': A.wavelength})
    
    D_xr = xr.DataArray(D_lst, dims=( "wavelength", "vertex", "channel",),
                        coords = {'wavelength': A.wavelength})

    F_xr = xr.DataArray(F_lst, dims=( "wavelength", "channel1", "channel2"),
                        coords = {'wavelength': A.wavelength})

    max_eig = xr.DataArray(max_eig_lst, dims=("wavelength"),
                        coords = {'wavelength': A.wavelength})

    return W_xr, D_xr, F_xr, max_eig

#%% do image recon
def _get_image_brain_scalp_direct(y, W, SB=False, G=None):
    
    y = y.stack(measurement=['channel', 'wavelength']).sortby('wavelength')

    if len(y.shape) > 1:
        y = y.transpose('measurement', 'time')

    X = W.values @ y.values

    split = len(X)//2
    
    if SB:
        X = sbf.go_from_kernel_space_to_image_space_direct(X, G)

    else:
        if len(X.shape) == 1:
            X = X.reshape([2, split]).T
        else:
            X = X.reshape([2, split, X.shape[1]])
            X = X.transpose(1,2,0)
    
    if len(X.shape) == 2:
        X = xr.DataArray(X, 
                         dims = ('vertex', 'chromo'),
                         coords = {'chromo': ['HbO', 'HbR']}
                         )
    else:
        if 'time' in y.dims:
            t = y.time
            t_name = 'time'
        elif 'reltime' in y.dims:
            t = y.reltime
            t_name = 'reltime'
        X = xr.DataArray(X, 
                         dims = ('vertex',  t_name, 'chromo',),
                         coords = {'chromo': ['HbO', 'HbR'],
                                   t_name: t},
                         )

    return X


def _get_image_brain_scalp_indirect(y, W, SB=False, G=None):
                  
    W_indirect_wl0 = W.isel(wavelength=0)
    W_indirect_wl1 = W.isel(wavelength=1)
    
    if len(y.shape) > 1:
        y = y.transpose('channel', 'time', 'wavelength')

    y_wl0 = y.isel(wavelength=0) 
    y_wl1 = y.isel(wavelength=1)

    X_wl0 = W_indirect_wl0.values @ y_wl0.values
    X_wl1 = W_indirect_wl1.values @ y_wl1.values
             
    if SB:
        X_wl0 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl0, G)
        X_wl1 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl1, G)
    
    X_od = np.stack([X_wl0, X_wl1], axis=1)  

    if len(X_od.shape) == 2:
        X_od = xr.DataArray(X_od, 
                        dims = ('vertex', 'wavelength'),
                        coords = {'wavelength': W.wavelength}
                        )
    else:
        if 'time' in y.dims:
            t = y.time
            t_name = 'time'
        elif 'reltime' in y.dims:
            t = y.reltime
            t_name = 'reltime'
        X_od = xr.DataArray(X_od, 
                        dims = ('vertex', 'wavelength', t_name),
                        coords = {'wavelength': W.wavelength,
                                t_name: t},
                        )

    # convert to concentration 
    E = nirs.get_extinction_coefficients('prahl', W.wavelength)
    einv = xrutils.pinv(E)

    X = xr.dot(einv, X_od/units.mm, dims=["wavelength"])
    
    return X

def do_image_recon(od, 
                   head, 
                   Adot, 
                   C_meas_flag, 
                   C_meas, 
                   wavelength, 
                   DIRECT,
                   SB, 
                   cfg_sbf, 
                   lambda_R, 
                   alpha_spatial, 
                   alpha_meas, 
                   D, 
                   F, 
                   G, 
                   max_eig ):
    
    Adot_tmp = Adot.copy()
    if DIRECT:
        Adot_stacked = get_Adot_scaled(Adot, wavelength)
        
        if SB:
            if G is None:
                M = sbf.get_sensitivity_mask(Adot, cfg_sbf['mask_threshold'], 1)
                G = sbf.get_G_matrix(head, 
                                    M, 
                                    threshold_brain=cfg_sbf['threshold_brain'],
                                    threshold_scalp = cfg_sbf['threshold_scalp'],
                                    sigma_brain=cfg_sbf['sigma_brain'],
                                    sigma_scalp=cfg_sbf['sigma_scalp'])
                
            H_stacked = sbf.get_H_stacked(G, Adot_stacked)
            Adot_stacked = H_stacked.copy()
            
        W, D, F, max_eig = calculate_W(Adot_stacked, 
                                        lambda_R=lambda_R, 
                                        alpha_meas=alpha_meas, 
                                        alpha_spatial=alpha_spatial, 
                                        C_meas_flag=C_meas_flag, 
                                        C_meas=C_meas, 
                                        DIRECT=DIRECT, 
                                        D=D, 
                                        F=F, 
                                        max_eig=max_eig)
       
        X = _get_image_brain_scalp_direct(od, W, SB=SB, G=G)
    
            
    else:
        if SB:
            if G is None:
                M = sbf.get_sensitivity_mask(Adot, cfg_sbf['mask_threshold'], 1)
                G = sbf.get_G_matrix(head, 
                                    M, 
                                    threshold_brain=cfg_sbf['threshold_brain'],
                                    threshold_scalp = cfg_sbf['threshold_scalp'],
                                    sigma_brain=cfg_sbf['sigma_brain'],
                                    sigma_scalp=cfg_sbf['sigma_scalp'])
            
            H = sbf.get_H(G, Adot)            
            Adot = H.copy()

            
        W, D, F, max_eig = calculate_W(Adot, 
                                        lambda_R=lambda_R,
                                        alpha_meas=alpha_meas, 
                                        alpha_spatial=alpha_spatial, 
                                        C_meas_flag=C_meas_flag, 
                                        C_meas=C_meas, 
                                        DIRECT=DIRECT, 
                                        D=D, 
                                        F=F, 
                                        max_eig=max_eig)

        X = _get_image_brain_scalp_indirect(od, W, SB=SB, G=G)

    if 'parcel' in Adot_tmp.coords:
        X = X.assign_coords({"parcel" : ("vertex", Adot_tmp.coords['parcel'].values)})
                            
    if 'is_brain' in Adot_tmp.coords:
        X = X.assign_coords({"is_brain": ("vertex", Adot_tmp.coords['is_brain'].values)}) 
            
    return X, W, D, F, G, max_eig
    

def _get_image_noise_post_direct(A, 
                                W, 
                                lambda_R=1e-6, 
                                alpha_spatial=1e-3, 
                                SB=False, 
                                G=None):
    """
    Compute W and mse_post for a given wavelength using
    spatial regularization (via column scaling) and
    measurement regularization in data space.
    """

    # ---------------------------------------------------------
    # Spatial regularization: R = diag(1 / (B + λ_spatial))
    # ---------------------------------------------------------
    R = _calculate_prior_R(A, alpha_spatial)
    R = R * lambda_R

    # ---------------------------------------------------------
    #  Posterior variance (diagonal only)
    # mse_post(j) = R_j * (1 - (W A^T)_{jj})
    # ---------------------------------------------------------
    s = np.sum(W * A.T, axis=1)   # elementwise multiply row i with column i
    mse_post = R * (1.0 - s)

    if SB:
        mse_post = sbf.go_from_kernel_space_to_image_space_direct(mse_post, G).T
    else:
        split = len(mse_post)//2
        mse_post =  np.reshape( mse_post, (2,split) )

    X_mse_post_xr = xr.DataArray(mse_post, 
                            dims = ['chromo', 'vertex'],
                            coords = {'chromo': ['HbO', 'HbR'] })
    return X_mse_post_xr

def _get_image_noise_post_indirect(A, 
                                W, 
                                lambda_R=1e-6, 
                                alpha_spatial=1e-3, 
                                SB=False, 
                                G=None):
    """
    Compute W and mse_post for a given wavelength using
    spatial regularization (via column scaling) and
    measurement regularization in data space.
    """
    
    lambda_R_indirect = compute_lambda_R_indirect(A, lambda_R, alpha_spatial, A.wavelength)
    mse_lst = []
    # ---------------------------------------------------------
    # 2) Spatial regularization: R = diag(1 / (B + λ_spatial))
    # ---------------------------------------------------------
    for wl in A.wavelength:

        lambda_R_wl = lambda_R_indirect.sel(wavelength=wl).values

        A_wl = A.sel(wavelength=wl).values
        W_wl = W.sel(wavelength=wl).values

        R = _calculate_prior_R(A_wl, alpha_spatial)
        R = R * lambda_R_wl

        # ---------------------------------------------------------
        # Posterior variance (diagonal only)
        # mse_post(j) = R_j * (1 - (W A^T)_{jj})
        # ---------------------------------------------------------
        s = np.sum(W_wl * A_wl.T, axis=1)   # elementwise multiply row i with column i
        mse_post = R * (1.0 - s)

        if SB:
            mse_post = sbf.go_from_kernel_space_to_image_space_indirect(mse_post, G).T

        mse_lst.append(mse_post)

    X_mse_post_xr = xr.DataArray(mse_lst, 
                            dims = ['wavelength', 'vertex'],
                            coords = {'wavelength': A.wavelength })

    return X_mse_post_xr


def get_image_noise_posterior(Adot, 
                            W, 
                            alpha_spatial=1e-3, 
                            lambda_R=1e-6,
                            DIRECT=True, 
                            SB=False, 
                            G=None):

    if DIRECT:

        mse_post = _get_image_noise_post_direct(Adot.values, 
                                                W.values,
                                                lambda_R=lambda_R,
                                                alpha_spatial=alpha_spatial, 
                                                SB=SB, 
                                                G=G)


    else: 
        
        E = nirs.get_extinction_coefficients('prahl', Adot.wavelength)
        einv = xrutils.pinv(E)

        mse_post_od = _get_image_noise_post_indirect(Adot, 
                                                    W,
                                                    lambda_R=lambda_R,
                                                    alpha_spatial=alpha_spatial,
                                                    SB=SB, 
                                                    G=G)

        mse_post = einv**2 @ mse_post_od


    # if 'parcel' in Adot.coords:
    #     X_mse_post_xr = X_mse_post_xr.assign_coords({"parcel" : ("vertex", Adot.coords['parcel'].values)})
                            
    # if 'is_brain' in Adot.coords:
    #     X_mse_post_xr = X_mse_post_xr.assign_coords({"is_brain": ("vertex", Adot.coords['is_brain'].values)}) 

    return mse_post

#%%  probe geometry

def gen_xform_from_pts(p1, p2):
    """
    given two sets of points, p1 and p2 in n dimensions,
    find the n-dims affine transformation matrix t, from p1 to p2.

    Source: https://github.com/bunpc/atlasviewer/blob/71fc98ec8ca54783378310304113e825bbcd476a/utils/gen_xform_from_pts.m#l4
    
    parameters:
    p1 : ndarray
        an array of shape (p, n) representing the first set of points.
    p2 : ndarray
        an array of shape (p, n) representing the second set of points.

    returns:
    t : ndarray
        the (n+1, n+1) affine transformation matrix.
    """
    p1, p2 = np.array(p1), np.array(p2)
    p = p1.shape[0]
    q = p2.shape[0]
    m = p1.shape[1]
    n = p2.shape[1]
    
    if p != q:
        raise ValueError('number of points for p1 and p2 must be the same')
    
    if m != n:
        raise ValueError('number of dimensions for p1 and p2 must be the same')
    
    if p < n:
        raise ValueError(f'cannot solve transformation with fewer anchor points ({p}) than dimensions ({n}).')
    
    t = np.eye(n + 1)
    a = np.hstack((p1, np.ones((p, 1))))
    
    for ii in range(n):
        x = np.linalg.pinv(a) @ p2[:, ii]
        t[ii, :] = x
        
    return t


def get_probe_aligned(head, geo3d):

    SEG_DATADIR, mask_files, landmarks_file = datasets.get_icbm152_segmentation()
    masks, t_ijk2ras = io.read_segmentation_masks(SEG_DATADIR, mask_files)

    probe_optodes = geo3d.loc[(geo3d.type==geo.PointType.SOURCE) | (geo3d.type==geo.PointType.DETECTOR)] 
    probe_landmarks = geo3d.loc[geo3d.type==geo.PointType.LANDMARK] 

    # Align fiducials to head coordinate system
    fiducials_ras = io.read_mrk_json(os.path.join(SEG_DATADIR, landmarks_file), crs="aligned")
    t_ijk2ras_inv = np.linalg.pinv(t_ijk2ras)

    t_ijk2ras_inv = t_ijk2ras_inv.pint.quantify('mm')
    t_ijk2ras_inv = t_ijk2ras_inv.rename({'ijk':'tmp', 'aligned':'ijk'})
    t_ijk2ras_inv = t_ijk2ras_inv.rename({'tmp':'aligned'})

    fiducials_ijk = fiducials_ras.points.apply_transform(t_ijk2ras_inv).pint.dequantify().pint.quantify('mm')
    # Compute landmarks by EEG's 1010 system rules
    lmbuilder = cgeolm.LandmarksBuilder1010(head.scalp, fiducials_ijk)
    all_landmarks = lmbuilder.build()

    # Individial landmarks
    model_ref_pos = np.array(all_landmarks)  
    model_ref_labels = [lab.item() for lab in all_landmarks.label] 

    # Load ninja cap data
    probe_landmark_pos = list(np.array(probe_landmarks.values))
    probe_landmark_labels = list(np.array(probe_landmarks.label))

    # Construct transform from intersection
    intersection = list(set(probe_landmark_labels) & set(model_ref_labels)) 
    model_ref_pos = [model_ref_pos[model_ref_labels.index(intsct)] for intsct in intersection]
    probe_ref_pos = [probe_landmark_pos[probe_landmark_labels.index(intsct)] for intsct in intersection]

    T = gen_xform_from_pts(probe_ref_pos, model_ref_pos) # get affine  

    probe_aligned = probe_optodes.points.apply_transform(T)
    probe_aligned = probe_aligned.points.set_crs('ijk')
    probe_aligned = probe_aligned.pint.dequantify().pint.quantify('mm')

    # Snap to surface
    probe_snapped_aligned = head.brain.snap(probe_aligned)

    return probe_snapped_aligned


#%% OLD

    # def get_image_noise(C_meas, X, W, SB=False, DIRECT=True, G=None):

    # TIME = False
    # if 'time' in C_meas.dims:
    #     t_dim = 'time'
    #     TIME = True
    # elif 'reltime' in C_meas.dims:
    #     t_dim = 'reltime'
    #     TIME = True

    # if DIRECT:
    #     if TIME:
    #         C_tmp_lst = []
    #         for t in C_meas[t_dim]:
    #                 C_tmp = C_meas.sel({t_dim:t})
    #                 cov_img_tmp = W * np.sqrt(C_tmp.values) # get diag of image covariance
    #                 cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
    #                 C_tmp_lst.append(cov_img_diag)

    #         cov_img_diag = np.vstack(C_tmp_lst)
    #     else:
    #         cov_img_tmp = W *np.sqrt(C_meas.values) # W is pseudo inverse  --- diagonal (faster than W C W.T)
    #         cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
        
    #     if SB:
    #         cov_img_diag = sbf.go_from_kernel_space_to_image_space_direct(cov_img_diag, G)
    #     else:
    #         if TIME:
    #             split = cov_img_diag.shape[1]//2
    #             HbO = cov_img_diag[:, :split]   # shape: (time, vertices)
    #             HbR = cov_img_diag[:, split:]   # shape: (time, vertices)

    #             # Stack into vertex x 2 x time
    #             cov_img_diag = np.stack([HbO.T, HbR.T], axis=1)  # (vertex, 2, time)
    #             cov_img_diag = cov_img_diag.transpose(0,2,1)
    #         else:
    #             split = len(cov_img_diag)//2
    #             cov_img_diag =  np.reshape( cov_img_diag, (2,split) ).T 
        

    # else:
    #     cov_img_lst = []
    #     E = nirs.get_extinction_coefficients('prahl', W.wavelength)
    #     einv = xrutils.pinv(E)

    #     for wavelength in W.wavelength:
    #         W_wl = W.sel(wavelength=wavelength)
    #         C_wl = C_meas.sel(wavelength=wavelength)

    #         if TIME:
    #             C_tmp_lst = []
    #             for t in C_wl[t_dim]:
    #                 C_tmp = C_wl.sel({t_dim:t})
    #                 cov_img_tmp = W_wl * np.sqrt(C_tmp.values) # get diag of image covariance
    #                 cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
    #                 C_tmp_lst.append(cov_img_diag)

    #             cov_img_diag = np.vstack(C_tmp_lst)

    #         else:
    #             cov_img_tmp = W_wl * np.sqrt(C_wl.values) # get diag of image covariance
    #             cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
            
    #         if SB:
    #             cov_img_diag = sbf.go_from_kernel_space_to_image_space_indirect(cov_img_diag, G)
            
    #         cov_img_lst.append(cov_img_diag)
            
    #     if TIME:
    #         cov_img_diag =  np.stack(cov_img_lst, axis=2) 
    #         cov_img_diag = np.transpose(cov_img_diag, [2,1,0])
    #         cov_img_diag = np.einsum('ij,jab->iab', einv.values**2, cov_img_diag)

    #     else:
    #         cov_img_diag =  np.vstack(cov_img_lst) 
    #         cov_img_diag = einv.values**2 @ cov_img_diag

    # noise = X.copy()
    # noise.values = cov_img_diag

    # return noise