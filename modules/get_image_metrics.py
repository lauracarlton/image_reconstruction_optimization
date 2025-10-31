#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:00:53 2024

@author: lcarlton
"""
import numpy as np 
import scipy
import sys 
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import spatial_basis_funs as sbf 

from cedalion import nirs, xrutils, units
import xarray as xr
#%%

def get_ROI(image, threshold = 0.5):
    
    max_vertex_amp = image.max()
    # max_vertex = image.argmax()
    ROI = np.where(image > max_vertex_amp*threshold)[0]
    
    return ROI

def get_ROI_contig(image, head, threshold = 0.5):
    
    ROI = get_ROI(image, threshold=threshold)
    
    ROI_contig = get_contiguous_blob(image.argmax(), head, ROI)
    
    return ROI_contig
    
    
def get_image_centroid(image, ROI, head):
    if len(ROI) == 1:
        weights = image[ROI]
    else:
        weights = image[ROI].squeeze()
        
    centroid = np.average(head.brain.mesh.vertices[ROI,:], axis=0, weights=weights)
    centroid = np.reshape(centroid, [1, len(centroid)])
    
    return centroid
        
    
def get_FWHM(image, head, ROI_threshold = 0.5, version='weighted_mean'):

    # get contiuguous region containing the max vertex
    # ROI_contig = get_contiguous_blob(max_vertex, head, ROI)

    # get mean distance between centroid of contig blob and all vertices in blob
    
    if version == 'mean':
        ROI = get_ROI(image, threshold=ROI_threshold)  

        ## find the centroid ##    
        centroid = get_image_centroid(image, ROI, head)
        
        ## find average distance from all points in ROI to the centroid ##
        distances = np.linalg.norm(centroid - head.brain.mesh.vertices[ROI,:], axis = 1)
        FWHM = np.mean(distances)
    
    if version == 'max':
        
        ROI_contig = get_ROI_contig(image, head, threshold=ROI_threshold)
        
        ## get all the pairwise distances
        distances = scipy.spatial.distance.cdist(head.brain.mesh.vertices[ROI_contig,:],head.brain.mesh.vertices[ROI_contig,:])
        
        ## take the max distance
        FWHM = np.max(distances)
        
    if version == 'weighted_mean':
        
        ROI = get_ROI(image, threshold=ROI_threshold)
        centroid = get_image_centroid(image, ROI, head)
        img_sum = np.sum(image[ROI])
        
        # get the weighted average distance between all vertices in ROI and the centroid 
        distances = np.linalg.norm(centroid - head.brain.mesh.vertices[ROI,:], axis=1)
        FWHM = np.sum(distances * image[ROI])/img_sum
        FWHM = 2* np.sqrt(2*np.log(2)) * FWHM

        
    return FWHM



def get_crosstalk(image_brain, image_scalp, ROI_threshold=0.5):
    
    ROI_scalp = get_ROI(image_scalp, threshold=ROI_threshold)
    ROI_brain = get_ROI(image_brain, threshold=ROI_threshold)
    
    # get crosstalk using the max of each image
    contrast_brain = np.mean(image_brain[ROI_brain])
    contrast_scalp = np.mean(image_scalp[ROI_scalp])
    
    crosstalk_max = contrast_scalp/contrast_brain
    
    # get the crosstalk using the RMS of each image 
    contrast_brain = np.sqrt(np.mean(image_brain[ROI_brain]**2))
    contrast_scalp = np.sqrt(np.mean(image_scalp[ROI_scalp]**2))
    
    crosstalk_rms = contrast_scalp/contrast_brain
    
    return crosstalk_max, crosstalk_rms


def get_CNR_hbo(image, y, W, ROI_threshold=0.5, n_noise_instances=25, perc_noise=0.01, SB=False, G=None, DIRECT_flag=True):
    
    image_hbo = image.sel(chromo='HbO')
    ROI = get_ROI(image_hbo, threshold=ROI_threshold)
    max_vertex_idx = image_hbo[ROI].argmax()
    contrast_brain = image_hbo[ROI].mean()
    split = len(y)//2
    n_brain = len(image_hbo)
    # run many instances of noise
    noise_instances = []
    max_values = []
    
    for kk in range(n_noise_instances):
        meas_noise = np.random.normal(0, perc_noise, len(y))  #np.random.normal(0, perc_noise, len(y)) #* perc_noise #  gaussian noise 
        y_noise = y + meas_noise
    
        if DIRECT_flag:
            
            image_noise = W.values @ y_noise
            
            if SB:
                image_noise = sbf.go_from_kernel_space_to_image_space_direct(image_noise, G)
                image_hbo = image_noise[:,0]
                image_noise_hbo = image_hbo[:n_brain]
            else:
                image_noise_brain = image_noise[W.is_brain.values]
                img_split = len(image_noise_brain)//2
                image_noise_hbo = image_noise_brain[:img_split]
            
                
        else:
            W_indirect_wl0 = W.isel(wavelength=0)
            W_indirect_wl1 = W.isel(wavelength=1)
            
            y_wl0 = y_noise[:split]
            X_wl0 = W_indirect_wl0.values @ y_wl0
          
            y_wl1 = y_noise[split:]
            X_wl1 = W_indirect_wl1.values @ y_wl1
                    
            if SB:
                X_wl0 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl0, G)
                X_wl1 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl1, G)
                
                
            X_od = np.vstack([X_wl0, X_wl1]).T
            E = nirs.get_extinction_coefficients('prahl', W.wavelength)
            einv = xrutils.pinv(E) #FIXME check unit
            X_od = xr.DataArray(X_od, dims=('vertex', 'wavelength'), coords={'wavelength':W.wavelength} )
            image_noise = xr.dot(einv, X_od/units.mm, dims=["wavelength"])
            image_noise_hbo = image_noise.sel(chromo='HbO')[:n_brain].values
        
        noise_instances.append(np.mean(image_noise_hbo[ROI]))
        max_values.append(image_noise_hbo[max_vertex_idx])
    
    noise_std = np.std(np.asarray(noise_instances))
    noise_mean = np.mean(np.asarray(noise_instances))
    
    CNR = contrast_brain/noise_std
    
    return CNR, contrast_brain, noise_mean, max_values, noise_instances, image.max()


def get_CNR(image, y, W, n_brain, head, ROI_threshold=0.5, n_noise_instances=25, perc_noise=0.01, SB=False, G=None):
    
    ROI = get_ROI(image, threshold=ROI_threshold)
    max_vertex_idx = image[ROI].argmax()
    contrast_brain = image[ROI].mean()
    
    # run many instances of noise
    noise_instances = []
    max_values = []
    
    for kk in range(n_noise_instances):
        meas_noise = np.random.normal(0, perc_noise, len(y))  #np.random.normal(0, perc_noise, len(y)) #* perc_noise #  gaussian noise 
        y_noise = y + meas_noise
        
        try:
            image_noise = W.values @ y_noise.values
        except:
            image_noise = W @ y_noise.values

        noise_brain = image_noise[:n_brain]
        
        if SB:
            noise_brain = G['G_brain'].values.T @ noise_brain
        
        noise_instances.append(np.mean(noise_brain[ROI]))
        max_values.append(noise_brain[max_vertex_idx])
        
    noise_std = np.std(np.asarray(noise_instances))
    noise_mean = np.mean(np.asarray(noise_instances))
    # noise = np.mean(image_std)
    # for alpha-spatial 1e-3, and alpha_meas 1e-3 (and also 1e-5) plot the distribution of the max amplitude for all the noise instances  - compare to value of noise free max
    # mean value of distribution should be equal to the max value of noise free image - also checking to see if distribution is skewed - could have large tail which is making std very large (only only single vertex instead of integrating over an error)
    # - if this is bad - get the mean of the contiguous vertices for both contrast and noise 
    # record and return also the mean of the noise instances 
    
    CNR = contrast_brain/noise_std
    
    return CNR, contrast_brain, noise_mean, max_values, noise_instances, image.max()

def get_CNR_v2(image, image_cov):
    
     max_vertex_idx = image.argmax()
     contrast = image.max()
     
     noise = image_cov[max_vertex_idx]
     CNR = contrast/noise
     
     return CNR

def get_localization_error(origin, image, head, ROI_threshold=0.01):
    
    ROI = get_ROI(image, threshold=ROI_threshold)  
    
    ## find the centroid ##    
    centroid = get_image_centroid(image, ROI, head)
    
    
    origin = np.reshape(origin.values, [1, 3])
    
    ## get distance between centroid and the origin
    localization_error = np.sum((origin - centroid)**2)**0.5
    
    return localization_error

def get_contiguous_blob(max_vertex, head, ROI):
    # get contiuguous region containing the max vertex
    
    stack = [max_vertex]
    visited = set()
    ROI_contig = []
    
    # Step 4: Perform DFS within the ROI
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            ROI_contig.append(v)
            
            # Find all neighboring vertices in faces that contain `v`
            for face in head.brain.mesh.faces:
                if v in face:
                    for neighbor in face:
                        if neighbor in ROI and neighbor not in visited:
                            stack.append(neighbor)

    return ROI_contig


def get_percent_reconstructed(X_brain, X_scalp, y_orig, A):
    
    if np.isnan(X_brain).sum() > 0:
        X_brain = np.nan_to_num(X_brain, nan=1e-18)
        X_scalp = np.nan_to_num(X_scalp, nan=1e-18)
        
    y_brain = A[:, A.is_brain.values].values @ X_brain
    y_scalp = A[:, ~A.is_brain.values].values @ X_scalp
    
    perc_brain = y_brain.sum()/y_orig.sum()
    perc_scalp = y_scalp.sum()/y_orig.sum()
    
    return perc_brain, perc_scalp
    
    
    
def get_ROI_volume(head, ROI):

    total_volume = 0
    faces = head.brain.mesh.faces
    vertices = head.brain.mesh.vertices
    ROI_faces = faces[np.any(np.isin(faces, ROI), axis=1)]

    for tri in ROI_faces:
        v0, v1, v2 = vertices[tri]
        total_volume += np.dot(v0, np.cross(v1,v2))/6
        
    return abs(total_volume)


def triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

def get_ROI_surface_area(head, ROI, min_vertices_in_face=2):
    faces = head.brain.mesh.faces
    vertices = head.brain.mesh.vertices
    roi_vertex_set = set(ROI)
    area = 0.0
    for f in faces:
        count = sum(v in roi_vertex_set for v in f)
        if count >= min_vertices_in_face:
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            area += triangle_area(v0, v1, v2)
    return area


def get_total_weighted_surface_area(head, image):
    
    faces = head.brain.mesh.faces
    vertices = head.brain.mesh.vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute triangle areas using cross product
    cross_prod = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    # Mean weight per triangle
    mean_weights = (image[faces[:, 0]] +
                    image[faces[:, 1]] +
                    image[faces[:, 2]]) / 3.0

    # Weighted surface area
    weighted_area = np.sum(areas * mean_weights)
    total_area = np.sum(mean_weights)
    return weighted_area/total_area
        

def get_total_surface_area(head):
    faces = head.brain.mesh.faces
    vertices = head.brain.mesh.vertices
    return get_total_weighted_surface_area(vertices, faces, np.ones(len(vertices)))
    
    
    
