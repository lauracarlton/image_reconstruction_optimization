#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP0_seed_vertex_selection.py

Interactive seed vertex selection for augmented fNIRS simulations. This module
allows users to visualize the cortical surface mesh and manually select seed
vertices that will be used to generate synthetic activation patterns for
simulation studies.

Usage
-----
Edit the CONFIG section (DATA_DIR, etc.) then run::

    python STEP0_seed_vertex_selection.py

Inputs
------
- Forward model file (Adot.nc) containing the cortical mesh and sensitivity
  matrix located in <DATA_DIR>/derivatives/cedalion/fw/ICBM152/

Configurables (defaults shown)
-----------------------------
- DATA_DIR (str): 
    - Root directory containing forward model data.
- N_SEEDS (int): varies by use case
    - Number of seed vertices to select for simulation.

Outputs
-------
- Selected seed vertex indices saved to a text or pickle file for use in
  subsequent simulation steps (FIG5&6_STEP1).

Dependencies
------------
- cedalion, xarray, numpy, matplotlib, pyvista

Author: Laura Carlton
"""
#%%
from __future__ import annotations

import sys
import os
import pyvista as pv
from cedalion import plots
import cedalion.dataclasses as cdc 

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import image_recon_func as irf

#%% LOAD HEAD MODEL 
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids')
PROBE_PATH = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw')

Adot, meas_list, geo3d, amp = irf.load_probe(PROBE_PATH)
head, parcel_dir = irf.load_head_model(with_parcels=False)

probe_aligned = irf.get_probe_aligned(head, geo3d)

#%% SELECT THE VERTICES
# right click on the probe to select a vertex 

b = cdc.VTKSurface.from_trimeshsurface(head.brain)
b = pv.wrap(b.mesh)

p = pv.Plotter()
plots.plot_surface(p, head.brain)
plots.plot_labeled_points(p, probe_aligned)
p.add_points(b, color = 'm', render_points_as_spheres=True, point_size=15, pickable=True)

# Define a callback function that will be called when a vertex is selected
def callback(point, picker):
    # Get the coordinates of the selected point
    coords = point
    # print(f"Coordinates of the selected vertex: {coords}")
    
    # Find the index of the selected vertex within the mesh
    point_index = b.find_closest_point(coords)
    # print(f"Index of the selected vertex: {point_index}")
    
    with open(os.path.join(PROBE_PATH, 'picked_vertex_info.txt'), 'a') as log_file:
        log_file.write(f"Coordinates of the selected vertex: {coords}\n")
        log_file.write(f"Index of the selected vertex: {point_index}\n")


# # Enable point picking, pass the callback function
p.enable_point_picking(callback=callback, show_point=True, use_picker=True)
p.show()


#%% VIEW THE VERTICES SELECTED IN THE TXT FILE
b = cdc.VTKSurface.from_trimeshsurface(head.brain)
b = pv.wrap(b.mesh)

VERTEX_LIST = []

with open(os.path.join(PROBE_PATH, 'picked_vertex_info.txt'), "r") as f:
    for line in f:
        if "Index of the selected vertex:" in line:
            # Split the line and take the last element (the number)
            idx = int(line.split(":")[-1].strip())
            VERTEX_LIST.append(idx)

print(VERTEX_LIST)

selected_vertex_coords = b.points[VERTEX_LIST]
p = pv.Plotter()

# Add the original mesh
plots.plot_surface(p, head.brain)

# Create a point cloud for the selected vertices and add it to the plotter
selected_vertices = pv.PolyData(selected_vertex_coords)
# plots.plot_labeled_points(p, ninja_snapped_aligned)

p.add_mesh(selected_vertices, color='m', point_size=15, render_points_as_spheres=True)
p.camera_position = 'yz'
# Show the plotter window
p.show()

# %%
