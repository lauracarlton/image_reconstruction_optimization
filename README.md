# Image Reconstruction Optimization for fNIRS

This repository provides tools to optimize surface-based image reconstruction for functional near-infrared spectroscopy (fNIRS) data. The code implements methods described in "Surface-Based Image Reconstruction Optimization for High-Density Functional Near Infrared Spectroscopy" and consists of two main analysis pipelines:

1. **Augmented Simulation Analysis**: Use data augmentation to systematically explore the image reconstruction parameter space
2. **Real Data Validation**: Apply optimized parameters to experimental ball-squeezing task data

---

## Citation

If you use this code, please cite:

```
L. Carlton et al., “Surface-Based Image Reconstruction Optimization for High-Density Functional Near Infrared Spectroscopy,” Neurophotonics, vol. (in review), 2026.
```
---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Core Modules](#core-modules)
- [Quick Start Guide](#quick-start-guide)
- [Pipeline 1: Augmented Simulation Analysis](#pipeline-1-augmented-simulation-analysis)
- [Pipeline 2: Real Data Analysis (Ball Squeezing)](#pipeline-2-real-data-analysis-ball-squeezing)
- [Batch Processing](#batch-processing)
- [Configuration Parameters](#configuration-parameters)
- [Output Files](#output-files)

---

## Requirements

### Software Dependencies
- **Python**: 3.8 or higher
- **Cedalion**: v24 (fNIRS analysis framework)
  - Installation: See [Cedalion GitHub](https://github.com/ibs-lab/cedalion)
  - This package provides the core fNIRS processing, forward modeling, and GLM functionality
- **Additional Python packages** (installed with Cedalion):
  - `xarray`, `numpy`, `pandas`, `scipy`
  - `matplotlib`, `pyvista` (visualization)
  - `trimesh` (mesh operations)
  - `pint` (unit handling)

### Optional for Batch Processing
- **SGE/qsub**: Sun Grid Engine for cluster computing (optional but recommended for large parameter sweeps)

### Data Requirements
- **BIDS-formatted fNIRS data**: Your dataset should follow BIDS structure with SNIRF files
- **Forward model**: Pre-computed sensitivity matrices (Adot) for your probe geometry 
    - for more information on generating the sensitivity matrix refer to the Cedalion documentation - see examples/head_models/
- **Head model**: ICBM152 or Colin27 template (included with Cedalion)
    - other head models can also be used - see examples/head_models/
---

## Installation

### 1. Install Cedalion (more detailed instructions availble in the Cedalion documentation)
```bash
# Clone and install Cedalion v24
git clone https://github.com/cedalion.git
conda env create -n cedalion -f environment_dev.yml
conda activate cedalion
cd cedalion
pip install -e .
```

### 2. Clone This Repository
```bash
git clone [your-repo-url]
cd image_reconstruction_optimization
```

### 3. Set Up Python Path
Add the modules directory to your Python path, or modify scripts to point to your installation:
```python
import sys
sys.path.append('/path/to/image_reconstruction_optimization/modules/')
```
This needs to be done for all scripts and modules
---

## Repository Structure

```
image_reconstruction_optimization/
├── README.md                          # This file
├── modules/                           # Core analysis modules
│   ├── image_recon_func.py           # Image reconstruction functions
│   ├── processing_func.py            # Preprocessing and GLM
│   ├── spatial_basis_func.py         # Spatial basis function generation
│   └── get_image_metrics.py          # Image quality metrics
├── augmented_simulation/              # Pipeline 1: Parameter optimization
│   ├── STEP0_seed_vertex_selection.py
│   ├── FIG5&6_STEP1_get_measurement_variance.py
│   ├── FIG5_STEP2_get_single_wavelength_image_metrics.py
│   ├── FIG6_STEP2_get_dual_wavelength_image_metrics.py
│   ├── FIG5_generate_figure.py
│   ├── FIG6_generate_figure.py
│   └── batch_codes/                   # Batch processing scripts
│       ├── augmentation/              # STEP1 measurement variance batch jobs
│       ├── single_wl_aug/             # STEP2A single-wavelength batch jobs
│       └── dual_wl_aug/               # STEP2B dual-wavelength batch jobs
└── ball_squeezing_analysis/           # Pipeline 2: Real data validation
    ├── FIG8_STEP1_hrf_estimation.py
    ├── FIG8_STEP2_image_recon_on_HRF.py
    ├── FIG8_STEP3_get_group_average.py
    ├── FIG8_ballsqueezing_images.py
    ├── FIG8_ballsqueezing_timeseries_plot.py
    └── batch_codes/                    # Batch processing scripts
        ├── batch_STEP1_preprocess/
        └── batch_STEP2_image_recon/
```

---

## Core Modules

### `modules/image_recon_func.py`
**Purpose**: Core image reconstruction functionality

**Key Functions**:
- `load_head_model(head_model='ICBM152', with_parcels=True)`: Load anatomical head model with optional parcellation
- `load_probe(probe_path, snirf_name, head_model)`: Load probe geometry and forward model
- `compute_lambda_R_indirect(Adot, lambda_R, alpha_spatial, wavelengths)`: Compute wavelength-specific prior scaling for indirect reconstruction
- `get_Adot_scaled(Adot, wavelengths)`: Create stacked sensitivity matrix for dual-wavelength direct reconstruction
- `calculate_W(A, lambda_R, alpha_meas, alpha_spatial, ...)`: Compute reconstruction matrix with regularization
- `do_image_recon(od, head, Adot, ...)`: Main image reconstruction function
- `get_image_noise_posterior(Adot, W, alpha_spatial, lambda_R, ...)`: Estimate posterior variance of reconstructed images
- `get_probe_aligned(head, geo3d)`: Align probe coordinates to head model coordinate system

**Usage Example**:
```python
import image_recon_func as irf

# Load head model and probe
head, PARCEL_DIR = irf.load_head_model('ICBM152', with_parcels=True)
Adot, meas_list, geo3d, amp = irf.load_probe(
    probe_path, 
    snirf_name='fullhead_56x144_System2.snirf',
    head_model='ICBM152'
)

# Perform image reconstruction
X, W, D, F, G, max_eig = irf.do_image_recon(
    od=od_data,
    head=head,
    Adot=Adot,
    wavelength=[690, 850],
    alpha_meas=1e4,
    alpha_spatial=1e-2,
    lambda_R=1e-6,
    DIRECT=True,  # True for dual-wavelength direct, False for indirect
    SB=True,      # Use spatial basis functions
    cfg_sbf=cfg_sbf
)
```

### `modules/processing_func.py`
**Purpose**: Data preprocessing and GLM analysis

**Key Functions**:
- `prune_channels(rec, amp_thresh, sd_thresh, snr_thresh)`: Quality control for channels
- `median_filter(rec, median_filt)`: Apply median filter to remove spike artifacts
- `GLM(runs, cfg_GLM, geo3d, pruned_chans_list, stim_list)`: General Linear Model for HRF estimation
- `get_drift_regressors(runs, cfg_GLM)`: Generate polynomial drift regressors
- `get_drift_legendre_regressors(runs, cfg_GLM)`: Generate Legendre polynomial drift regressors
- `get_short_regressors(runs, pruned_chans_list, geo3d, cfg_GLM)`: Generate short-separation channel regressors
- `concatenate_runs(runs, stim)`: Concatenate multiple runs for combined GLM analysis
- `get_spatial_smoothing_kernel(V_ras, sigma_mm)`: Generate spatial smoothing kernel for cortical surface

**Preprocessing Pipeline**:
1. Replace invalid/zero values
2. Apply median filter
3. Prune bad channels (SNR, amplitude, source-detector distance)
4. Convert to optical density
5. Motion correction (TDDR)
6. Bandpass filtering
7. Convert to concentration
8. GLM regression to remove systemic signals

### `modules/spatial_basis_funs.py`
**Purpose**: Spatial basis function generation for regularization

**Key Functions**:
- `get_sensitivity_mask(sensitivity, threshold, wavelength_idx)`: Identify sensitive vertices from forward model
- `downsample_mesh(mesh, mask, threshold)`: Create downsampled mesh for basis function centers
- `get_kernel_matrix(mesh_downsampled, mesh, sigma)`: Generate Gaussian kernel matrix
- `get_G_matrix(head, M, threshold_brain, sigma_brain, threshold_scalp, sigma_scalp)`: Build full spatial basis for brain and scalp
- `get_H(G, A)`: Transform forward model to kernel space (indirect method)
- `get_H_stacked(G, A)`: Transform stacked forward model to kernel space (direct method)
- `go_from_kernel_space_to_image_space_direct(X, G)`: Transform reconstructed kernel weights back to vertex space (direct)
- `go_from_kernel_space_to_image_space_indirect(X, G)`: Transform reconstructed kernel weights back to vertex space (indirect)

**Usage Example**:
```python
import spatial_basis_func as sbf

# Create spatial basis
M = sbf.get_sensitivity_mask(Adot, threshold=-2, wavelength_idx=1)
G = sbf.get_G_matrix(
    head,
    M,
    threshold_brain=1*units.mm,
    threshold_scalp=5*units.mm,
    sigma_brain=1*units.mm,
    sigma_scalp=5*units.mm
)

# Transform forward model
H = sbf.get_H_stacked(G, Adot_stacked)
```

### `modules/get_image_metrics.py`
**Purpose**: Quantitative image quality assessment

**Key Metrics**:
- `get_ROI(image, threshold)`: Identify region of interest above threshold
- `get_ROI_contig(image, head, threshold)`: Get largest contiguous activation region
- `get_image_centroid(image, ROI, head)`: Compute centroid of activation
- `get_FWHM(image, head, ROI_threshold, version)`: Full-width at half-maximum (spatial resolution)
- `get_localization_error(origin, image, head, ROI_threshold)`: Distance from true source to reconstructed centroid
- `get_crosstalk(image_brain, image_scalp, ROI_threshold)`: Brain-scalp signal separation
- `get_CNR(image, y, W, n_brain, head, ...)`: Contrast-to-noise ratio
- `get_CNR_v2(image, image_cov)`: CNR computed from posterior variance
- `get_ROI_volume(head, ROI)`: Volume of activation region
- `get_ROI_surface_area(head, ROI, min_vertices_in_face)`: Surface area of activation region

---

## Quick Start Guide

### For First-Time Users

**Step 1**: Install Cedalion and dependencies using installation instructions above

**Step 2**: Prepare your data in BIDS format:
```
your_dataset/
├── sub-001/
│   └── nirs/
│       ├── sub-001_task-taskname_run-01_nirs.snirf
│       └── sub-001_task-taskname_run-01_events.tsv
├── sub-002/
│   └── nirs/
│       └── ...
└── derivatives/
    └── cedalion/
        └── fw/
            └── ICBM152/
                └── Adot.nc  # Forward model
```

**Step 3**: Choose your pipeline:
- **Parameter Optimization**: Use augmented_simulation/ if you want to find optimal reconstruction parameters
- **Apply Known Parameters**: Use ball_squeezing_analysis/ if you already have parameters and want to process real data

---

## Pipeline 1: Augmented Simulation Analysis

This pipeline uses synthetic data augmentation to systematically test reconstruction parameters without requiring multiple real datasets.

### Overview
The augmented simulation approach:
1. Takes resting-state fNIRS data (can also specify any task)
2. Adds synthetic HRF activations at known cortical locations
3. Reconstructs the synthetic activations using different parameter combinations
4. Quantifies reconstruction quality (localization error, FWHM, CNR, etc.)

### Step-by-Step Workflow

#### **STEP 0: Seed Vertex Selection** (Optional)
Select cortical locations for synthetic activation placement.
If using a head model different from the Cedalion ICBM152 implementation then this step is required. 

**Script**: `augmented_simulation/STEP0_seed_vertex_selection.py`

**Configuration**:
```python
ROOT_DIR = '/path/to/your/dataset'
PROBE_PATH = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw')
```

**Run**:
```bash
python STEP0_seed_vertex_selection.py
```

**Instructions**:
1. Visualization window opens showing brain surface and probe
2. Right-click on brain vertices to select seed locations
3. Selected vertices are saved to `picked_vertex_info.txt`
4. Aim for diverse locations covering your probe sensitivity region

**Output**: List of vertex indices (e.g., `[10089, 10453, 14673, ...]`) in a txt file 
            that can be copied into the subsquent scripts

---

#### **STEP 1: Measurement Variance Estimation**
Estimate realistic noise characteristics from your data.

**Script**: `augmented_simulation/FIG5&6_STEP1_get_measurement_variance.py`

**Purpose**: Quantify measurement noise covariance by adding synthetic HRFs and running GLM.

**Key Configuration**:
```python
ROOT_DIR = '/path/to/BIDS/dataset'
TASK = 'RS'  # Resting state or task name
NOISE_MODEL = 'ols'  # 'ols' or 'ar_irls'

# Synthetic HRF parameters
VERTEX_LIST = [10089, 10453, 14673, 11323, 13685, 11702, 8337]
BLOB_SIGMA = 15 * units.mm  # Spatial extent of activation
SCALE_FACTOR = 0.02  # Peak amplitude in OD
STIM_DUR = 5  # Stimulus duration in seconds

# Preprocessing
D_RANGE = [1e-3, 0.84]  # Amplitude thresholds (V)
SNR_THRESH = 5  # SNR threshold
```

**Run**:
```bash
python FIG5&6_STEP1_get_measurement_variance.py
```

**Process**:
1. Loads resting-state data for each subject
2. Creates synthetic HRF at each seed vertex
3. Adds synthetic HRF to real data
4. Runs GLM to estimate HRF
5. Calculates measurement variance from GLM HRF estimate

**Output**: `C_meas_subj_task-RS_blob-15mm_scale-0.02_ols.pkl`
- Measurement covariance matrix (channel × wavelength × subject × vertex)

**Expected Runtime**: ~10-30 minutes per subject (depends on data size and GLM solve method used)

---

#### **STEP 2A: Single-Wavelength Image Metrics**
Test single-wavelength reconstruction across parameter space.

**Script**: `augmented_simulation/FIG5_STEP2_get_single_wavelength_image_metrics.py`

**Purpose**: Evaluate reconstruction quality when using just a single-wavelength.

**Key Configuration**:
```python
# Parameter space to test
alpha_meas_list = [10**i for i in range(-6, 6)] 
alpha_spatial_list = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Spatial basis parameters
sigma_brain_list = [0, 1, 3, 5]  mm
sigma_scalp_list = [0, 1, 5, 10, 20] mm

```

**Run**:
```bash
python FIG5_STEP2_get_single_wavelength_image_metrics.py
```

**Process for Each Parameter Combination**:
1. Loads synthetic data and measurement variance
2. Computes reconstruction matrix W using given parameter combination
3. Reconstructs images at each seed vertex for each subject
4. For each seed, get the group average image
5. Calculates metrics:
   - Localization error (distance from true source)
   - FWHM (spatial resolution)
   - CNR (contrast-to-noise ratio)
   - Crosstalk (brain vs. scalp separation)

**Output**:  `COMPILED_METRIC_RESULTS_task-RS_blob-15mm_scale-0.02_ols_single_wl.pkl`

- Dictionary with metrics for all parameter combinations
- Dimensions: (vertex × alpha_meas × alpha_spatial × sigma combinations)

**Expected Runtime**: 
- Without batch: Several hours (large parameter space)
- With batch: 1-2 hours (parallelized)

---

#### **STEP 2B: Dual-Wavelength Image Metrics**
Test dual-wavelength reconstruction across parameter space.

**Script**: `augmented_simulation/FIG6_STEP2_get_dual_wavelength_image_metrics.py`

**Similar to STEP 2A but for direct and indirect reconstruction.**

**Key Differences**:
- Performs image reconstruction using both the indirect and direct methods

**Run**:
```bash
python FIG6_STEP2_get_dual_wavelength_image_metrics.py
```

**Output**: `COMPILED_METRIC_RESULTS_task-RS_blob-15mm_scale-0.02_ols_dual_wl.pkl`

---

#### **STEP 3: Generate Figures**
Visualize results and identify optimal parameters.

**Scripts**:
- `augmented_simulation/FIG5_generate_figure.py` (single-wavelength)
- `augmented_simulation/FIG6_generate_figure.py` (dual-wavelength)

**Run**:
```bash
python FIG5_generate_figure.py
python FIG6_generate_figure.py
```

**Outputs**:
- plots showing metric values across parameter space

**Interpretation**:
- **Low localization error**: Good spatial accuracy
- **Small FWHM**: High spatial resolution
- **High CNR**: Good signal detection
- **Low crosstalk**: Minimal brain-scalp / HbO-HbR confusion

---

### Using Batch Processing for Augmented Simulations

For large parameter sweeps, use batch submission to parallelize computations.

#### STEP 1 Batch (Measurement Variance Estimation)

**Location**: `augmented_simulation/batch_codes/augmentation/`

**Files**:
- `STEP1_submit_get_measurement_variance.py`: Submission script
- `get_measurement_variance_per_subject.py`: Worker script (per-subject processing)
- `batch_shell_script_augment_data.sh`: Shell wrapper
- `compile_measurement_variance_across_subjects.py`: Compilation script

**Setup**:
1. Edit `STEP1_submit_get_measurement_variance.py`:
```python
CODE_DIR = '/path/to/batch_codes/augmentation'
ROOT_DIR = '/path/to/BIDS/dataset'
EXCLUDED = ['sub-577']  # Subjects to skip
```

2. Submit jobs:
```bash
python STEP1_submit_get_measurement_variance.py
```

3. Monitor job status:
```bash
qstat  # Check running jobs
```

4. Compile results after all jobs complete:
```bash
python compile_measurement_variance_across_subjects.py
```

**Output**: `C_meas_subj_task-RS_blob-15mm_scale-0.02_ols.pkl` containing combined measurement variance from all subjects

---

#### Single-Wavelength Batch Processing (STEP 2A)

**Location**: `augmented_simulation/batch_codes/single_wl_aug/`

**Files**:
- `STEP2_submit_single_wl_aug_batch_job.py`: Submission script
- `single_wl_metrics_batch_aug.py`: Worker script
- `batch_shell_script_single_wl_aug.sh`: Shell wrapper

**Setup**:
1. Edit `STEP2_submit_single_wl_aug_batch_job.py`:
```python
CODE_DIR = '/path/to/batch_codes/single_wl_aug'
# Parameter space to test
alpha_meas_list = [10**i for i in range(-6, 6)] 
alpha_spatial_list = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Spatial basis parameters
sigma_brain_list = [0, 1, 3, 5]  mm
sigma_scalp_list = [0, 1, 5, 10, 20] mm
```

2. Submit jobs:
```bash
python STEP2_submit_single_wl_aug_batch_job.py
```

3. Monitor job status:
```bash
qstat  # Check running jobs
```

4. Compile results after all jobs complete:
```bash
python STEP3_compile_results_single_wl.py
```

**Output**: Combined results file with all parameter combinations (same path as when using the script)

---

#### Dual-Wavelength Batch Processing (STEP 2B)

**Location**: `augmented_simulation/batch_codes/dual_wl_aug/`

**Files**:
- `STEP2_submit_dual_wl_aug_batch_job.py`: Submission script
- `dual_wl_metrics_batch_aug.py`: Worker script
- `batch_shell_script_dual_wl_aug.sh`: Shell wrapper

**Setup**:
1. Edit `STEP2_submit_dual_wl_aug_batch_job.py`:
```python
CODE_DIR = '/path/to/batch_codes/dual_wl_aug'
# Parameter space to test
alpha_meas_list = [10**i for i in range(-6, 6)] 
alpha_spatial_list = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Spatial basis parameters
sigma_brain_list = [0, 1, 3, 5]  mm
sigma_scalp_list = [0, 1, 5, 10, 20] mm
```

2. Submit jobs:
```bash
python STEP2_submit_dual_wl_aug_batch_job.py
```

3. Monitor job status:
```bash
qstat  # Check running jobs
```

4. Compile results after all jobs complete:
```bash
python STEP3_compile_results_dual_wl.py
```

**Output**: Combined results file with all parameter combinations
---

## Pipeline 2: Real Data Analysis (Ball Squeezing)

This pipeline applies optimized parameters to experimental data.

### Overview
The ball squeezing analysis:
1. Preprocesses multi-run task data
2. Estimates HRFs from real activations
3. Performs image reconstruction with optimal parameters
4. Computes group averages
5. Generates activation maps

### Step-by-Step Workflow

#### **STEP 1: HRF Estimation**
Preprocess data and estimate HRFs for each subject.

**Script**: `ball_squeezing_analysis/FIG8_STEP1_hrf_estimation.py`

**Configuration**:
```python
ROOT_DIR = '/path/to/ball_squeezing_dataset'
TASK = 'BS'  # Ball squeezing task
N_RUNS = 3  # Number of runs per subject
NOISE_MODEL = 'ols'  # or 'ar_irls'
EXCLUDED = []  # Bad subjects

# GLM parameters
cfg_GLM = {
    'do_drift': True,
    'do_drift_legendre': False
    'do_short_sep': True,
    'drift_order': 3,
    'distance_threshold': 20 * units.mm,
    't_pre': 2 * units.s,
    't_post': 10 * units.s,
}

# Channel quality thresholds
cfg_prune = {
    'snr_thresh': 5,
    'sd_thresh': [1, 40] * units.mm,
    'amp_thresh': [1e-3, 0.84] * units.V,
}
```

**Run**:
```bash
python FIG8_STEP1_hrf_estimation.py
```

**Process**:
1. **Load Data**: Reads SNIRF files and event timing
2. **Quality Control**: Identify bad channels based on SNR, amplitude
3. **Preprocessing**:
   - Motion correction (TDDR if OLS)
   - Bandpass filtering (if OLS)
   - Convert to concentration
4. **GLM**: Estimates HRF for each channel/chromophore
5. **Save Results**: HRF estimates and measurement covariance per subject

**Output Files** (per subject, saved to `<ROOT_DIR>/derivatives/cedalion/processed_data/<subject>/`):
- `<subject>_task-BS_preprocessed_results_ols.pkl.gz`
  - Contains: `all_runs`, `chs_pruned`, `all_stims`, `geo3d`
- `<subject>_task-BS_conc_o_hrf_estimates_ols.pkl.gz`
  - Contains: `hrf_per_subj`, `hrf_mse_per_subj`, `bad_indices`

**Expected Runtime**: ~1-20 minutes per subject depending on GLM solve method used

---

#### **STEP 2: Image Reconstruction**
Reconstruct cortical images from estimated HRFs.

**Script**: `ball_squeezing_analysis/FIG8_STEP2_image_recon_on_HRF.py`

**Configuration**:
```python
cfg_list = [
    {"alpha_meas": 1e4, "alpha_spatial": 1e-3, "lambda_R": 1e-6, "DIRECT": False, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e2, "alpha_spatial": 1e-3, "lambda_R": 1e-6, "DIRECT": True, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-2, "lambda_R": 1e-6, "DIRECT": False, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e2, "alpha_spatial": 1e-2, "lambda_R": 1e-6, "DIRECT": True, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
]

# Time window for averaging to get magnitude images 
MAG_TS_FLAG = 'mag' # 'mag' if want magnitude images, 'ts' if want the full timeseries in image space
T_AVG = [4, 7]  # seconds to average over if just interested in magnitude images
```

**Run**:
```bash
python FIG8_STEP2_image_recon_on_HRF.py
```

**Process**:
1. Loads per-subject HRF estimates from STEP 1
2. Converts HRF to optical density
3. Builds spatial basis functions (if enabled)
4. Computes reconstruction matrix W
5. Reconstructs image for each subject/trial_type
6. Saves full time-course or magnitude images

**Output Files** (per subject, saved to `<ROOT_DIR>/derivatives/cedalion/processed_data/image_space/<subject>/`):
- `<subject>_task-BS_images_direct_sb-{sigma}mm_alpha-{alpha}.pkl.gz`
  - Contains: Image time-courses (vertex × time × chromo × trial_type)

**Expected Runtime**: ~5-15 minutes per subject depending on how many image recon parameters looped through

---

#### **STEP 3: Group Average**
Compute group-level statistics across subjects.

**Script**: `ball_squeezing_analysis/FIG8_STEP3_get_group_average.py`

**Configuration**:
```python
# Same parameters as STEP 2
SPATIAL_SMOOTHING = True # if you want to do spatial filtering
SIGMA_SMOOTH = 80 * units.mm  # Spatial smoothing kernel
```

**Run**:
```bash
python FIG8_STEP3_get_group_average.py
```

**Process**:
1. Loads all subject images
2. Applies spatial smoothing across subjects if enabled
3. Computes group mean and standard error
4. Saves group-level images

**Output**:
- `group_images_direct_sb-{sigma}mm_alpha-{alpha}_smooth-{sigma_smooth}mm.pkl.gz` #FIXME 
  - Contains: Group mean, standard error, group t-stat

---

#### **STEP 4: Generate Visualizations**
Create publication-quality figures.

**Scripts**:
- `ball_squeezing_analysis/FIG8_ballsqueezing_images.py`: Surface activation maps
- `ball_squeezing_analysis/FIG8_ballsqueezing_timeseries_plot.py`: Time-course plots

**Run**:
```bash
python FIG8_ballsqueezing_images.py
python FIG8_ballsqueezing_timeseries_plot.py
```

**Outputs**:
- PNG figures showing:
  - Spatial activation patterns on brain surface
  - HbO/HbR time-courses
  - Group averages with error bars

---

### Using Batch Processing for Real Data

#### STEP 1 Batch (HRF Estimation)

**Location**: `ball_squeezing_analysis/batch_codes/batch_STEP1_preprocess/`

**Submit**:
```bash
python submit_do_hrf_estimation.py
```

**Worker**: `estimate_HRF_per_subj.py` (runs for one subject per job)

#### STEP 2 Batch (Image Reconstruction)

**Location**: `ball_squeezing_analysis/batch_codes/batch_STEP2_image_recon/`

**Submit**:
```bash
python submit_do_image_recon_on_HRF.py
```

**Worker**: `image_recon_on_HRF_per_subj.py` (runs for one subject per job)

---

## Configuration Parameters

### Regularization Parameters

| Parameter | Symbol | Range | Effect |
|-----------|--------|-------|--------|
| `alpha_meas` | α_meas | 1e-6 to 1e5 | Measurement regularization. Higher = smoother, lower resolution |
| `alpha_spatial` | α_spatial | 0 to 1 | Spatial regularization. Higher = smoother spatial patterns |
| `sigma_brain` | σ_brain | 0-5 mm | Brain spatial basis width. 0 = no basis, higher = smoother |
| `sigma_scalp` | σ_scalp | 0-20 mm | Scalp spatial basis width. Typically larger than brain |
| `lambda_R` | λ_R | 1e-6 | Scaling parameter placed on image prior, R |

### Preprocessing Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `snr_thresh` | 2-10 | Signal-to-noise ratio threshold for channel pruning |
| `sd_thresh` | [1, 40] mm | Valid source-detector separation range |
| `amp_thresh` | [1e-3, 0.84] V | Valid amplitude range |
| `fmin`, `fmax` | 0-0.5 Hz | Bandpass filter cutoffs |
| `drift_order` | 0-5 | Polynomial order for drift removal |

### GLM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `t_pre` | 2 s | Time before stimulus for baseline |
| `t_post` | 10-15 s | Time after stimulus for HRF |
| `t_delta` | 1 s | Temporal spacing of basis functions |
| `t_std` | 1 s | Temporal width of basis functions |
| `noise_model` | 'ols' | 'ols' (ordinary least squares) or 'ar_irls' (autoregressive) |

---

## Output Files

### Augmented Simulation Outputs

```
derivatives/cedalion/augmented_data/
├── C_meas_subj_task-RS_blob-15mm_scale-0.02_ols.pkl
│   # Measurement variance estimates
├── COMPILED_METRIC_RESULTS_task-RS_blob-15mm_scale-0.02_ols_single_wl.pkl
│   # Single-wavelength metrics across parameters
└── COMPILED_METRIC_RESULTS_task-RS_blob-15mm_scale-0.02_ols_dual_wl.pkl
    # Dual-wavelength metrics across parameters
```

### Real Data Outputs

```
derivatives/cedalion/processed_data/
├── sub-001/
│   ├── sub-001_task-BS_preprocessed_results_ols.pkl.gz
│   ├── sub-001_task-BS_conc_o_hrf_estimates_ols.pkl.gz
│   └── sub-001_task-BS_glm_residual_ols.pkl  # Optional if SAVE_RESIDUAL=True
├── sub-002/
│   └── ...
├── image_space/
│   ├── sub-001/
│   │   └── sub-001_task-BS_image_hrf_mag_as-1e-02_ls-1e-06_am-1e+04_sb-1_ss-5_indirect_Cmeas_ar_irls.pkl.gz
│   ├── sub-002/
│   │   └── ...
│   └── group_task-BS_image_hrf_mag_as-1e-02_ls-1e-06_am-1e+04_sb-1_ss-5_indirect_Cmeas_ar_irls.pkl.gz
```
```

---

## Contact

For questions or issues:
- **Author**: Laura Carlton
- **Email**: lcarlton@bu.edu
- **GitHub**: [https://github.com/lauracarlton/image_reconstruction_optimization.git]

---

## Acknowledgments
This code builds on the Cedalion fNIRS analysis framework. Special thanks to the Cedalion development team. They can be cited:  
Intelligent Biomedical Sensing (IBS) Lab, & Cedalion Developers. (2024). Cedalion Python Toolbox (Version 1.0.0) [Computer software]. https://github.com/ibs-lab/cedalion
