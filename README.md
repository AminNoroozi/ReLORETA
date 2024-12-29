# ReLORETA
The brain signals are usually distorted on their way from the inside of the brain to the surface of the brain for a variety of reasons. As a result, the EEG method, which collects the brain signals on the surface of the brain (scalp), might result in misleading and distorted signals which are different from the original brain signals inside the brain. 

To overcome this problem, the ReLORETA algorithm was introduced by Noroozi et al, 2022 to accurately calculate the brain signals inside the brain, which are called source signals. If the brain signal is produced by an abnormal activity like a seizure, the ReLORETA algorithm can localize the exact source of the seizure as well, which is referred to as brain source localization. 

According to what was discussed above, the ReLORETA algorithm can be used for two application: 

__Application 1: Classification problems__: If you have EEG signals from some subjects and want to use their signals for classification, for example, classification of mental disorders, emotion, movement type, etc. In this case, you can calculate the source signals using ReLORETA and use them for classification instead of the EEG signals. __This can boost your classification accuracy between 1 to 15%__

__Application 2: Source localization__: If you want to accurately localize the source of abnormal activities such as the seizure inside the brain. 


## How to use the reloreta module

You can install the package using the following command in Python: 
```python
pip install reloreta
```
or using the following command in IPython
```python
!pip install reloreta
```
After the installation is completed, you need to import the core module from the reloreta package as follows: 
```python
from reloreta import core
```
### Parameters
To apply the ReLORETA algorithm, you need to create an object by calling the core module and then the ReLORETA class which uses the following parameters:
```python
core.ReLORETA(lambda2 = 0.05, dimension=3,n_source=82, epsilon=1e-29, max_iter=100,lambda1=1,lr=1e8)
```
where 

__lambda2__: eLORETA regularisation parameter (default 0.05)

__dimension__: The Leadfield matrix dimension (default 3. See the example below)

__lambda1__: ReLORETA regularisation parameter

__n_source__: number of voxels (dipoles or source points) set by the user. (default 82)

__epsilon__:  the threshold to stop ReLORETA. Unless there is a good justification, it is usually set to a high value to let max_iter stop the algorithm (default 1e-29). 

__max_iter__: The maximum number of iterations before stopping the algorithm (default 100)

__lr__: ReLORETA learning rate (default 1e8. However, it needs adjustment by the user as it can significantly vary depending on the EEG data. See the example below)

### attributes
The ReLORETA object uses the following attributes: 

__E__: The ReLORETA objective function values for all iterations

__y_rel__:The final source signals calculated by ReLORETA

__K_rel__:The final leadfield matrix calculated by ReLORETA

__y_rel_all__:The source signals calculated by ReLORETA for all iterations

__X_rel__: The final reconstructed EEG signals by ReLORETA

__X_rel_all__:The reconstructed EEG signals for all iterations

__K_all__:The leadfield matrix calculated by ReLORETA for all iterations

__pow_all__: The source power signal for all iterations. The power signal shows the power of source signals in all source points (voxels). 

### Example:
Assume you have gathered EEG data from subjects using 61 electrodes, namely, 'Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 'Cz',
'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5', 'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5','P7', 'PO3', 'PO7', 'Oz', 'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8',
'FC2', 'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8','TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2', and want to calculate (reconstruct) the source signals inside the brain using ReLORETA. 

__Step 1__: Calculate a leadfield matrix using a simulated head model. 

This lead field matrix will be used to start the ReLORETA algorithm. The following code can be used to calculate the leadfield matrix. You can customize the source points and the electrodes in this code using source_points and selected_channels variables: 

```python
# First install the mne module 
!pip install mne
```
```python

# Fetching necessary files to calculate the leadfield matrix
import mne
mne.datasets.fetch_fsaverage(subjects_dir=str(mne.datasets.sample.data_path()) + '/subjects')
```
```python

# Calculating the leadfield matrix using the electrode positions and source points coordinates
import mne
import numpy as np
from mne import make_bem_model, make_bem_solution, make_forward_solution
import os
from mne.transforms import apply_trans
# Load the sample dataset for head model and transformation
data_path = mne.datasets.sample.data_path()
data_path=str(data_path)
subjects_dir = data_path + '/subjects'
subject = 'sample'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

###### Define the custom source points (convert from mm to meters)
source_points = np.array([
    [41, -27, 47], [-40, -27, 47], [38, -18, 45], [-36, -19, 48], [15, -33, 48],
    [-14, -33, 48], [28, -1, 51], [-28, -2, 52], [23, -60, 61], [-18, -61, 55],
    [22, 26, 45], [-23, 24, 44], [35, 39, 31], [-39, 34, 37], [23, 55, 7],
    [-23, 55, 4], [12, 37, -19], [-11, 38, -19], [44, 4, 0], [-42, 4, -1],
    [11, -78, 9], [-11, -81, 7], [29, -92, 2], [-19, -92, 2], [44, -75, 5],
    [-45, -75, 11], [48, -17, -31], [-47, -14, -34], [60, -27, -9], [-59, -25, -13],
    [54, -19, 1], [-57, -20, 1], [9, -45, 24], [-10, -45, 24], [5, 5, 31],
    [-5, 1, 32], [7, 17, -14], [-5, 17, -13], [12, -45, 8], [-12, -43, 8],
    [8, -48, 39], [-8, -49, 38], [6, 33, 16], [-5, 39, 20], [31, 3, -15],
    [-28, 3, -17], [26, -19, -25], [-26, -20, -22], [47, -51, -14], [-47, -52, -12],
    [40, 11, -30], [-43, 13, -30], [46, -59, 31], [-46, -60, 33], [51, -33, 34],
    [-53, -32, 33], [50, -21, 7], [-52, -19, 7], [49, 12, 17], [-48, 13, 17],
    [46, 26, 7], [-47, 27, 6], [43, 38, 12], [-46, 38, 8], [38, 30, -12],
    [-40, 31, -13], [14, 13, 11], [-11, 13, 10], [25, 3, -1], [-26, 3, -1],
    [10, -19, 6], [-9, -17, 6], [19, 0, -2], [-20, 0, -2], [10, 10, -12],
    [-11, 9, -11], [21, -1, -22], [-24, 0, -21], [28, -22, -14], [-29, -19, -15],
    [3, -1, -11], [-4, -2, -11]
]) / 1000  # Convert from mm to meters

##### Select your channels here
selected_channels = ['Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 'Cz',
                     'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5', 'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5',
                     'P7', 'PO3', 'PO7', 'Oz', 'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8',
                     'FC2', 'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
                     'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2']

subjects_dir = str(mne.datasets.sample.data_path()) + '/subjects'
os.environ['SUBJECTS_DIR'] = subjects_dir

trans = mne.read_talxfm(subject='fsaverage', subjects_dir=subjects_dir)

# Step 2: Apply the transformation to convert MNI coordinates to MRI surface RAS coordinates
source_points = apply_trans(trans['trans'], source_points) # convert source points from MNI to MRI surface RAS coordinates which is used by MNE


# Step 2: Create a source space for the fsaverage subject (which is in MNI space)
# Set up a source space with the fsaverage subject (a standard template)



# source_space = mne.setup_source_space('fsaverage', spacing='oct6', subjects_dir=subjects_dir, add_dist=False)


#################################### Use this to spread source over volume

source_space = mne.setup_volume_source_space(subject='fsaverage', pos=15, subjects_dir=subjects_dir)

####################################

# Create an empty volume source space, and use custom source points instead
# source_space = mne.setup_volume_source_space(subject=subject, pos=15, subjects_dir=subjects_dir)

# Replace the source points in the source space with the provided custom source points
source_space[0]['rr'] = source_points
source_space[0]['nn'] = np.zeros_like(source_points)  # Dummy normals, can set as zeros
source_space[0]['nuse'] = len(source_points)
source_space[0]['inuse'] = np.ones(len(source_points), dtype=int)

# ########################## This part should only be used with setup_source_space not  setup_volume_source_space
# source_space[1]['rr'] = source_points
# source_space[1]['nn'] = np.zeros_like(source_points)  # Dummy normals, can set as zeros
# source_space[1]['nuse'] = len(source_points)
# source_space[1]['inuse'] = np.ones(len(source_points), dtype=int)
# #####################################



# Load the precomputed BEM model
# bem_model = make_bem_model(subject=subject, ico=2, subjects_dir=subjects_dir)
# bem = make_bem_solution(bem_model)
model = mne.make_bem_model(subject='fsaverage', ico=2, subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
# Select 10 specific electrodes from the 10-20 system (you can choose any 10)
# For example: ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2']

# Use the montage (10-20 system) and create info with the selected channels
montage = mne.channels.make_standard_montage('standard_1005')
raw_info = mne.create_info(ch_names=selected_channels, sfreq=1000, ch_types='eeg')
raw_info.set_montage(montage)

# Create the forward solution using 10 EEG electrodes and the custom source points with 3D orientations
fwd = make_forward_solution(info=raw_info, trans=trans_fname, src=source_space, bem=bem, eeg=True, meg=False)

# Extract the lead field matrix (gain matrix)
lead_field = fwd['sol']['data']
print("Lead field matrix shape (3D):", lead_field.shape)

# For the first source point, extract the x, y, z components
first_source_contributions = lead_field[:, :3]
print("First source point contributions (shape):", first_source_contributions.shape)

# Optionally save the lead field matrix for further analysis
np.savetxt("lead_field_3d.csv", lead_field, delimiter=",")
```

As the leadfield matrix is simulated, the reconstructed source signals are not accurate initially. ReLORETA will iteratively update the leadfield matrix to achieve an accurate estimation of the source signals. 

__Step 2__: Import EEG data (and the real source location if available and you want to calculate the localization error). 
You can download three sample EEG data and corresponding source locations from the data folder and import them into Python. 
__NOTE__:If you are calculating the source signals for classification problems, you don't need the real source location. You can import the data as follows: 

```python
from scipy.io import loadmat
data = loadmat('/content/data1.mat') # Replace the data path with the path on your system
data1=data['datat1']
```
__Step 3__: Apply the ReLORETA algorithm 

You can do it as follows: 
```python
model1=core.ReLORETA(lr=1e7,max_iter=50)
model1.fit(data1,lead_field)
```
Here I adjusted the learning rate and set it to 1e7. Please read the following on how to adjust the learning rate.



```python
print("Hello, World!")
print("amin")
```
Ok Let's continue
