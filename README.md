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
### Parameters, attributes, and methods
To apply the ReLORETA algorithm, you need to create an object by calling the core module and then the ReLORETA class which uses the following parameters:
```python
core.ReLORETA(lambda2 = 0.05, dimension=3,n_source=82, epsilon=1e-29, max_iter=100,lambda1=1,lr=1e8)
```
where 
lambda2: eLORETA regularisation parameter (default 0.05)

dimension: The Leadfield matrix dimension (default 3. See the example below)

lambda1: ReLORETA regularisation parameter

n_source: number of voxels (dipoles or source points) set by the user. (default 82)

epsilon:  the threshold to stop ReLORETA. Unless there is a good justification, it is usually set to a high value to let max_iter stop the algorithm (default 1e-29). 

max_iter: The maximum number of iterations before stopping the algorithm (default 100)

lr: ReLORETA learning rate (default 1e8. However, it needs adjustment by the user as it can significantly vary depending on the EEG data. See the example below)

### Example:
Assume you have gathered EEG data from subjects using 61 electrodes, namely, 'Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 'Cz',
'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5', 'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5','P7', 'PO3', 'PO7', 'Oz', 'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8',
'FC2', 'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8','TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2', and want to calculate (reconstruct) the source signals inside the brain using ReLORETA. 

1- The first step is to calculate a leadfield matrix using a simulated head model. This lead field matrix will be used to start the ReLORETA algorithm. The following code can be used to calculate the leadfield matrix. You can customize the source points and the electrodes in this code using 


As the leadfield matrix is simulated, the reconstructed source signals are not accurate. ReLORETA will iteratively update the leadfield matrix to achieve an accurace estimation of the source signals. 
```python
print("Hello, World!")
print("amin")
```
Ok Let's continue
