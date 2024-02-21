import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
import skimage 
from skimage import io 
from skimage import metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import scipy.io as spio
from scipy import interpolate
from scipy.interpolate import griddata 
mass = io.imread('./0.png')
print(mass.min())
print(mass.max())
print(mass.max()-mass.min())