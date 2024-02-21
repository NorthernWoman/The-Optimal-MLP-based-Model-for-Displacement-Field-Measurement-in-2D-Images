import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
import skimage 
from skimage import metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import scipy.io as spio
from scipy import interpolate
from scipy.interpolate import griddata 
A = []
B = []
C = []
D = []
A0 = []
B0 = []
C0 = []
D0 = []
#N = list(range(4,5))+list(range(7,9))+list(range(10,14))+list(range(15,18))+list(range(19,23))
N = list(range(0,51,5))
n0 = np.zeros(shape = (168,168,2),dtype = np.float32)
n1 = np.zeros(shape = (168,168,2),dtype = np.float32)
n2 = np.zeros(shape = (168,168,2),dtype = np.float32)
n3 = np.zeros(shape = (168,168,2),dtype = np.float32)
n4 = np.zeros(shape = (168,168,2),dtype = np.float32)
s = 3
for n in N:
	u = np.load('./noise/{n}/{s}/u.npy'.format(n = n,s = s)).reshape(240,240)
	v = np.load('./noise/{n}/{s}/v.npy'.format(n = n,s = s)).reshape(240,240)
	z = np.zeros(shape = (168,168,2),dtype = np.float32)
	z[:,:,0] = u[31:199,31:199]
	z[:,:,1] = v[31:199,31:199]
	#plt.subplot(141)
	n0 = z
	#plt.imshow(n0[:,:,0])
	mass1 = -np.load('./noise/{n}/{s}.npy'.format(n = n,s=s))[31:199,31:199]
	n1 = (z-mass1)**2+n1
	# plt.subplot(142)
	# plt.imshow(mass1[:,:,0])
	mse1 = ((z-mass1)**2).mean()
	ssim1 = ssim(z,mass1,multichannel=True)
	mass2 = np.load('./noise/{n}/{s}d.npy'.format(n = n,s=s))
	for i in range(4,236):			
		for j in range(4,236):				
			mass2[i,j,0] = mass2[i-4:i+4,j-4:j+4,0].mean()
			mass2[i,j,1] = mass2[i-4:i+4,j-4:j+4,1].mean()
	mass2 = mass2[31:199,31:199]
	n2 = (z-mass2)**2+n2
	# plt.subplot(143)
	# plt.imshow(mass2[:,:,0])
	mse2 = ((z-mass2)**2).mean()
	ssim2 = ssim(z,mass2,multichannel=True)
	mass4 = np.load('./noise/{n}/{s}ds.npy'.format(n = n,s=s))
	for i in range(4,236):			
		for j in range(4,236):				
			mass4[i,j,0] = mass4[i-4:i+4,j-4:j+4,0].mean()
			mass4[i,j,1] = mass4[i-4:i+4,j-4:j+4,1].mean()
	mass4 = mass4[31:199,31:199]
	n4 = (z-mass4)**2+n4
	# plt.subplot(143)
	# plt.imshow(mass4[:,:,0])
	mse4 = ((z-mass4)**2).mean()
	ssim4 = ssim(z,mass4,multichannel=True)
	A.append(mse1)
	B.append(mse2)
	D.append(mse4)
	# print(mse1)
	# print(mse2)
	# print('#')
	data = spio.loadmat('./noise0/{n}/{s}/matlab.mat'.format(n = n,s=s))
	mass0 = data['u']
	mass1 = data['v']
	X = []
	Y = []
	U = []
	V = []
	x0 = mass0.shape[0]
	y0 = mass0.shape[1]
	for i in range(15):
		for j in range(15):
			X.append(j)
			Y.append(i)
			U.append(mass0[i,j])
			V.append(mass1[i,j])
	points = np.vstack([X,Y]).T
	grid_x, grid_y = np.mgrid[0:15:182j, 0:15:182j]
	grid_z0 = griddata(points, U, (grid_x, grid_y), method='cubic')
	grid_z1 = griddata(points, V, (grid_x, grid_y), method='cubic')
	# plt.subplot(144)
	# plt.imshow(grid_z0[0:168,0:168])
	grid_z0 = grid_z0[0:168,0:168]
	grid_z1 = grid_z1[0:168,0:168]
	z3 = np.zeros(shape = (168,168,2))
	z3[:,:,0] = grid_z0
	z3[:,:,1] = grid_z1
	mse3 = ((z-z3)**2).mean()
	ssim3 = ssim(z,z3,multichannel=True)
	C.append(mse3)
	n3 = (z-z3)**2+n3
	A0.append(ssim1)
	B0.append(ssim2)
	D0.append(ssim4)
	C0.append(ssim3)
	# print(mse3)
	# print(n)
	# print("#")
	#plt.show()

print(sum(A)/len(A))
print(sum(B)/len(B))
print(sum(D)/len(D))
print(sum(C)/len(C))
print(len(N))
import pandas as pd
data0 = np.array([A,A0,B,B0,D,D0,C,C0]).T
df = pd.DataFrame(data0) 
plt.plot(A)
plt.plot(B)
plt.show()
#df.to_excel('{s}.xlsx'.format(s = s))
# plt.subplot(141)
# plt.imshow((n1[:,:,1]+n1[:,:,0])/19,vmin = 0,vmax = 0.45,cmap = 'jet')
# plt.subplot(142)
# plt.imshow((n2[:,:,1]+n2[:,:,0])/19,vmin = 0,vmax = 0.45,cmap = 'jet')
# plt.subplot(143)
# plt.imshow((n4[:,:,1]+n4[:,:,0])/19,vmin = 0,vmax = 0.45,cmap = 'jet')
# plt.subplot(144)
# plt.imshow((n3[:,:,1]+n3[:,:,0])/19,vmin = 0,vmax = 0.45,cmap = 'jet')
# plt.show()




