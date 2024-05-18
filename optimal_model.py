import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from torchsummary import summary
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import skimage 
from skimage import io
import time
from networks2.conv_layers import *
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
start_time = time.time()
def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    #print(mgrid.size())
    return mgrid
#get_mgrid(200)

# taken from work Sitzmann, Vincent, et al. "Implicit neural representations with periodic activation functions." Advances in neural information processing systems 33 (2020): 7462-7473.
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30,need_sigmoid = True, need_tanh = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Conv2d(in_features, out_features, bias=bias, kernel_size = 1, padding = 0)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30, need_sigmoid = True, need_tanh = False):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Conv2d(hidden_features, out_features, kernel_size = 1, padding = 0)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        #self.net.append(nn.Sigmoid())
        if need_sigmoid:
            self.net.append(nn.Sigmoid())
        elif need_tanh:
            self.net.append(nn.Tanh())
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
         

# Reading images
image = io.imread('./{n}/0.png'.format(n = n))
image0 = io.imread('./{n}/1.png'.format(n = n))
d = image.shape[0]-8
d0 = image.shape[0] - 4
images_warp_np = np.array(image[4:d0,4:d0]).reshape(-1,d,d)
images_warp_np = np.array(images_warp_np/255,dtype = np.float32)
img_gt_batch_var = torch.from_numpy(images_warp_np).type(dtype).cuda()

images_warp_np0 = np.array(image0[4:d0,4:d0]).reshape(-1,d,d)
images_warp_np0 = np.array(images_warp_np0/255,dtype = np.float32)
img_gt_batch_var0 = torch.from_numpy(images_warp_np0).type(dtype).cuda()

# Make a straight grid
xy_grid_batch = []
coords_x = np.linspace(-1, 1, d)
coords_y = np.linspace(-1, 1, d)
xy_grid = np.stack(np.meshgrid(coords_x, coords_y), -1)
xy_grid_var = np_to_torch(xy_grid.transpose(2,0,1)).type(dtype).cuda()
xy_grid_batch_var = xy_grid_var.repeat(1, 1, 1, 1)
grid_input_single_gd = xy_grid_var.detach().clone()


# Image generator
img_siren = Siren(in_features=2, out_features=1, hidden_features=256, 
                  hidden_layers=4, outermost_linear=True)
img_siren.cuda()
summary(img_siren, (2,d,d))
ground_truth = img_gt_batch_var
model_input = grid_input_single_gd

# Grid generator
img_grid = conv_layers(2,2, num_hidden = 400, need_sigmoid = False, need_tanh = True)
img_grid.cuda()
summary(img_grid, (2,d,d))

vec_scale = 1.1
model_params_list = [{'params':img_siren.parameters()}]
model_params_list.append({'params':img_grid.parameters()})


total_steps = 10001 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 100

optim = torch.optim.Adam(lr=1e-4, params=model_params_list)

# The first step in paragraph 3 in paper Daria Mangileva. 2024. The Optimal MLP-based Model for Displacement Field Measurement in 2D Images and Its Application Perspective 
for step in range(total_steps):
    model_output, coords = img_siren(model_input) 
    h1 = img_grid(model_input)
    grid_output = 1.1*torch.cat([h1])
    
    loss = torch.nn.functional.l1_loss(ground_truth, model_output[0])
    loss += torch.nn.functional.l1_loss(model_input, grid_output)
    loss += torch.nn.functional.mse_loss(ground_truth, model_output[0])
    loss += torch.nn.functional.mse_loss(model_input, grid_output)
    if not step % steps_til_summary:
        print('Epoch %d, loss = %.06f' % (step, float(loss)))

        
        #plt.imshow(model_output.cpu().view(200,200).detach().numpy())
        #im = np.array(model_output.cpu().view(200,200).detach().numpy())
        #io.imsave('./s4.png',im)
        #plt.savefig('./1.png')
    optim.zero_grad()
    loss.backward()
    optim.step()
torch.save(img_siren.state_dict(), './{n}/best-model-parameters44s.pt'.format(n = n))
torch.save(img_grid.state_dict(), './{n}/best-model-parameters044s.pt'.format(n = n))

img_siren = Siren(in_features=2, out_features=1, hidden_features=256, 
                  hidden_layers=4, outermost_linear=True)
img_siren.cuda()
img_siren.load_state_dict(torch.load('./{n}/best-model-parameters44s.pt'.format(n = n)))
img_siren.eval()

img_grid = conv_layers(2,2, num_hidden = 400, need_sigmoid = False, need_tanh = True)
img_grid.cuda()
img_grid.load_state_dict(torch.load('./{n}/best-model-parameters044s.pt'.format(n = n)))
img_grid.eval() 

  

# The second step in paragraph 3 in paper Daria Mangileva. 2024. The Optimal MLP-based Model for Displacement Field Measurement in 2D Images and Its Application Perspective
model_params_list = [{'params':img_grid.parameters()}]
optimizer = torch.optim.Adam(model_params_list, lr=1e-4)
num_iter_i = 5001
grid = []
losses = []
for epoch in range(num_iter_i):
    optimizer.zero_grad()    
    h0 = img_grid(model_input) 
    h1 = h0+torch.randn_like(h0)*0.0005
    h11 = 1.1*torch.cat([h0])
    h = 1.1*torch.cat([h1])
    generated, coords = img_siren(h) 
    loss = torch.nn.functional.l1_loss(img_gt_batch_var0, generated[0]) 
    loss += torch.nn.functional.mse_loss(img_gt_batch_var0, generated[0])
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch %d, loss = %.09f' % (epoch, float(loss)))   
    grid.append(h11)
    losses.append(round(float(loss),9))

ind = losses.index(min(losses))
torch.save(grid[ind],'./{n}/tensor_2344s.pt'.format(n = n))
refined_xy = torch.load('./{n}/tensor_2344s.pt'.format(n = n))
refined_warp = refined_xy - xy_grid_batch_var
refined_uv = torch.cat(((d - 1.0)*refined_warp[:, 0:1, :, :]/2 , (d - 1.0)*refined_warp[:, 1:2, :, :]/2), 1)                 
warp_img= refined_uv[0].detach().cpu().numpy().transpose(1,2,0)
z = np.zeros(shape = (d,d,2))
for i in range(4,d-4):
    for j in range(4,d-4):
        warp_img[i,j,0] = warp_img[i-4:i+4,j-4:j+4,0].mean()
        warp_img[i,j,1] = warp_img[i-4:i+4,j-4:j+4,1].mean()
np.save('./{n}.npy'.format(n = n), warp_img)
end_time = time.time()  # время окончания выполнения
execution_time = round(end_time - start_time)
print(execution_time)

