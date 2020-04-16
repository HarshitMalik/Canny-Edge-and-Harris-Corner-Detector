import numpy as np
import skimage
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage.filters import convolve
from scipy import misc
import numpy as np

# Function to do convolution of input image with sobel filter to smoothen the image and finding gradients
def convolveWithGaussianDerivative(img):
  Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
  Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
  
  Ix = ndimage.filters.convolve(img, Kx)
  Iy = ndimage.filters.convolve(img, Ky)
  theta = np.arctan2(Iy, Ix)
  return Ix, Iy, theta

# Function to perform non maximal supppression by quantizing angles to 4 bins- 0, 45, 95 and 135 degrees
def nonMaxSuppression(img, D):
  M, N = img.shape
  out = np.zeros((M,N), dtype=np.float32)
  angle = D * 180.0 / np.pi
  angle[angle < 0] += 180

  for i in range(1,M-1):
    for j in range(1,N-1):
      x = 1.0
      y = 1.0

      #angle 0
      if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
        x = img[i, j+1]
        y = img[i, j-1]
      
      #angle 45
      elif (22.5 <= angle[i,j] < 67.5):
        x = img[i+1, j-1]
        y = img[i-1, j+1]
    
      #angle 90           
      elif (67.5 <= angle[i,j] < 112.5):
        x = img[i+1, j]
        y = img[i-1, j]
        
      #angle 135
      elif (112.5 <= angle[i,j] < 157.5):
        x = img[i-1, j-1]
        y = img[i+1, j+1]

      if (img[i,j] >= x) and (img[i,j] >= y):
        out[i,j] = img[i,j]
      else:
        out[i,j] = 0.0
  return out

#Function to do thresholding to get weak and strong edges
def threshold(img, T_low, T_high):
  out = np.zeros(img.shape,dtype='float32')
  strong_rows, strong_cols = np.where(img >= T_high)
  weak_rows, weak_cols = np.where((img <= T_high) & (img >= T_low))
  out[strong_rows, strong_cols] = 1.0 #strong pixel edge value
  out[weak_rows, weak_cols] = 0.5 #weak pixel edge value
  return out

#Function to perform hysteresis to get connected edges and ignoring disconnected weak edges
def hysteresis(img):
  M, N = img.shape
 
  img1 = np.copy(img)
  for i in range(1, M):
    for j in range(1, N):
      if img1[i, j] == 0.5:
        if img1[i, j + 1] == 1.0 or img1[i, j - 1] == 1.0 or img1[i - 1, j] == 1.0 or img1[i + 1, j] == 1.0 or img1[i - 1, j - 1] == 1.0 or img1[i + 1, j - 1] == 1.0 or img1[i - 1, j + 1] == 1.0 or img1[i + 1, j + 1] == 1.0:
          img1[i, j] = 1.0
 
  for i in range(M - 1, 0, -1):
    for j in range(N - 1, 0, -1):
      if img1[i, j] == 0.5:
        if img1[i, j + 1] == 1.0 or img1[i, j - 1] == 1.0 or img1[i - 1, j] == 1.0 or img1[i + 1, j] == 1.0 or img1[i - 1, j - 1] == 1.0 or img1[i + 1, j - 1] == 1.0 or img1[i - 1, j + 1] == 1.0 or img1[i + 1, j + 1] == 1.0:
          img1[i, j] = 1.0

  for i in range(1, M):
    for j in range(N - 1, 0, -1):
      if img1[i, j] == 0.5:
        if img1[i, j + 1] == 1.0 or img1[i, j - 1] == 1.0 or img1[i - 1, j] == 1.0 or img1[i + 1, j] == 1.0 or img1[i - 1, j - 1] == 1.0 or img1[i + 1, j - 1] == 1.0 or img1[i - 1, j + 1] == 1.0 or img1[i + 1, j + 1] == 1.0:
          img1[i, j] = 1.0     
          
  for i in range(M-1,0,-1):
    for j in range(1,N):
      if img1[i, j] == 0.5:
        if img1[i, j + 1] == 1.0 or img1[i, j - 1] == 1.0 or img1[i - 1, j] == 1.0 or img1[i + 1, j] == 1.0 or img1[i - 1, j - 1] == 1.0 or img1[i + 1, j - 1] == 1.0 or img1[i - 1, j + 1] == 1.0 or img1[i + 1, j + 1] == 1.0:
          img1[i, j] = 1.0
        else:
          img1[i,j] = 0.0
 
  return img1

input_img = 'bicycle' #input image name
img = mpimg.imread('data/'+input_img+'.bmp').astype('float32')/255.0 #Read Image and convert to float
#plt.imshow(img)

img = (np.dot(img[...,:3], [0.299, 0.587, 0.114])).astype('float32') #RGB2Gray
#plt.imshow(img,cmap=plt.get_cmap('gray'))
plt.imsave('output/edge_detection/'+input_img+'_grey_scale.jpg',img,cmap=plt.get_cmap('gray'))

Ix, Iy, theta = convolveWithGaussianDerivative(img) #x and y gradients and angle
G = np.hypot(Ix, Iy)
G = G / G.max()
plt.imsave('output/edge_detection/'+input_img+'_gradient_itensity_x.jpg',Ix,cmap=plt.get_cmap('gray'))
plt.imsave('output/edge_detection/'+input_img+'_gradient_itensity_y.jpg',Iy,cmap=plt.get_cmap('gray'))
plt.imsave('output/edge_detection/'+input_img+'_gradient_itensity.jpg',G,cmap=plt.get_cmap('gray'))
plt.imsave('output/edge_detection/'+input_img+'_gradient_theta.jpg',theta,cmap=plt.get_cmap('gray'))

thinned_edge_image = nonMaxSuppression(G, theta)
plt.imsave('output/edge_detection/'+input_img+'_thin_edge.jpg',thinned_edge_image,cmap=plt.get_cmap('gray'))

T_low = 0.1 #lower threshold
T_high = 0.25 #upper threshold
thresholded_img = threshold(thinned_edge_image, T_low, T_high)
plt.imsave('output/edge_detection/'+input_img+'_threshold.jpg',thresholded_img,cmap=plt.get_cmap('gray'))
#plt.imshow(thresholded_img,cmap=plt.get_cmap('gray'))

hyst_img = hysteresis(thresholded_img)
#plt.imshow(hyst_img,cmap=plt.get_cmap('gray'))
plt.imsave('output/edge_detection/'+input_img+'_edges.jpg',hyst_img,cmap=plt.get_cmap('gray'))
