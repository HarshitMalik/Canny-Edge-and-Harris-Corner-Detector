import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from scipy import ndimage

# Function to do convolution of input image with sobel filter to smoothen the image and finding gradients
def convolveWithGaussianDerivative(img):
  Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
  Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
  
  Ix = ndimage.filters.convolve(img, Kx)
  Iy = ndimage.filters.convolve(img, Ky)
  
  return Ix, Iy

# Function to determine R score for each pixel
def findRScore(img, Iy, Ix, m, k):
    Ixx = Ix**2
    Ixy = Iy*Ix
    Iyy = Iy**2
    height, width = img.shape[0], img.shape[1]

    r_mat = np.zeros([height, width],dtype='float64')

    for y in range(m, height-m):
        for x in range(m, width-m):
            Kxx = Ixx[y-m:y+m+1, x-m:x+m+1]
            Kxy = Ixy[y-m:y+m+1, x-m:x+m+1]
            Kyy = Iyy[y-m:y+m+1, x-m:x+m+1]
            Sxx = Kxx.sum()
            Sxy = Kxy.sum()
            Syy = Kyy.sum()

            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
            r_mat[y][x] = r
    return r_mat

# Function to detect corners based on threshold values and r score
def findCorners(img, r_mat):
  corner_list = []
  T = 0.01*r_mat.max()
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      if r_mat[y][x] > T:
        corner_list.append([r_mat[y][x], y, x])
  return corner_list

#Funcion to return neighbours of a pixel present in the corners list
def getIndecies(corner_list, y, x):
  ind_list = []
  for i in range(len(corner_list)):
    if corner_list[i][1] != -1:
      if abs(corner_list[i][1] - y) < 3 and abs(corner_list[i][2] - x) < 3:
        ind_list.append(i)
  return ind_list

#Function to non-maximum suppression
def nonMaximalSuppression(corner_list):
  final_corner_list = []
  sorted_ind = np.argsort(np.array(corner_list)[:,0])
  for i in range(len(sorted_ind)):
    if corner_list[sorted_ind[i]][1] != -1:
      y = corner_list[sorted_ind[i]][1]
      x = corner_list[sorted_ind[i]][2]
      final_corner_list.append([corner_list[sorted_ind[i]][0],int(y),int(x)])
      corner_list[sorted_ind[i]][1] = -1
      corner_list[sorted_ind[i]][2] = -1
      ind_list = getIndecies(corner_list, y, x)
      for j in range(len(ind_list)):
        corner_list[ind_list[j]][1] = -1
        corner_list[ind_list[j]][2] = -1
  return final_corner_list  

# Function to highlight corners in the input image
def getCornerImg(img, corner_list):
  out1 = np.zeros(img.shape,img.dtype)
  out2 = img.copy()
  
  for i in range(len(corner_list)):
    y = corner_list[i][1]
    x = corner_list[i][2]
    out1[y][x][0] = 1.0
    out1[y][x][1] = 1.0
    out1[y][x][2] = 1.0
    out2[y][x][0] = 1.0
    out2[y][x][1] = 0.0
    out2[y][x][2] = 0.0
    
  return out1, out2

input_img = 'plane' # Input image name
img = mpimg.imread('data/'+input_img+'.bmp').astype('float32')/255.0 #Read Image and convert to float
#plt.imshow(img)

Ix, Iy = convolveWithGaussianDerivative((np.dot(img[...,:3], [0.299, 0.587, 0.114])).astype('float32'))

r_mat = findRScore(img, Ix, Iy, m=4, k=0.04)
temp = r_mat/r_mat.max()

plt.imsave('output/corner_detection/'+input_img+'_R_Value.jpg',temp,cmap=plt.get_cmap('gray'))

corner_list = findCorners(img, r_mat)
corner_list_final = nonMaximalSuppression(np.copy(corner_list))
print('Corner pixel detected : ' + str(len(corner_list)))
print('Corner pixel after NMS : ' + str(len(corner_list_final)))

out1, out2 = getCornerImg(img,corner_list)
plt.imsave('output/corner_detection/'+input_img+'_corner_points.jpg',out1)
plt.imsave('output/corner_detection/'+input_img+'_corners.jpg',out2)

out1, out2 = getCornerImg(img,corner_list_final)
plt.imsave('output/corner_detection/'+input_img+'_corner_points_final.jpg',out1)
plt.imsave('output/corner_detection/'+input_img+'_corners_final.jpg',out2)
