import cv2
import numpy as np


def task1_2(src_path, clean_path, dst_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_path' is path for source image.
    'clean_path' is path for clean image.
    'dst_path' is path for output image, where your result image should be saved.

    You should load image in 'src_path', and then perform task 1-2,
    and then save your result image to 'dst_path'.
    """
    #####
    #This code takes really long time ! (About ~20 min per image) 
    #####
    
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)
    result_img = None
    
    # do noise removal
    
    result1 = apply_median_filter(noisy_img,3)
    rms_1 = calculate_rms(result1,clean_img)
    
    result2 = apply_bilateral_filter(noisy_img,7,9,40)
    rms_2 = calculate_rms(result2,clean_img)
    result3 = apply_bilateral_filter(noisy_img,7,9,70)
    rms_3 = calculate_rms(result3,clean_img)
    result4 = apply_bilateral_filter(noisy_img,7,9,100)
    rms_4 = calculate_rms(result4,clean_img)
    
    result5 = apply_my_filter(noisy_img)
    rms_5 = calculate_rms(result5,clean_img)
    
    
    if min(rms_1,rms_2, rms_3, rms_4, rms_5) == rms_1:
        #print("Filtered with: Median filter  RMS: ",rms_1)
        result_img = result1.copy()
    elif min(rms_1,rms_2, rms_3, rms_4, rms_5) == rms_2:
        #print("Filtered with: Bilateral filter  RMS: ",rms_2)
        result_img = result2.copy()    
    elif min(rms_1,rms_2, rms_3, rms_4, rms_5) == rms_3:
       # print("Filtered with: Bilateral filter  RMS: ",rms_3)
        result_img = result3.copy()     
    elif min(rms_1,rms_2, rms_3, rms_4, rms_5) == rms_4:
        #print("Filtered with: Bilateral filter  RMS: ",rms_4)
        result_img = result4.copy()    
    elif min(rms_1,rms_2, rms_3, rms_4, rms_5) == rms_5:
        #print("Filtered with: Nagao-Matsuyama filter  RMS: ",rms_5)
        result_img = result5.copy()    

    cv2.imwrite(dst_path, result_img)
    
    pass


def apply_median_filter(img, kernel_size):
    """
    You should implement median filter using convolution in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is an int value, which determines kernel size of median filter.

    You should return result image.
    """
    
    row = img.shape[0]
    col = img.shape[1]
    channel = img.shape[2]
    
    half_kernel = int(kernel_size / 2)
    
    
    #Method 1

    median_image = img.copy()
    

    for i in range(half_kernel,row-half_kernel):
        for j in range(half_kernel,col-half_kernel):
            for k in range(channel):
                    median_image[i][j][k] = sort(img[i-half_kernel:i+half_kernel*2,j-half_kernel:j+half_kernel*2,k],kernel_size)
       
    for i in range(half_kernel,col - half_kernel):
        for k in range(channel):
            x = [img[0][i-1][k],img[0][i][k],img[0][i+1][k],img[1][i-1][k],img[1][i][k]]
            x.sort()
            median_image[0][i][k] = x[2]
            x = [img[row-1][i-1][k],img[row-1][i][k],img[row-1][i+1][k],img[row-2][i-1][k],img[row-2][i][k]]
            x.sort()
            median_image[row-1][i][k] = x[2]
        
    for i in range(half_kernel,row-half_kernel):
        for k in range(channel):
            x = [img[i-1][0][k],img[i][0][k],img[i+1][0][k],img[i-1][1][k],img[i][1][k]]
            x.sort()
            median_image[i][0][k] = x[2]
            x = [img[i-1][col-1][k],img[i][col-1][k],img[i+1][col-1][k],img[i-1][col-2][k],img[i][col-2][k]]
            x.sort()
            median_image[i][col-1][k] = x[2]
    
    return median_image

def sort(array,kernel_size):
    
    mid = int((kernel_size**2)/2)
    
    row = array.shape[0]
    col = array.shape[1]
    
    temp = [] 
    
    for i in range(row):
        for j in range(col):
            temp.append((array[i][j]))  
    
    temp.sort()
    
    
    return temp[mid]

def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement bilateral filter using convolution in this function.
    It takes at least 4 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s(gaussian function for space)
    'sigma_r' is a int value, which is a sigma value for G_r(gaussian function for range)

    You should return result image.
    """
    row = img.shape[0]
    col = img.shape[1]
    channel = img.shape[2]
    
    half_kernel = int(kernel_size/2)
    
    bi_image = np.zeros([row + 2 * half_kernel,col + 2 * half_kernel,channel])
    bi_image[half_kernel:half_kernel+row,half_kernel:half_kernel+col] = img.copy()
    compare_img = bi_image.copy()
    
    
    for i in range(row):
        for j in range(col):
            for k in range(channel):
                wp = 0.0
                kernel_sum = 0.0
                
                kernel = compare_img[i:i+kernel_size,j:j+kernel_size,k]
                kernel2 = compare_img[i:i+kernel_size,j:j+kernel_size]
               
                for x in range(kernel_size):
                    for y in range(kernel_size):
                        if (i<half_kernel or j<half_kernel) and np.array_equal(kernel2[x,y],[0,0,0]):
                            continue
                        if (i>row-half_kernel or j>col-half_kernel) and np.array_equal(kernel2[x,y],[0,0,0]):
                            continue    
                        space = gaussian_2d(half_kernel-x,half_kernel-y,sigma_s)
                        color = gaussian(kernel[half_kernel,half_kernel] - kernel[x,y],sigma_r)
                        wp += space * color
                        
                        kernel_sum+= space*color*kernel[x,y]
                
                new_color = kernel_sum // wp
                bi_image[i+half_kernel,j+half_kernel,k] = new_color
                
    bi_image = bi_image[half_kernel:half_kernel+row,half_kernel:half_kernel+col]
    
    return bi_image

def gaussian(x,sigma):
    return 1/(np.sqrt(2*np.pi*sigma)) * np.exp(-1 * (x**2) / (2* sigma**2))
def gaussian_2d(x,y,sigma):
    return 1/(2*np.pi*(sigma**2)) * np.exp(-1 * (x**2+y**2) / (2 * sigma**2))


def apply_my_filter(img,kernel_size=5):
    """
    You should implement additional filter using convolution.
    You can use any filters for this function, except median, bilateral filter.
    You can add more arguments for this function if you need.

    You should return result image.
    """
    
    row = img.shape[0]
    col = img.shape[1]
    channel = img.shape[2]
    
    half_kernel = 2
    
    result = np.zeros([row + 2 * half_kernel,col + 2 * half_kernel,channel])
    result[half_kernel:half_kernel+row,half_kernel:half_kernel+col] = img.copy()
    compare_img = result.copy()
    
    for i in range(row):
        for j in range(col):
            for k in range(channel):
                kernel = compare_img[i:i+kernel_size,j:j+kernel_size,k]
                
                temp1 = [kernel[0,1],kernel[0,2],kernel[0,3],kernel[1,1],kernel[1,2],kernel[1,3],kernel[2,2]]
                temp2 = [kernel[1,3],kernel[1,4],kernel[2,2],kernel[2,3],kernel[2,4],kernel[3,3],kernel[3,4]]
                temp3 = [kernel[2,2],kernel[3,1],kernel[3,2],kernel[3,3],kernel[4,1],kernel[4,2],kernel[4,3]]
                temp4 = [kernel[1,0],kernel[1,1],kernel[2,0],kernel[2,1],kernel[2,2],kernel[3,0],kernel[3,1]]
                temp5 = [kernel[0,0],kernel[0,1],kernel[1,0],kernel[1,1],kernel[1,2],kernel[2,1],kernel[2,2]]
                temp6 = [kernel[0,3],kernel[0,4],kernel[1,2],kernel[1,3],kernel[1,4],kernel[2,2],kernel[2,3]]
                temp7 = [kernel[2,2],kernel[2,3],kernel[3,2],kernel[3,3],kernel[3,4],kernel[4,3],kernel[4,4]]
                temp8 = [kernel[2,1],kernel[2,2],kernel[3,0],kernel[3,1],kernel[3,2],kernel[4,0],kernel[4,1]]
                temp9 = [kernel[1,1],kernel[1,2],kernel[1,3],kernel[2,1],kernel[2,2],kernel[2,3],kernel[3,1],kernel[3,2],kernel[3,3]]
                
                min_var =  min(np.var(temp1), np.var(temp2), np.var(temp3), np.var(temp4), np.var(temp5), np.var(temp6), np.var(temp7), np.var(temp8),np.var(temp9))
                if min_var == np.var(temp1):
                    result[i+half_kernel,j+half_kernel,k] = int(np.mean(temp1))
                elif min_var == np.var(temp2):
                    result[i+half_kernel,j+half_kernel,k] = int(np.mean(temp2))
                elif min_var == np.var(temp3):
                    result[i+half_kernel,j+half_kernel,k] = int(np.mean(temp3))
                elif min_var == np.var(temp4):
                    result[i+half_kernel,j+half_kernel,k] = int(np.mean(temp4))
                elif min_var == np.var(temp5):
                    result[i+half_kernel,j+half_kernel,k] = int(np.mean(temp5))
                elif min_var == np.var(temp6):
                    result[i+half_kernel,j+half_kernel,k] = int(np.mean(temp6))
                elif min_var == np.var(temp7):
                    result[i+half_kernel,j+half_kernel,k] = int(np.mean(temp7))
                elif min_var == np.var(temp8):
                    result[i+half_kernel,j+half_kernel,k] = int(np.mean(temp8))
                elif min_var == np.var(temp9):
                    result[i+half_kernel,j+half_kernel,k] = int(np.mean(temp9))
                
                
    result = result[half_kernel:half_kernel+row,half_kernel:half_kernel+col]
    
    return result


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have same sizes.")

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int) - img2.astype(dtype=np.int))
    return np.sqrt(np.mean(diff ** 2))

