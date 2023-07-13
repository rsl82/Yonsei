import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####

def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''
   
    row = img.shape[0]
    col = img.shape[1]
  
    half_row = int(np.ceil(row/2))
    half_col = int(np.ceil(col/2))
    
    shift_img =img.copy()
    
    if(row%2 == 0):
        shift_img[:half_row] = img[half_row:].copy()
        shift_img[half_row:] = img[:half_row].copy()
    else:
        shift_img[:half_row-1] = img[half_row:].copy()
        shift_img[half_row-1:] = img[:half_row].copy()
    
    temp = shift_img.copy()

    for i in range(row):
        if(col%2 == 0):
            shift_img[i][:half_col] = temp[i][half_col:].copy()
            shift_img[i][half_col:] = temp[i][:half_col].copy()
            
        else:
            shift_img[i][:half_col-1] = temp[i][half_col:].copy()
            shift_img[i][half_col-1:] = temp[i][:half_col].copy()
                
    return shift_img

def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    
    row = img.shape[0]
    col = img.shape[1]
  
    half_row = int(np.ceil(row/2))
    half_col = int(np.ceil(col/2))
    
    shift_img =img.copy()
    
    
    if(row%2 == 0):
        shift_img[:half_row] = img[half_row:].copy()
        shift_img[half_row:] = img[:half_row].copy()
    else:
        shift_img[:half_row] = img[half_row-1:].copy()
        shift_img[half_row:] = img[:half_row-1].copy()
        
    temp = shift_img.copy()

    for i in range(row):
        if(col%2 == 0):
            shift_img[i][:half_col] = temp[i][half_col:].copy()
            shift_img[i][half_col:] = temp[i][:half_col].copy()
            
        else:
            shift_img[i][:half_col] = temp[i][half_col-1:].copy()
            shift_img[i][half_col:] = temp[i][:half_col-1].copy()

    
    return shift_img

def fm_spectrum(img):
    '''
    This function should get the frequency magnitude spectrum of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    You may have to multiply the resultant spectrum by a certain magnitude in order to display it correctly.
    '''
    
    fft_image = np.fft.fft2(img)
    shift_fft = fftshift(fft_image)
    
    spectrum = np.log1p(np.abs(shift_fft))
    
     
    
    return spectrum

def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''
    
    row = img.shape[0]
    col = img.shape[1]
    middle_row = int(np.ceil(row/2))
    middle_col = int(np.ceil(col/2))
   
    
    fft_image = np.fft.fft2(img)
    shift_image = fftshift(fft_image)
    
    low_pass = np.zeros((row,col))
    
    for i in range(row):
        for j in range(col):
            if distance(np.abs(i-middle_row),np.abs(j-middle_col)) <= r:
                low_pass[i,j]=1
    
    filter_image = shift_image * low_pass
    
    
    ishift_image = ifftshift(filter_image)
    ifft_image = np.fft.ifft2(ishift_image)
 
    final_image = np.real(ifft_image)

    
    return final_image

def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''
    
    row = img.shape[0]
    col = img.shape[1]
    middle_row = int(np.ceil(row/2))
    middle_col = int(np.ceil(col/2))
   
    
    fft_image = np.fft.fft2(img)
    shift_image = fftshift(fft_image)
    
    high_pass = np.ones((row,col))
    for i in range(row):
        for j in range(col):
            if distance(np.abs(i-middle_row),np.abs(j-middle_col)) <= r:
                high_pass[i,j] = 0
                
    filter_image = shift_image * high_pass
    
    
    ishift_image = ifftshift(filter_image)
    ifft_image = np.fft.ifft2(ishift_image)
    final_image = np.real(ifft_image)
    
    return final_image

def denoise1(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    row = img.shape[0]
    col = img.shape[1]
    middle_row = int(np.ceil(row/2))
    middle_col = int(np.ceil(col/2))
   
    fft_image = np.fft.fft2(img)
    shift_image = fftshift(fft_image)
    
    denoise = np.ones((row,col))

    for i in range(middle_row-85,middle_row-75):
        for j in range(middle_col-85,middle_col-75):
            denoise[i][j]=0
    for i in range(middle_row+75,middle_row+85):
        for j in range(middle_col+75,middle_col+85):
            denoise[i][j]=0 
            
    for i in range(middle_row-60,middle_row-50):
        for j in range(middle_col-60,middle_col-50):
            denoise[i][j]=0
    for i in range(middle_row+50,middle_row+60):
        for j in range(middle_col+50,middle_col+60):
            denoise[i][j]=0 
            
    for i in range(middle_row-60,middle_row-50):
        for j in range(middle_col+50,middle_col+60):
            denoise[i][j]=0
    for i in range(middle_row+50,middle_row+60):
        for j in range(middle_col-60,middle_col-50):
            denoise[i][j]=0
            
    for i in range(middle_row-85,middle_row-75):
        for j in range(middle_col+75,middle_col+85):
            denoise[i][j]=0
    for i in range(middle_row+75,middle_row+85):
        for j in range(middle_col-85,middle_col-75):
            denoise[i][j]=0
            
       
    
    filter_image = shift_image * denoise

    ishift_image = ifftshift(filter_image)
    ifft_image = np.fft.ifft2(ishift_image)
    final_image = np.real(ifft_image)
    
    
    return final_image

def denoise2(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    
    row = img.shape[0]
    col = img.shape[1]
    middle_row = int(np.ceil(row/2))
    middle_col = int(np.ceil(col/2))
   
    
    fft_image = np.fft.fft2(img)
    shift_image = fftshift(fft_image)
    
    denoise = np.ones((row,col))
    
    
    for i in range(row):
        for j in range(col):
            if distance(np.abs(i-middle_row),np.abs(j-middle_col)) >= 25 and distance(np.abs(i-middle_row),np.abs(j-middle_col)) <=28:
                denoise[i][j]= 0
     
    
    filter_image = shift_image * denoise
    
    
    ishift_image = ifftshift(filter_image)
    ifft_image = np.fft.ifft2(ishift_image)
    final_image = np.real(ifft_image)
    
    return final_image

def distance(x,y):
    return np.sqrt(x**2 + y**2)

#################

# Extra Credit
def dft2(img):
    '''
    Extra Credit. 
    Implement 2D Discrete Fourier Transform.
    Naive implementation runs in O(N^4).
    '''
    
    row = img.shape[0]
    col = img.shape[1]
    result = np.zeros((row,col)).astype(complex)
    
    for u in range(row):
        for v in range(col):
            for x in range(row):
                for y in range(col):
                    result[u,v] += img[x,y] * np.exp(-2j*np.pi*(u*x/row+v*y/col))
     
    return result

def idft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Discrete Fourier Transform.
    Naive implementation runs in O(N^4). 
    '''
    
    row = img.shape[0]
    col = img.shape[1]
    result = np.zeros((row,col)).astype(complex)
    
    for u in range(row):
        for v in range(col):
            
            temp = complex(0)
            
            for x in range(row):
                for y in range(col): 
                    temp += img[x,y] * np.exp(2j*np.pi*(u*x/row + v*y/col))
                    
            result[u,v] = temp/(row*col)
                    
    return result

def fft2(img):
    '''
    Extra Credit. 
    Implement 2D Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
  
            
    
    return img

    
    
def ifft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

if __name__ == '__main__':
    img = cv2.imread('task2_filtering.png', cv2.IMREAD_GRAYSCALE)
    noised1 = cv2.imread('task2_noised1.png', cv2.IMREAD_GRAYSCALE)
    noised2 = cv2.imread('task2_noised2.png', cv2.IMREAD_GRAYSCALE)
   
    
    
    low_passed = low_pass_filter(img)
    high_passed = high_pass_filter(img) 
    denoised1 = denoise1(noised1)
    denoised2 = denoise2(noised2)
   
    # save the filtered/denoised images
    cv2.imwrite('low_passed.png', low_passed)
    cv2.imwrite('high_passed.png', high_passed)
    cv2.imwrite('denoised1.png', denoised1)
    cv2.imwrite('denoised2.png', denoised2)

    # draw the filtered/denoised images
    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_passed, 'Low-pass')
    drawFigure((2,7,3), high_passed, 'High-pass')
    drawFigure((2,7,4), noised1, 'Noised')
    drawFigure((2,7,5), denoised1, 'Denoised')
    drawFigure((2,7,6), noised2, 'Noised')
    drawFigure((2,7,7), denoised2, 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_passed), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_passed), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(noised1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoised1), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(noised2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoised2), 'Spectrum')
    
    plt.show()
    
    
    