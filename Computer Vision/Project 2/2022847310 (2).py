import numpy as np
import cv2
import os
import sys

try:
    if not os.path.exists('./2022847310'):
        os.makedirs('./2022847310')
except OSError:
    print("ERROR: Creating Directory")

#IMG = 168 * 192

#Step 1
percentage = str(float(sys.argv[1]))
output = open("./2022847310/output.txt",'w')
output.write("########## STEP 1 ##########\n")
output.write("Input Percentage: "+percentage+"\n")


#Train set matrix
train_set = []
train_name = []
for i in os.listdir('./faces_training/'):
    path = './faces_training/' + i
    img = cv2.imread(path,0)
    train_name.append(i)
    train_set.append(img.reshape(32256,))
    

train_set = np.array(train_set).astype(float)

#Valid set matrix
valid_set = []
valid_name = []
for i in os.listdir('./faces_test/'):
    path = './faces_test/' + i
    img = cv2.imread(path,0)
    valid_name.append(i)
    valid_set.append(img.reshape(32256,))
    
valid_set = np.array(valid_set).astype(float)

#print(train_set.mean(axis=0).shape)


#Subtract by mean
mean_face = train_set.mean(axis=0)
#cv2.imwrite("mean_face.pgm",mean_face.reshape(192,168))

train_set_2 = train_set - mean_face
valid_set_2 = valid_set - mean_face


#SVD

u, s, v = np.linalg.svd(train_set_2,full_matrices=False)

for i in range(1,len(s)+1):
    if (np.power(s[0:i],2)).sum() / (np.power(s,2)).sum() >= float(percentage):
        dimen = i
        break

output.write("Selected Dimension: "+str(dimen)+"\n")
output.write("\n")
u_ = u[:,:dimen]
s_ = s[:dimen]
v_ = v[:dimen,:]
#print(u_.shape,s_.shape,v_.shape)
recon = np.around(np.matmul(np.matmul(u_,np.diag(s_)), v_) + mean_face).astype(int)

output.write("########## STEP 2 ##########\n")
output.write("Reconstruction error\n")

 
residual= []

for i in range(len(train_set)):
    x1 = train_set[i]
    x2 = recon[i]
    
    cv2.imwrite("./2022847310/face"+str(i+1).zfill(2)+".pgm",x2.reshape(192,168))
    
    temp = 0
    for j in range(len(x1)):
        temp += np.power(x1[j]-x2[j],2)
    temp /= len(x1)
    temp = round(temp,4)
    residual.append(temp)

avg = round(sum(residual) / len(residual),4)
output.write("Average: "+ str(f"{avg:.4f}") +"\n")


for i,j in enumerate(residual):
    now = str(i+1).zfill(2)
    output.write(now+": "+str(f"{j:.4f}")+"\n")
output.write("\n")
output.write("########## STEP 3 ##########\n")

train_mat = np.dot(train_set_2,v_.T)
valid_mat = np.dot(valid_set_2,v_.T)
#print(train_mat.shape)
face = []
for i in valid_mat:
    temp = []
    for j in train_mat:
        diff = 0
        for k in range(len(i)):
            diff += np.power(i[k]-j[k],2)
        diff = np.sqrt(diff)
        temp.append(diff)
    face.append(np.argmin(temp))

count=0
for i in os.listdir('./faces_test/'):
    output.write(i+" ==> "+train_name[face[count]]+"\n")
    count+=1
    
