# nang cap kernel 
import matplotlib.pyplot as plt 
import cv2
import numpy as np  

img = cv2.imread("image_examnple.jpg") 
np.random.seed(48)

img = cv2.resize(img, (400, 400))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# doi voi anh thi thong so cuoi cung la chi so cua chieu sau
print(img_gray.shape)
# print(img.shape)

class conv2d():
    def __init__(self, input, numofkernel, kernalSize, padding=0, stride=1):
        # self.input = input
        self.input = np.pad(input, [(padding, padding), (padding, padding)], 'constant')
        # height  = input[0]
        # width = input[1]
        self.stride = stride
        
        
        self.kernalSize = kernalSize   # kich thuoc cua kernel
        # self.kernel = np.random.rand(kernalSize, kernalSize)  
        self.kernel = np.random.randn(numofkernel ,kernalSize, kernalSize)  
        # self.height, self.width = input.shape
        # self.result = np.zeros((self.height - kernalSize + 1, self.width - kernalSize + 1))
        # cai tien result 
        self.result = np.zeros((
            int((self.input.shape[0] - self.kernel.shape[1])/self.stride) + 1,
            int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1,
            self.kernel.shape[0]
        ))
   
     
    def getRoi(self):
        index_row_input = self.input.shape[0] - self.kernel.shape[1] 
        index_col_input = self.input.shape[1] - self.kernel.shape[2] 
        for row in range(int(index_row_input/self.stride) + 1):
            for col in range(int(index_col_input/self.stride) + 1):
                roi = self.input[row*self.stride:row*self.stride+self.kernel.shape[1], 
                                 col*self.stride:col*self.stride+self.kernel.shape[2]] # lan dau chay la 0 +3
                yield row, col, roi

    def operator(self):
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.getRoi():
                self.result[row, col, layer] = np.sum(roi * self.kernel[layer,:,:])
        # for row, col, roi in self.getRoi():
        #     self.result[row, col] = np.sum(roi * self.kernel)
        return self.result

class Relu():
    def __init__(self, input):
        self.input = input
        print(self.input.shape)
        self.result = np.zeros_like(self.input)  # Correctly initialize self.result with the same shape as self.input

    def operator(self):
        for layer in range(self.input.shape[2]):  # Iterate over the number of layers
            for row in range(self.input.shape[0]):  # Iterate over the number of rows
                for col in range(self.input.shape[1]):  # Iterate over the number of columns
                    self.result[row, col, layer] = 0 if self.input[row, col, layer] < 0 else self.input[row, col, layer]
                   
        return self.result
    
class LeakRelu():
    def __init__(self, input):
        self.input = input
        self.result = np.zeros((
            self.input.shape[0],
            self.input.shape[1],
            self.input.shape[2]
        ))
        
    def operator(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row, col, layer] = 0.1*self.input[row, col, layer] if self.input[row, col, layer] < 0 else self.input[row, col, layer]
                    
        return self.result
    
class MaxPooling():
    def __init__(self, input, poolingSize=2):
        self.input = input
        print(self.input.shape)
        self.poolingSize = poolingSize
        self.result = np.zeros((
            int(self.input.shape[0]/self.poolingSize),
            int(self.input.shape[1]/self.poolingSize),
            self.input.shape[2]
        ))
        
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(0, self.input.shape[0], self.poolingSize):
                for col in range(0, self.input.shape[1], self.poolingSize):
                    # print(row, col, layer)
                    self.result[int(row/self.poolingSize), int(col/self.poolingSize), layer] = np.max(self.input[row:row+self.poolingSize, 
                                                                                                                 col:col+self.poolingSize,
                                                                                                                 layer])
        return self.result

    
        
                    
class SoftMax():
    # so node minfh du doan  
    def __init__(self, input, nodes):
        self.input = input
        self.nodes = nodes
        # y = biass + weight*x
        #  self.input.flatten() = self.input.shape[0] + self.input.shape[1] * self.input.shape[2]
        self.flatten = self.input.flatten()
        print(self.flatten.shape)
        print(self.flatten.shape[0])
        # self.weights = np.random.randn(self.flatten)/self.flatten.shape[0]
        self.weights = np.random.randn(self.flatten.shape[0])/self.flatten.shape[0]
        # self.weights = np.random.randn(self.flatten.shape[0])
        self.bias = np.random.randn(self.nodes)
    
    def operator(self):
        totals = np.dot(self.flatten, self.weights) + self.bias        
        #  luy thua e 
        exp = np.exp(totals)
        return exp/sum(exp)
        
        
        
    


# xu li anh co layer cho 48 layer cho relu 

img = conv2d(img_gray, 32, 3).operator()
# img_relu = Relu(img).operator()
# img_leakrelu = LeakRelu(img).operator()
img_maxpooling = MaxPooling(img).operate()  


#  in tất cả hìh ảnh lên một lầ
fig = plt.figure(figsize=(8, 10))
for i in range(8):
    
    plt.subplot(4, 2, i+1)
    plt.imshow(img_maxpooling[:,:,i], cmap='gray')
    plt.axis('off')
plt.savefig("output_maxpooling.jpg")
plt.show()


softmax = SoftMax(img_gray, 10).operator()
print(softmax)
