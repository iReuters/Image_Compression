import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import easygui as g
import time
import os

class Node: #节点类
    def __init__(self,data,freq,left=None,right=None):
        self.data=data
        self.freq=freq
        self.left=left
        self.right=right

class Huff: #huffman树类
    def __init__(self,img):
        self.dict={}
        self.code=[] #编码表
        self.hist_dic={}
        self.img=img
        hist,bins = np.histogram(self.img.ravel(),256,[0,256])
        bins=bins.tolist()
        hist=hist.tolist()

        for hist,bin in zip(hist,bins):
            self.hist_dic[bin]=hist
        dic=self.hist_dic.copy()
        for key in dic.keys():
            if self.hist_dic[key]==0:
                del(self.hist_dic[key]) #删除所有0值元素

        self.bins=self.hist_dic.keys()
        self.hist=[self.hist_dic[x] for x in self.bins]
        print(self.bins)
        print(self.hist)

    def CreateTree(self):
        charList=self.bins #字符列表
        freqList=self.hist #频数列表
        minHeap=[Node(c,f) for c,f in zip(charList,freqList)] #生成节点列表

        while(len(minHeap)!=1):
            minHeap=sorted(minHeap,key=lambda x:x.freq,reverse=True) #按频数从大到小排列节点
            intNode=Node(None,minHeap[-1].freq+minHeap[-2].freq)# 新节点的频数为最小的两个节点之和
            intNode.left=minHeap[-2]
            intNode.right=minHeap[-1]
            minHeap.pop()
            minHeap.pop() #从节点列表中删除最小的两个节点
            minHeap.append(intNode)#将新节点添加到节点列表中

        return minHeap[0]

    def Code(self,tree,s=""):
        if tree.data is not None:
            print(tree.data,end=" ")
            print(s)
            self.dict[tree.data]=s
            return
        self.Code(tree.left,s+'0') #每个左节点编码为0
        self.Code(tree.right,s+'1')#每个右节点编码为1

    def encode(self):
        flat=self.img.flatten().tolist()
        for pix in flat:
            self.code.append(self.dict[pix])
        return self.code

    def decode(self,tree):
        '''
        :param tree: Huffman树，从.bin文件中加载出来
        :return: 解码出的一幅图像的像素值
        '''
        current=tree
        self.string=[]
        for code in self.code:
            for bit in code:
                if bit=="0":
                    current=current.left
                else:
                    current=current.right

            if current.left==None and current.right==None:#没有子节点的节点
                self.string.append(current.data)#获取该节点保存的像素值
                current=tree
        return self.string


def compress(image, fileDir):
    '''
    压缩与解压函数
    :param image: 输入图像
    :param fileDir: 图像目录, 用于保存解码后的图像
    :return: 无
    '''
    obj = Huff(image)
    root = obj.CreateTree()#创建huffman树
    obj.Code(root)
    coded_image = obj.encode()
    temp = 0
    for code in coded_image:
        temp += len(code)
    shape = image.shape
    with open(fileDir + '/encode.bin', 'wb') as encode:#以二进制形式保存huffman树对象
        pickle.dump(root, encode)

    with open(fileDir + '/encode.bin', 'rb') as decode: #读取编码
        decoded = pickle.load(decode)
    ret = obj.decode(decoded) #解码
    ret = np.array(ret, np.uint8)
    ret_image = np.reshape(ret, shape)
    out = cv2.cvtColor(ret_image, 0)

    cv2.imwrite(fileDir + 'compressed_image.jpg', out)
    compSize = getFileSize(fileDir + 'compressed_image.jpg')

    print("压缩前图片大小：%.2fKB" % originalSize)
    print("压缩后图片大小：%.2fKB" % compSize)
    print("压缩率：%.2f%%" % (compSize * 100 / originalSize))
    print("PSNR：%.2f%%" % psnr(image, ret_image))
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(image, cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.title('Decoded image')
    plt.imshow(out)
    plt.show()


def selectFile():
    msg = '浏览文件并打开'
    title = '选择需要压缩的图片'
    default = './*.png'
    filetypes = ["*.png", "*.jpg", "*.bmp"]
    filePath = g.fileopenbox(msg, title, default, filetypes)
    return filePath


def getFileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize / float(1024)
    return round(fsize, 2)


def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
    while True:
        print('*' * 10)
        print('注意文件路径不能存在中文')
        print('*' * 10)
        filePath = selectFile()
        print(filePath)
        originalSize = getFileSize(filePath)
        if filePath.isascii() != True:
            print('路径含有中文,请重新选择')
            time.sleep(2)
        else:
            fileDir = filePath[0: filePath.rfind('\\')]
            print(fileDir)
            image = cv2.imread(filePath, 0)
            start = time.time()
            compress(image, fileDir)
            end = time.time()
            print('压缩与解压总用时：%.1fs' % (end - start))
            print('压缩后的图片已写入到' + fileDir + '目录下')
            time.sleep(100)
            break

