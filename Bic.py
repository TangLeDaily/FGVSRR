import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

def BiBubic(x):
    x=abs(x)
    if x<=1:
        return 1-2*(x**2)+(x**3)
    elif x<2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0

def BiCubic_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=i*(scrH/dstH)
            scry=j*(scrW/dstW)
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            tmp=0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii<0 or y+jj<0 or x+ii>=scrH or y+jj>=scrW:
                        continue
                    tmp+=img[x+ii,y+jj]*BiBubic(ii-u)*BiBubic(jj-v)
            retimg[i,j]=np.clip(tmp,0,255)
    return retimg
def load_img(filepath):
    img = Image.open(filepath)
    return img

def get_pic_name_lis(rootpath, path): #datasets/train/   and   000/
    imgls = []
    for i in os.listdir(rootpath + "input/" + path):
        imgls.append(i)
    imglss = sorted(imgls)
    return imglss
class test_data_set():
    def __init__(self, rootpath, path): #datasets/test/  and 000/
        super(test_data_set, self).__init__()
        self.feat_ID = "test"
        self.input_path = rootpath + "input/" + path  #datasets/test/input/000/
        self.target_path = rootpath + "target/" + path
        self.pic_name_lis = get_pic_name_lis(rootpath, path)


    def get(self, idx):
        lrc = load_img(self.input_path + self.pic_name_lis[idx])
        return lrc

def main(name):
    data = test_data_set("datasets/test/", name+"/")
    for i in range(100):
        image = np.array(data.get(i))
        image3 = BiCubic_interpolation(image, image.shape[0] * 2, image.shape[1] * 2)
        image3 = Image.fromarray(image3.astype('uint8')).convert('RGB')
        image3.save('data/{}/BIC/{}.png'.format(name, i))
        print(i, end=",")
    print("Over {}".format(name))

if __name__ == "__main__":
    main("000")
    main("011")
    main("015")
    main("020")