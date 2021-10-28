import cv2 as cv
import os
import numpy as np
import skimage
import math

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

#删掉文件夹内另一个文件内没有的文件
def delt_image_smname(source_path,target_path):#target_path有更少的文件
    target_names=sorted(os.listdir(target_path))
    source_names = sorted(os.listdir(source_path))
    # con_target_names=target_names.copy()
    # con_source_names=source_names.copy()
    # for i in range(len(target_names)):
    #     con_target_names[i]=target_names[i].split('-')[0]
    # for i in range(len(source_names)):
    #     con_source_names[i]=source_names[i].split('.')[0]
    for i in range(len(source_names)):
        if source_names[i] not in target_names:
            os.remove(os.path.join(source_path,source_names[i]))

def del_frame(image,size):
    img=image[int(size):image.shape[0]-size,size:image.shape[1]-size]
    return img

#将GST模拟HMI分辨率的模糊效果，用来做对齐
def downsmp_1843_122(GST_image_1843):
    img2 = cv.GaussianBlur(GST_image_1843, (19, 19), 1)
    img2 = cv.pyrDown(img2)
    img2 = cv.GaussianBlur(img2, (5, 5), 1)
    img2 = cv.pyrDown(img2)
    img2 = cv.GaussianBlur(img2, (5, 5), 1)
    img2 = cv.pyrDown(img2)
    img2 = cv.resize(img2, (122, 122))
    img2 = cv.GaussianBlur(img2, (3, 3), 1)
    return img2

#将GST模拟HMI分辨率的模糊效果，用来做对齐
def downsmp_900_138(GST_image_1843):
    img2 = cv.pyrDown(GST_image_1843)
    img2 = cv.pyrDown(img2)
    img2 = cv.resize(img2, (138, 138))
    img2 = cv.GaussianBlur(img2, (3, 3), 0.8)
    return img2


#将GST下采样到模拟HMI的超分辨率目标，用来作为数据集
def downsmp_1843_488(GST_image_1843):
    img2 = cv.pyrDown(GST_image_1843)
    img2 = cv.resize(img2, (488, 488))
    img2 = cv.GaussianBlur(img2, (3, 3), 1.5)
    return img2

#将GST下采样到模拟HMI的超分辨率目标，用来作为数据集
def downsmp_900_552(GST_image_1843):
    img2 = cv.resize(GST_image_1843, (552, 552))
    img2 = cv.GaussianBlur(img2, (3, 3), 1.5)
    return img2

#beta提高图像亮度
def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

#将GST 和结果降低至HMI分辨率
def blur_GST():
    read_path='/home/songwei/dengjunlan/my_pix2pix/result/test_l1/test_results(all)/'
    save_path='/home/songwei/dengjunlan/my_pix2pix/draw/HMI_downpredict_downGST_100'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_list=sorted(os.listdir(read_path))
    for ll in range(len(img_list)):
        a=cv.imread(os.path.join(read_path,img_list[ll]),1)
        size=a.shape[0]
        c = np.zeros((size, size * 3,3))
        HMI=a[:,:size,:]
        predict=a[:, size:size * 2, :]
        GTS=a[:,size*2:,:]
        predict = cv.GaussianBlur(predict, (9, 9), 3)
        predict = cv.GaussianBlur(predict, (9, 9), 2)
        predict=Contrast_and_Brightness(1,40,predict)
        GTS = cv.GaussianBlur(GTS, (9, 9), 3)
        GTS = cv.GaussianBlur(GTS, (9, 9), 2)
        GTS = Contrast_and_Brightness(1, 40, GTS)
        c[:,:size,:]=HMI
        c[:,size:size*2,:]=predict
        c[:, size*2:, :]=GTS
        cv.imwrite(os.path.join(save_path,img_list[ll]),c)


#截取中间的256像素作为数据集
def cut_256():
    read_path='E:\wang_test_dataset_4up/'
    save_path='E:\wang_test_dataset_256/'
    check_dir(save_path)
    oringe_names = sorted(os.listdir(read_path))
    for i in range(833):
        img=cv.imread(os.path.join(read_path,oringe_names[i]),0)
        img1=np.zeros((256,512))
        w=int(img.shape[0]/2)
        img1[:,256:]=img[w-128:w+128,w-128:w+128]
        cv.imwrite(os.path.join(save_path,oringe_names[i]),img1)


def minmax(image):
    image=image.astype(float)
    aa = np.amin(image)
    bb = np.amax(image)
    img2 = (image - aa) * 255 / (bb - aa)
    return img2

def psnr(img1, img2):
    psnr = skimage.measure.compare_psnr(img1, img2, 255)
    return psnr

def SSIM(img1, img2):
    ssim = skimage.measure.compare_ssim(img1, img2, data_range=255)
    return ssim

def rms(img):
    img_av = np.mean(img)
    img_rms = math.sqrt(np.mean((img - img_av) * (img - img_av))) / img_av
    return img_rms

def coss_correlation(img1,img2):
    sum_ab,sum_aa,sum_bb=0,0,0
    width=img1.shape[0]
    avg_img1=sum(map(sum,img1))/(width*width)
    avg_img2=sum(map(sum,img2))/(width*width)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            ab=(img1[i][j]-avg_img1)*(img2[i][j]-avg_img2)
            aa=(img1[i][j]--avg_img1)*(img1[i][j]-avg_img1)
            bb=(img2[i][j]-avg_img2)*(img2[i][j]-avg_img2)
            sum_ab += ab
            sum_aa+=aa
            sum_bb+= bb
    cc=sum_ab/math.sqrt(sum_aa*sum_bb)
    return cc

def makeMovie():
    path = '/home/vip/zxk/sun image sr/result/wang_sa2_test/'
    filelist = os.listdir(path)
    fps = 12  # 视频每秒24帧
    size = (720,240 )  # 需要转为视频的图片的尺寸

    video = cv.VideoWriter("Video_wang_sa2.avi", cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    # 视频保存在当前目录下

    for item in filelist:
        if item.endswith('.png'):
            # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
            item = path + item
            img = cv.imread(item)
            video.write(img)
    video.release()
    cv.destroyAllWindows()


