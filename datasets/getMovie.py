# -*- coding: utf-8 -*-
import os
import shutil
import sys
import xlrd


def test():
    ex = xlrd.open_workbook('data_train.xls')
    table = ex.sheet_by_name('Sheet1')
    video_name = []
    video_w =[]
    video_h =[]
    video_frame = []
    video_speed = []
    for i in range(1, 201):
        video_name.append(int(table.cell_value(i, 0)))
        video_w.append(int(table.cell_value(i, 1)))
        video_h.append(int(table.cell_value(i, 2)))
        video_frame.append(int(table.cell_value(i, 3)))
        video_speed.append(int(table.cell_value(i, 4)))

    train_mkv_name = []
    k = 0
    for i in os.listdir("training_raw/"):
        name = i.split('.')[0]
        train_mkv_name.append(name)
        # 001 002 200
        # mvk --> yuv
        os.system("ffmpeg -i training_raw/{}.mkv -pix_fmt yuv420p ffmpeg_test/yuv/{}.yuv".format(name, name))

        # comp yuv --> mkv
        os.system(
            "ffmpeg -pix_fmt yuv420p -s {}x{} -r {} -i ffmpeg_test/yuv/{}.yuv -c:v libx265 -b:v 200k -x265-params pass=1:log-level=error -f null ffmpeg_test/dev/null".format(video_w[k], video_h[k], video_speed[k], name))
        os.system(
            "ffmpeg -pix_fmt yuv420p -s {}x{} -r {} -i ffmpeg_test/yuv/{}.yuv -c:v libx265 -b:v 200k -x265-params pass=2:log-level=error ffmpeg_test/comp/{}.mkv".format(video_w[k], video_h[k], video_speed[k], name, name))

        # split
        if not os.path.exists("ffmpeg_test/input_or/{}".format(name)):
            os.makedirs("ffmpeg_test/input_or/{}".format(name))
        os.system("ffmpeg -i ffmpeg_test/comp/{}.mkv ffmpeg_test/input_or/{}/%3d.png".format(name, name))
        if not os.path.exists("ffmpeg_test/target_or/{}".format(name)):
            os.makedirs("ffmpeg_test/target_or/{}".format(name))
        os.system("ffmpeg -i training_raw/{}.mkv ffmpeg_test/target_or/{}/%3d.png".format(name, name))

        # 滑窗
        k += 1

        print("{} 视频已经处理完毕！".format(i))

def mymovefile(srcfile, dstpath):  # 移动函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.move(srcfile, dstpath + fname)  # 移动文件
        print("move %s -> %s" % (srcfile, dstpath + fname))

def mycopyfile(srcfile,dstpath,name):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + name + ".png")          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + name + ".png"))

def huachuang(what):
    png_sum = []
    for i in os.listdir("ffmpeg_test/{}_or/".format(what)):
        # i = dir: 001 002 003
        png_video = []
        for j in os.listdir("ffmpeg_test/{}_or/".format(what) + i):
            # j = png: 001.png 002.png
            png_video.append("ffmpeg_test/{}_or/".format(what)+i+"/"+j)
            # png_video = list: ffmpeg_test/input/012/005.png ffmpeg_test/input/012/006.png
        png_sum.append(png_video)
        # png_sum = list[]
    video_sum_num = len(png_sum)
    new_video = []
    for video_id, k in enumerate(png_sum):
        # k = png_video : list :
        frame_num = len(k)
        new_this_video = []
        new_ewai_video = []
        for num, mm in enumerate(k):
            # mm = png : ffmpeg_test/input/012/005.png
            new_this_video.append(mm)
            if (num+1) % 100 == 0:
                new_video.append(new_this_video)
                new_this_video = []
            if frame_num % 100 != 0:
                if num > frame_num-101:
                    new_ewai_video.append(mm)
                    if num == frame_num-1:
                        new_video.append(new_ewai_video)

    for kk, video_now in enumerate(new_video):
        input_dir = "enhance/{}/{}/".format(what, str(kk).zfill(3))
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        for jj, png_now in enumerate(video_now):
            mycopyfile(png_now, input_dir, str(jj).zfill(3))






                    # print("ewai: {} , sum: {}".format(mm, frame_num))




def main():
    train_mkv_name = []
    for i in os.listdir("training_raw/"):
        name = i.split('.')[0]
        train_mkv_name.append(name)
        # 001 002 200
        os.system("ffmpeg -i training_raw/{}.mkv -pix_fmt yuv420p training_raw_yuv/{}.yuv".format(name, name))
    os.system("ffmpeg -pix_fmt yuv420p -s (width)x(height) -r (frame_rate) -i xxx.yuv -c:v libx265 -b:v 200k -x265-params pass=1:log-level=error -f null /dev/null")
    os.system("ffmpeg -pix_fmt yuv420p -s (width)x(height) -r (frame_rate) -i xxx.yuv -c:v libx265 -b:v 200k -x265-params pass=2:log-level=error xxx.mkv")

if __name__ == "__main__":
    # print(str(1).zfill(3))
    huachuang(what="target")