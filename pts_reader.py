import os
import re
import struct
import shutil


def get_files(dname, suffix):
    pts_list = []
    for fname in os.listdir(dname):
        if fname.endswith(suffix):
            pts_list += [fname]
    return pts_list


def pts2txt(din, dout, src):
    src_p = os.path.join(din, src)
    data = open(src_p, 'rb').read()
    points = struct.unpack('i172f', data)
    # print points

    dst = src.lower()
    dst = dst.replace('pts', 'txt')
    dst_p = os.path.join(dout, dst)
    # print dst_p

    fout = open(dst_p, 'w')
    pnum = len(points[1:])
    for i in range(1, pnum, 2):
        fout.write('%f ' % points[i])
        fout.write('%f\n' % points[i + 1])

    fout.close()


def main():
    src = 'c:\\Users\\u\\Downloads\\SCUT-FBP5500_v2.1\\SCUT-FBP5500_v2\\facial landmark'
    dst = 'c:\\Users\\u\\Downloads\\SCUT-FBP5500_v2.1\\SCUT-FBP5500_v2\\facial landmark\\cc_86'

    if not os.path.exists(dst):
        os.mkdir(dst)

    pts_list = get_files(src, 'pts')
    for pts in pts_list:
        pts2txt(src, dst, pts)

    jpg_list = get_files(src, 'jpg')
    for img in jpg_list:
        src_img = os.path.join(src, img)
        img_lower = img.lower()
        dst_img = os.path.join(dst, img_lower)
        shutil.copy(src_img, dst_img)


if __name__ == "__main__":
    main()
