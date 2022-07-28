import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('data_path', metavar='DIR', help='path to data')

    return parser.parse_args()


def data_view(col, imgs, infos=None, moji_size=100):

    imgs = np.uint8(imgs[:, ::-1, :, 0]) if imgs.shape[3] == 1 else np.uint8(imgs[:,::-1])
    row = (lambda x, y: x//y if x/y-x//y==0.0 else x//y+1)(imgs.shape[0], col)
    dst = Image.new('RGB', (imgs.shape[1]*col, imgs.shape[2]*row))
    # font = ImageFont.truetype(‘/usr/share/fonts/truetype/freefont/FreeMono.ttf’, moji_size)
    for i, arr in enumerate(imgs):
        img = Image.fromarray(arr)
        img = img.point(lambda x: x * 1.5)
        if infos != None:
            draw = ImageDraw.Draw(img)
            # draw.text((10, 10), ‘%s’%infos[i], font=font)
        quo, rem = i//col, i%col
        dst.paste(img, (arr.shape[0]*rem, arr.shape[1]*quo))
    return dst



def main(args):
    
    data = np.load(args.data_path)
    data = data*255
    data = np.uint8(data)
    data[:,:,:,2]=0
    data_view(100, data).save('data_tile_picture.pdf')



if __name__ == '__main__':
    args = parse_args()
    main(args)