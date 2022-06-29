# 画像をタイル状に並べるためのコード
# data_view_rectangleは、画像一枚一枚にリングを囲む長方形を描くことを目的としている


from PIL import Image, ImageDraw, ImageFont

def data_view_rectangle(col, imgs, infos=None, moji_size=100):
    '''
    col: number of columns
    imgs: tensor or nparray with a shape of (?, y, x, 1) or (?, y, x, 3)
    infos: dictonary from CutTable
    '''
    imgs = np.uint8(imgs[:, ::-1, :, 0]) if imgs.shape[3] == 1 else np.uint8(imgs[:,::-1])
    row = (lambda x, y: x//y if x/y-x//y==0.0 else x//y+1)(imgs.shape[0], col)
    dst = Image.new('RGB', (imgs.shape[1]*col, imgs.shape[2]*row))

    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', moji_size)
    for i, arr in enumerate(imgs):
        img = Image.fromarray(arr)
        img = img.point(lambda x: x * 1.5)
        if infos is not None:
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), '%s'%infos['id'].tolist()[i], font=font)
            for j in range(len(infos['xmin'].tolist()[i])):
                draw.rectangle((infos['xmin'].tolist()[i][j]*300, (1 - infos['ymax'].tolist()[i][j])*300,
                            infos['xmax'].tolist()[i][j]*300, (1 - infos['ymin'].tolist()[i][j])*300), width=2)

        quo, rem = i//col, i%col
        dst.paste(img, (arr.shape[0]*rem, arr.shape[1]*quo))

    return dst



def data_view(col, imgs, infos=None, moji_size=100):
    '''
    col: number of columns
    imgs: tensor or nparray with a shape of (?, y, x, 1) or (?, y, x, 3)
    infos: dictonary from CutTable
    '''
    imgs = np.uint8(imgs[:, ::-1, :, 0]) if imgs.shape[3] == 1 else np.uint8(imgs[:,::-1])
    row = (lambda x, y: x//y if x/y-x//y==0.0 else x//y+1)(imgs.shape[0], col)
    dst = Image.new('RGB', (imgs.shape[1]*col, imgs.shape[2]*row))
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', moji_size)
    for i, arr in enumerate(imgs):
        img = Image.fromarray(arr)
        img = img.point(lambda x: x * 1.5)
        if infos != None:
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), '%s'%infos[i], font=font)
        quo, rem = i//col, i%col
        dst.paste(img, (arr.shape[0]*rem, arr.shape[1]*quo))

    return dst
