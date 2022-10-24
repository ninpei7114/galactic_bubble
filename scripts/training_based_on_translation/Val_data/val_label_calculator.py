import numpy as np

class label_caliculator(object):

    def __init__(self, choice, mode, world):

        if  choice == 'MWP':
            self.Rout = 'Reff'
        else:
            self.Rout = 'Rout'

        self.mode = mode
        self.world = world
        self.choice = choice


    def find_cover(self, star_dic, x_pix_min, y_pix_min, x_pix_max, y_pix_max):
        """
        切り出した画像の中に、他のリングが入っていないか確かめる。
        入っていたら、ラベル付けする
        star_listはdictionaryで、中身は、x_pix_min, y_pix_min, x_pix_max, y_pix_maxという順になっている
        """
        width = (x_pix_max - x_pix_min)/4
        hight = (y_pix_max - y_pix_min)/4
        
        # g_area = ((x_pix_max-width)-(x_pix_min+width))*((y_pix_max-hight)-(y_pix_min+hight))
        
        overlapp_list = []
        overlapp_name = []
        for d in star_dic.items():
            s_xmin = d[1][0]
            s_xmax = d[1][2]
            s_ymin = d[1][1]
            s_ymax = d[1][3]
            
            xx = np.array([s_xmin, s_xmax])
            yy = np.array([s_ymin, s_ymax])
            c_xx = np.clip(xx, x_pix_min+width, x_pix_max-width)
            c_yy = np.clip(yy, y_pix_min+hight, y_pix_max-hight)   
            s_width =  c_xx[1]-c_xx[0]
            s_height = c_yy[1]-c_yy[0]
            s_area = (xx[1]-xx[0])*(yy[1]-yy[0])
            c_area = (c_xx[1]-c_xx[0])*(c_yy[1]-c_yy[0])
            
            # 場合分け、全体に対してringが1/2以上入っていないといけない
            # 大きさが画像に対して、1/8以上でないとlabel付けしない
            if (c_area>=s_area*1/4 and s_height/(s_width + 1e-9)>1/3 and s_width/(s_height + 1e-9)>1/3):
                overlapp_list.append(d)
                overlapp_name.append(d[0])

            else:pass
            
        return overlapp_list, overlapp_name



    def all_star(self, dataframe):
        """
        データセットのringの範囲をここで決める
        1.5倍で切り出し
        """

        self.star_dic = {}

        for _, row in dataframe.iterrows():    
        
            lmax = row['GLON'] + row[self.Rout]/60
            bmin = row['GLAT'] - row[self.Rout]/60
            #右端
            lmin = row['GLON'] - row[self.Rout]/60
            bmax = row['GLAT'] + row[self.Rout]/60
            #これは、リングを切り取る範囲　　切り取る範囲はRoutの3倍
            x_pix_min, y_pix_min = self.world.all_world2pix(lmax, bmin, 0)
            x_pix_max, y_pix_max = self.world.all_world2pix(lmin, bmax, 0)

            self.star_dic[row[self.choice]] = [x_pix_min, y_pix_min, x_pix_max, y_pix_max]
            
        return self.star_dic


    def judge_01(self, number):
        if number > 1:
            return 1
        elif number<0:
            return 0
        else:
            return number



    def make_label(self, x_pix_min, y_pix_min, x_pix_max, y_pix_max, cover_star_position, cover_star_name,
                   width, height, Ring_CATA):
        """
        sは、主体となるringの位置情報
        x_pix_min, y_pix_min,x_pix_max, y_pix_maxは、切り出す画像のサイズ
        主体となるringに重なっているringのindex情報、重なったringの情報はstar_listの中にある。
        """

        self.xmin_list = []
        self.ymin_list = []
        self.xmax_list = []
        self.ymax_list = []
        self.named_list = []
        MWP_name_select = Ring_CATA.index.tolist()
        #切り出した画像にたまたま入った天体があるか、ないか
        if len(cover_star_position) == 0:
            pass
        else:
            
            for p, n in zip(cover_star_position, cover_star_name):
                # pは、('2G0020120-0068213', [array(7573.50002914), array(4663.19997904), 
                #                           array(7673.50003014), array(4763.19998004)])
                #のように、天体名とpostionが入っている
                if p[0] in MWP_name_select:
                    
                    xmin_c = p[1][0] - (x_pix_min+width/4)
                    ymin_c = p[1][1] - (y_pix_min+height/4)
                    xmax_c = p[1][2] - (x_pix_min+width/4)
                    ymax_c = p[1][3] - (y_pix_min+height/4)
                    self.xmin_list.append(self.judge_01(xmin_c/(width/2)))
                    self.xmax_list.append(self.judge_01(xmax_c/(width/2)))
                    self.ymin_list.append(self.judge_01(ymin_c/(height/2)))
                    self.ymax_list.append(self.judge_01(ymax_c/(height/2)))
                    self.named_list.append(n)
                
        # return self.xmin_list, self.ymin_list, self.xmax_list, self.ymax_list, self.named_list


    def check_list(self):
        xmin_list_, ymin_list_, xmax_list_, ymax_list_, name_list_ = [], [], [], [], []
        for xy_num in range(len(self.xmin_list)):
            if ((self.xmax_list[xy_num] - self.xmin_list[xy_num])==0 or
                (self.ymax_list[xy_num] - self.ymin_list[xy_num])==0):
                pass
            else:
                xmin_list_.append(self.xmin_list[xy_num])
                ymin_list_.append(self.ymin_list[xy_num])
                xmax_list_.append(self.xmax_list[xy_num])
                ymax_list_.append(self.ymax_list[xy_num])
                name_list_.append(self.named_list[xy_num])
        
        return xmin_list_, ymin_list_, xmax_list_, ymax_list_, name_list_


