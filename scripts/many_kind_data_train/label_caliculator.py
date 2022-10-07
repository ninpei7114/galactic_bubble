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

    def find_cover(self):
        """
        切り出した画像の中に、他のリングが入っていないか確かめる。
        入っていたら、ラベル付けする
        star_listはdictionaryで、中身は、x_pix_min, y_pix_min, x_pix_max, y_pix_maxという順になっている
        """
        width = (self.x_pix_max - self.x_pix_min)/4
        hight = (self.y_pix_max - self.y_pix_min)/4
        
        # g_area = ((x_pix_max-width)-(x_pix_min+width))*((y_pix_max-hight)-(y_pix_min+hight))
        
        self.overlapp_list = []
        self.overlapp_name = []
        for d in self.star_dic.items():
            s_xmin = d[1][0]
            s_xmax = d[1][2]
            s_ymin = d[1][1]
            s_ymax = d[1][3]
            
            xx = np.array([s_xmin, s_xmax])
            yy = np.array([s_ymin, s_ymax])
            c_xx = np.clip(xx, self.x_pix_min+width, self.x_pix_max-width)
            c_yy = np.clip(yy, self.y_pix_min+hight, self.y_pix_max-hight)   
            s_area = (xx[1]-xx[0])*(yy[1]-yy[0])
            c_area = (c_xx[1]-c_xx[0])*(c_yy[1]-c_yy[0])
            
            # 場合分け、全体に対してringが1/2以上入っていないといけない
            # 大きさが画像に対して、1/8以上でないとlabel付けしない
            if (c_area>=s_area*1/4 and (d[1][2]-d[1][0])>=(width*2)/8 and 
                (d[1][3]-d[1][1])>=(hight*2)/10):
                self.overlapp_list.append(d)
                self.overlapp_name.append(d[0])

            else:pass
            
        # return overlapp_list, overlapp_name



    def all_star(self, dataframe):
        """
        データセットのringの範囲をここで決める
        1.5倍で切り出し
        """

        self.star_dic = {}

        for _, row in dataframe.iterrows():    
        
            lmax = row['GLON'] + 1.5*row[self.Rout]/60
            bmin = row['GLAT'] - 1.5*row[self.Rout]/60
            #右端
            lmin = row['GLON'] - 1.5*row[self.Rout]/60
            bmax = row['GLAT'] + 1.5*row[self.Rout]/60
            #これは、リングを切り取る範囲　　切り取る範囲はRoutの3倍
            x_pix_min, y_pix_min = self.world.all_world2pix(lmax, bmin, 0)
            x_pix_max, y_pix_max = self.world.all_world2pix(lmin, bmax, 0)

            self.star_dic[row[self.choice]] = [x_pix_min, y_pix_min, x_pix_max, y_pix_max]
            
        # return star_dic



    def calc_pix(self, row, GLON_min, GLON_max, GLAT_min, GLAT_max, scale):
        """
        切り出す画像の範囲をここで決める

        """
        # import random

        ccc = 0
        ok = True
        
        while ok:
            if self.mode=='train':
                # random_num = 1/np.random.uniform(0.3, 0.89) #サイズが一様ver
                random_num = scale
            else:
                random_num = 1/0.89
            lmax = row['GLON'] + random_num*1.5*row[self.Rout]/60
            bmin = row['GLAT'] - random_num*1.5*row[self.Rout]/60
            #右端
            lmin = row['GLON'] - random_num*1.5*row[self.Rout]/60
            bmax = row['GLAT'] + random_num*1.5*row[self.Rout]/60
            ccc += 1
            if GLON_min<=lmin and lmax<=GLON_max and GLAT_min<=bmin and bmax<=GLAT_max:
                ok = False
                flag = True
            if ccc>=400:
                ok = False
                flag = False
            
        #これは、リングを切り取る範囲　　
        x_min, y_min = self.world.all_world2pix(lmax, bmin, 0)
        x_max, y_max = self.world.all_world2pix(lmin, bmax, 0)
        r = int((x_max - x_min)/(2*random_num))#ringの半径pixel
        
        self.width = x_max - x_min
        self.height = y_max - y_min
        
        self.x_pix_min = x_min - self.width/2
        self.y_pix_min = y_min - self.height/2
        self.x_pix_max = x_max + self.width/2
        self.y_pix_max = y_max + self.height/2
        
        self.width = self.x_pix_max - self.x_pix_min
        self.height = self.y_pix_max - self.y_pix_min
        
        return self.x_pix_min, self.y_pix_min, self.x_pix_max, self.y_pix_max, flag



    def judge_01(self, number):
        if number > 1:
            return 1
        elif number<0:
            return 0
        else:
            return number



    def make_label(self, MWP):
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
        MWP_name_select = MWP.index.tolist()
        #切り出した画像にたまたま入った天体があるか、ないか
        if len(self.overlapp_list) == 0:
            pass
        else:
            
            for p, n in zip(self.overlapp_list, self.overlapp_name):
                # pは、('2G0020120-0068213', [array(7573.50002914), array(4663.19997904), 
                #                           array(7673.50003014), array(4763.19998004)])
                #のように、天体名とpostionが入っている
                if p[0] in MWP_name_select:
                    
                    xmin_c = p[1][0] - (self.x_pix_min+self.width/4)
                    ymin_c = p[1][1] - (self.y_pix_min+self.height/4)
                    xmax_c = p[1][2] - (self.x_pix_min+self.width/4)
                    ymax_c = p[1][3] - (self.y_pix_min+self.height/4)
                    self.xmin_list.append(self.judge_01(xmin_c/(self.width/2)))
                    self.xmax_list.append(self.judge_01(xmax_c/(self.width/2)))
                    self.ymin_list.append(self.judge_01(ymin_c/(self.height/2)))
                    self.ymax_list.append(self.judge_01(ymax_c/(self.height/2)))
                    self.named_list.append(n)
                
        # return xmin_list, ymin_list, xmax_list, ymax_list, named_list


    def check_list(self):
        xmin_list_, ymin_list_, xmax_list_, ymax_list_ = [], [], [], []
        for xy_num in range(len(self.xmin_list)):
            if ((self.xmax_list[xy_num] - self.xmin_list[xy_num])==0 or
                (self.ymax_list[xy_num] - self.ymin_list[xy_num])==0):
                pass
            else:
                xmin_list_.append(self.xmin_list[xy_num])
                ymin_list_.append(self.ymin_list[xy_num])
                xmax_list_.append(self.xmax_list[xy_num])
                ymax_list_.append(self.ymax_list[xy_num])
        
        return xmin_list_, ymin_list_, xmax_list_, ymax_list_, self.named_list, self.star_dic






### MWPカタログを選定する時の関数
#  
# def rank_catalogue(sentei_path, mwp_catalogu_path):
#     nishimoto = pd.read_csv(sentei_path)
#     nishimoto = nishimoto.drop('Unnamed: 0', axis=1)
#     nishimoto = nishimoto.fillna(0)


#     rank1 = []
#     rank2 = []
#     rank3 = []
#     rank4 = []
#     rank5 = []
#     for i in range(len(nishimoto)):
#         nishimoto_s = nishimoto.loc[i]
#         Q = np.where(np.array(nishimoto_s.tolist())==1)[0]
#     #     print(Q)
#         if Q == 0:
#             rank1.append(i)
#         elif Q == 1:
#             rank2.append(i)
#         elif Q == 2:
#             rank3.append(i)
#         elif Q == 3:
#             rank4.append(i)
#         elif Q == 4:
#             rank5.append(i)

    
#     catalogue = pd.read_csv(mwp_catalogu_path)

#     each_rank = []
#     for i in range(len(nishimoto)):
#         nishimoto_s = nishimoto.loc[i]
#         Q = np.where(np.array(nishimoto_s.tolist())==1)[0][0]
#         each_rank.append(Q+1)

#     catalogue['rank'] = each_rank
#     catalogue_ = pd.concat([catalogue.iloc[rank3], catalogue.iloc[rank4], catalogue.iloc[rank5]])
#     catalogue_ = catalogue_.rename(columns={'Unnamed: 0':'MWP'})
#     MWP = catalogue_.set_index('MWP')

#     return MWP

