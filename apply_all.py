import numpy as np
import itertools
from colour import table_interpolation






def apply_film(q_in_film,q_out_film):
    while True:
        get_params = q_in_film.get()

        in_img = get_params[0]
        Lut = get_params[1]
        Wb_b = get_params[2]
        Wb_r = get_params[3]
        Wb_b2 = get_params[4]
        Wb_r2 = get_params[5]
        print_exp = get_params[6]
        light_compr = get_params[7]
        zone = get_params[8]
        print_cont = get_params[9]
        gamma = get_params[10]

        grain_curve = get_params[11]
        prep_grain = get_params[12]
        amplify = get_params[13]
        amplify_mask = get_params[14]
        on_grain = get_params[15]
        paper = get_params[16]
        def my_interpolation_trilinear(
            V_xyz,table):
            V_xyz = np.clip(V_xyz, 0, 1)

            V_xyz2 = np.reshape(V_xyz, (-1, 3))
            i_m = np.array(table.shape[0:-1],np.uint8) - 1
            i_f = np.floor(V_xyz2 * i_m)
            i_f = np.array(i_f,np.uint8)
            i_f = np.clip(i_f, 0, i_m)
            i_c = np.clip(i_f + 1, 0, i_m)
            i_f_c = i_f, i_c
            vertices = np.array(
                [
                    table[
                        i_f_c[i[0]][..., 0], i_f_c[i[1]][..., 1], i_f_c[i[2]][..., 2]
                    ]
                    for i in itertools.product(*zip([0, 0, 0], [1, 1, 1]))
                ]
            )


            # Relative to indexes ``V_xyz`` values.
            V_xyzr = i_m * V_xyz2 - i_f


            vertices = np.moveaxis(vertices, 0, 1)

            x, y, z = (f[:, None] for f in np.transpose(V_xyzr,np.concatenate([[V_xyzr.ndim - 1], np.arange(0, V_xyzr.ndim - 1)]),))
            weights = np.moveaxis(
                np.transpose(
                    [
                        (1 - x) * (1 - y) * (1 - z),
                        (1 - x) * (1 - y) * z,
                        (1 - x) * y * (1 - z),
                        (1 - x) * y * z,
                        x * (1 - y) * (1 - z),
                        x * (1 - y) * z,
                        x * y * (1 - z),
                        x * y * z,
                    ]
                ),
                0,
                -1,
            )

            xyz_o = np.reshape(np.sum(vertices * weights, 1), V_xyz.shape)
            return xyz_o



        for z in in_img:
            

            in_img[z][...,0]-=Wb_b/90
            in_img[z][...,1]-=((Wb_b/90)+(Wb_r/90))
            in_img[z][...,2]-=Wb_r/90

            #in_img[z]=(((in_img[z]-np.min(in_img[z]))/(np.max(in_img[z])-np.min(in_img[z]))))
            in_img[z]=my_interpolation_trilinear(in_img[z],table=Lut)
            #in_img[z]=table_interpolation(in_img[z],table=Lut,method='Tetrahedral')
            #in_img[z] = np.array(in_img[z],dtype=np.float32)
            

            #table=
            """in_img[z][...,0]-=((Wb_b2/90)+(Wb_r2/90))
            in_img[z][...,1]-=Wb_b2/90
            in_img[z][...,2]-=Wb_r2/90"""
            in_img[z]=(in_img[z]-0.6)+(print_exp/9)+(print_cont/16)
            if gamma<=1:
                in_img[z]-=(gamma/9)
            elif gamma>1:
                in_img[z]-=((gamma**0.66)/9)
            #in_img[z]=(in_img[z]-0.6)-(gamma/9)+(print_cont/16)


            #in_img[z]-=((gamma-3.1)/9)
            #in_img[z][in_img[z]<-0.6]=-0.75
            #gamma_coef=(1.5**-print_exp/16)*10

            '''                         CONTRAST                    '''
            #if light_compr>=0:
            #    light_compr = np.power((6-light_compr),3)
            #elif light_compr<0:
            #    light_compr = np.power((-6-light_compr),3)
            #in_img[z]+=zone
            #in_img[z][in_img[z]>0]=(((light_compr)/(-in_img[z][in_img[z]>0]-(light_compr)))+1)*(light_compr)
            #in_img[z]-=zone

            
            #img_contrast=1/(1+(np.power(55,(-in_img[z]))))
            in_img[z]=1/(1+(np.power((10**(print_cont*3)),(-in_img[z]))))

            #in_img[z]=my_interpolation_trilinear(in_img[z],table=paper)

            if on_grain:
                Mask_gr=grain_curve.apply(in_img[z])
                grain=((prep_grain[z]*amplify*(Mask_gr**amplify_mask)))/(amplify_mask**1.5)+1
                grain=np.array(grain,dtype=np.float32)
                in_img[z]=in_img[z]*grain
                in_img[z][in_img[z]>=1]=1
                in_img[z][in_img[z]<=0]=0
    #grained=(grained+np.fabs(np.min(grained)))*(1/np.max(grained+np.fabs(np.min(grained))))

            

            #img_contrast+=0.16
            #img_contrast-=np.min(img_contrast)
            #img_contrast*=(1/np.max(img_contrast))

            

            #img_contrast-=((gamma-3.1)/24)
        q_out_film.put(in_img)
        q_in_film.task_done()


