import numpy as np
from colour import colour_correction, sRGB_to_XYZ, XYZ_to_ProLab,ProLab_to_XYZ,XYZ_to_sRGB

   #hald[i]/=3 #img/1=вся сцена занимает диапазон0-1 img/10=вся сцена занимает диапазон 0.45-0.55
def lut_main(q_in_lut,q_out_lut):
    while True:
        
        get_params = q_in_lut.get()
        hald = get_params[0]
        lut_params = get_params[1]
        r_hue = get_params[2]
        r_sut = get_params[3]
        r_g = get_params[4]
        r_b = get_params[5]
        g_hue = get_params[6]
        g_sut = get_params[7]
        g_r = get_params[8]
        g_b = get_params[9]
        b_hue = get_params[10]
        b_sut = get_params[11]
        b_r = get_params[12]
        b_g = get_params[13]
        right = get_params[14]
        film = get_params[15]
        accuracy = get_params[16]
        sut = get_params[17]
        end_a_plus = get_params[18]
        end_a_minus = get_params[19]
        end_b_plus = get_params[20]
        end_b_minus = get_params[21]
        a_min_sut = get_params[22]
        a_plus_sut = get_params[23]
        b_min_sut = get_params[24]
        b_plus_sut = get_params[25]
        a_m_hue = get_params[26]
        a_p_hue = get_params[27]
        b_m_hue = get_params[28]
        b_p_hue = get_params[29]
        sat_comp= get_params[30]
        steepness_shad = get_params[31]
        width_shad = get_params[32]
        steepness_light = get_params[33]
        width_light = get_params[34]
        shad_a = get_params[35]
        shad_b = get_params[36]
        mid_a = get_params[37]
        mid_b = get_params[38]
        light_a = get_params[39]
        light_b = get_params[40]
        neutral_mask = get_params[41]
        amplyfy_wheel = get_params[42]

        

        right-=np.min(right)
        right*=(1/np.max(right))


        params = lut_params[1]
        wrong = lut_params[0]
        wrong-=np.min(wrong)
        wrong*=(1/np.max(wrong))
        for i in hald:
            #                      RGB MIXER
            hald[i][...,0]=hald[i][...,0]*(1+r_sut)+(hald[i][...,1]*(-r_sut/2))+(hald[i][...,2]*(-r_sut/2))
            hald[i][...,0]=hald[i][...,0]+(hald[i][...,1]*r_hue)+(hald[i][...,2]*(-r_hue)) #r+-
            hald[i][...,0]=(hald[i][...,0]*(1-r_g))+(hald[i][...,1]*r_g)
            hald[i][...,0]=(hald[i][...,0]*(1-r_b))+(hald[i][...,2]*r_b)


            hald[i][...,1]=hald[i][...,1]*(1+g_sut)+(hald[i][...,0]*(-g_sut/2))+(hald[i][...,2]*(-g_sut/2))
            hald[i][...,1]=hald[i][...,1]+(hald[i][...,0]*g_hue)+(hald[i][...,2]*(-g_hue)) #r+-
            hald[i][...,1]=(hald[i][...,1]*(1-g_r))+(hald[i][...,0]*g_r)
            hald[i][...,1]=(hald[i][...,1]*(1-g_b))+(hald[i][...,2]*g_b)



            hald[i][...,2]=hald[i][...,2]*(1+b_sut)+(hald[i][...,0]*(-b_sut/2))+(hald[i][...,1]*(-b_sut/2))
            hald[i][...,2]=hald[i][...,2]+(hald[i][...,0]*b_hue)+(hald[i][...,1]*(-b_hue)) #r+-
            hald[i][...,2]=(hald[i][...,2]*(1-b_r))+(hald[i][...,0]*b_r)
            hald[i][...,2]=(hald[i][...,2]*(1-b_g))+(hald[i][...,1]*b_g)





            poli=np.polynomial.polynomial.Polynomial.fit(params["lut_curve_in"][:,1],params["lut_curve_out"][:,1],deg=params["degr"])
            for z in range(3):
                hald[i][...,z] = poli(hald[i][...,z])

            """hald[i][...,0]=hald[i][...,0]*params["r_r"]+hald[i][...,1]*params["r_g"]+hald[i][...,2]*params["r_b"]
            hald[i][...,1]=hald[i][...,0]*params["g_r"]+hald[i][...,1]*params["g_g"]+hald[i][...,2]*params["g_b"]
            hald[i][...,2]=hald[i][...,0]*params["b_r"]+hald[i][...,1]*params["b_g"]+hald[i][...,2]*params["b_b"]"""


            acc=0
            if accuracy=="High":
                    acc=22

            else:
                    acc=35

            if film=="sRgb":
                acc=20
            
            right = np.reshape(right,(456,3))

            hald[i][hald[i]<0]=0
            hald[i][hald[i]>1]=1

            wrong[wrong>1]=1
            wrong[wrong<0]=0

            hald[i]=colour_correction(hald[i],wrong,right,method='Cheung 2004', degree=4,root_polynomial_expansion=True,terms=16)
            wrong=colour_correction(wrong,wrong,right,method='Cheung 2004', degree=4,root_polynomial_expansion=True,terms=16)

            hald[i]=colour_correction(hald[i],wrong,right,method='Cheung 2004', degree=4,root_polynomial_expansion=True,terms=16)
            #hald[i]=XYZ_to_sRGB(hald[i])

            hald[i][hald[i]<0]=0
            hald[i][hald[i]>1]=1

            hald[i]=np.array(hald[i],dtype=np.float32)
            
            #wrong[wrong>1]=1
            #wrong[wrong<0]=0

            hald[i]=sRGB_to_XYZ(hald[i])
            hald[i]=XYZ_to_ProLab(hald[i])    






            #             PROLAB PARAMS

            hald[i][...,1]=(hald[i][...,1]*sut*2)
            hald[i][...,2]=(hald[i][...,2]*sut*2)

            hald[i][...,1][hald[i][...,1]<0]=hald[i][...,1][hald[i][...,1]<0]*a_min_sut
            hald[i][...,1][hald[i][...,1]<0]=((2/(1+(np.power((16**(end_a_minus*sat_comp)),-(((hald[i][...,1][hald[i][...,1]<0])))/100))))-1)*(31/(end_a_minus*sat_comp))#*mask[:,:,1][np.where(hald[i][...,1]<0)]     #a+ ибо обратились к 2 каналу где этот канал больше нуля

            #hald[i][...,0]+=a_min_L*(-a_min_mask/(mask_compress*100))

            hald[i][...,1][hald[i][...,1]>=0]=hald[i][...,1][hald[i][...,1]>=0]*a_plus_sut
            hald[i][...,1][hald[i][...,1]>=0]=((2/(1+(np.power((16**(end_a_plus*sat_comp)),-((hald[i][...,1][hald[i][...,1]>=0]))/100))))-1)*(31/(end_a_plus*sat_comp))#*mask[:,:,1][np.where(hald[i][...,1]>=0)]

            #hald[i][...,0]+=a_plus_L*a_plus_mask/(mask_compress*100)

            hald[i][...,2][hald[i][...,2]<0]=hald[i][...,2][hald[i][...,2]<0]*b_min_sut
            hald[i][...,2][hald[i][...,2]<0]=((2/(1+(np.power((16**(end_b_minus*sat_comp)),-((hald[i][...,2][hald[i][...,2]<0]))/100))))-1)*(31/(end_b_minus*sat_comp))#*mask[:,:,2][np.where(hald[i][...,2]<0)]

            #hald[i][...,0]+=b_min_mask*(-b_min_L)/(mask_compress*100)

            hald[i][...,2][hald[i][...,2]>=0]=hald[i][...,2][hald[i][...,2]>=0]*b_plus_sut
            hald[i][...,2][hald[i][...,2]>=0]=((2/(1+(np.power((16**(end_b_plus*sat_comp)),-((hald[i][...,2][hald[i][...,2]>=0]))/100))))-1)*(31/(end_b_plus*sat_comp))#*mask[:,:,2][np.where(hald[i][...,2]>=0)]                 #((2/1+16**yx)-1)*100/y'''

            #hald[i][...,0]+=b_plus_L*b_plus_mask/(mask_compress*100)
            
            #HUE MiX
            a_min=np.array(hald[i][...,1])
            a_plus=np.array(hald[i][...,1])
            b_min=np.array(hald[i][...,2])
            b_plus=np.array(hald[i][...,2])

            a_min=a_min*(1-a_m_hue)+hald[i][...,2]*a_m_hue
            a_plus=a_plus*(1-a_p_hue)+hald[i][...,2]*a_p_hue
            b_min=b_min*(1-b_m_hue)+hald[i][...,1]*b_m_hue
            b_plus=b_plus*(1-b_p_hue)+hald[i][...,1]*b_p_hue

            a_min[a_min>0]=0
            a_plus[a_plus<0]=0
            b_min[b_min>0]=0
            b_plus[b_plus<0]=0

            a=a_min+a_plus
            b=b_min+b_plus

            hald[i][...,1]=a
            hald[i][...,2]=b
            hald[i]=np.nan_to_num(hald[i])



                #     MASK FOR WHEELS
            wheel_sut_mask=np.fabs(np.array(hald[i][...,1]+hald[i][...,2]))
            wheel_sut_mask/=np.max(wheel_sut_mask)
            wheel_sut_mask=np.stack((wheel_sut_mask,wheel_sut_mask,wheel_sut_mask),axis=-1)


            shad_mask=np.array(hald[i][...,0])/100
            mid_mask=np.array(hald[i][...,0])/100
            mid_mask=(mid_mask*0)+1
            light_mask=np.array(hald[i][...,0])/100
            def curve(img,steepness,width,apex):
                img=(2/(1+np.power(10**steepness,(-width-apex+img))))*(2/(1+np.power(10**steepness,(-width+apex-img))))
                return img

            shad_mask=curve(shad_mask,steepness_shad,width_shad,0.0)
            shad_mask*=(1/np.max(shad_mask))
            light_mask=curve(light_mask,steepness_light,width_light,1.)
            light_mask*=(1/np.max(light_mask))
            mid_mask=(mid_mask-shad_mask-light_mask)
            #mid_mask[mid_mask<0]=0
            shad_mask=np.stack((shad_mask,shad_mask,shad_mask),axis=-1)
            light_mask=np.stack((light_mask,light_mask,light_mask),axis=-1)
            mid_mask=np.stack((mid_mask,mid_mask,mid_mask),axis=-1)

            shad_mask=np.nan_to_num(shad_mask)
            light_mask=np.nan_to_num(light_mask)
            mid_mask=np.nan_to_num(mid_mask)




            #      WHEELS
            shad=np.array(hald[i])
            mid=np.array(hald[i])
            light=np.array(hald[i])

            shad[...,1]+=shad_a*(((1-neutral_mask)+1)**2)*amplyfy_wheel
            shad[...,2]+=shad_b*(((1-neutral_mask)+1)**2)*amplyfy_wheel
            mid[...,1]+=mid_a*(((1-neutral_mask)+1)**2)*amplyfy_wheel
            mid[...,2]+=mid_b*(((1-neutral_mask)+1)**2)*amplyfy_wheel
            light[...,1]+=light_a*(((1-neutral_mask)+1)**2)*amplyfy_wheel
            light[...,2]+=light_b*(((1-neutral_mask)+1)**2)*amplyfy_wheel

            shad*=shad_mask
            mid*=mid_mask
            light*=light_mask
            wheel_sut_mask=(wheel_sut_mask+10**(1-neutral_mask))/(10**(1-neutral_mask)+1)
            after_wheel=shad+mid+light
            after_wheel*=(1-wheel_sut_mask)
            hald[i]*=wheel_sut_mask
            hald[i]=hald[i]+after_wheel





            to_XYZ=ProLab_to_XYZ(hald[i])
            to_XYZ[to_XYZ>1]=1
            to_XYZ[to_XYZ<0]=0
            hald[i]=XYZ_to_sRGB(to_XYZ)

            #hald[i]=(hald[i]-0.2-Exp_shift/9)*3
            hald[i][hald[i]>1]=1
            hald[i][hald[i]<0]=0
            hald[i]=np.array(hald[i],dtype=np.float32)

            q_out_lut.put(hald)
            q_in_lut.task_done()