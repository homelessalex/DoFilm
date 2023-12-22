import numpy as np
from PIL import Image
from PIL.Image import Resampling
from colour import io,sRGB_to_XYZ,XYZ_to_ProLab,LUT3D,colour_correction,ProLab_to_XYZ,XYZ_to_sRGB,LUT1D, XYZ_to_RGB
import rawpy
from scipy import ndimage,signal
import itertools
from streamlit import cache_data

import time
from copy import deepcopy
'''
def imoprt_const():
    dir_path = Path.cwd()
    print(dir_path)
    Film_curve=read_LUT(Path(dir_path,'Do_film_app','data','film','portra400','film_curve_porttra_400.spi1d'))
    Grain_curve=read_LUT(Path(dir_path,'Do_film_app','data','grain','Grain_curve.spi1d'))
    Gr_sample=Image.open(Path(dir_path,'Do_film_app','data','grain','grain_portra400.tif'))

    wrong=read_image(Path(dir_path,'Do_film app','data','cum', 'sigma_fp.tif'),'uint16',"ImageIO")
    wrong=io.convert_bit_depth(wrong, "float32")
    wrong=sRGB_to_XYZ(wrong)
    wrong=XYZ_to_ProLab(wrong)
    wrong=((np.reshape(wrong, (216,3))).tolist())
    right=read_image('/Users/aleksejromadin/Desktop/portra scans/fitted/Portra_400_colour.tif','uint16',"ImageIO")
    right=io.convert_bit_depth(right, "float32")
    right=sRGB_to_XYZ(right)
    right=XYZ_to_ProLab(right)
    right=((np.reshape(right, (216,3))).tolist())

    img_in="/Users/aleksejromadin/Desktop/IMG_4799.cr3"
    
   
    return Film_curve,wrong,right,img_in,Grain_curve,Gr_sample,'''

'''IMPORT'''
'''constants=imoprt_const()
Film_curve=constants[0]
wrong=constants[1]
right=constants[2]
img_in=constants[3]
gr_curve=constants[4]
gr_sample=constants[5]'''
'''PARAMS'''
'''resolution=1500

blur_rad=2.5
blur_spred=6
halation=1
gamma=0.7
               #в стопах экспозиции 1=вся сцена в один стоп 9=сцена занимает весь диапазон пленки
exp_shift=-1               #перемещаем сцену вверх/вниз по диапазону пленки
WB_r=-2.2
WB_b=0.5
sut=1.2                    #Насыщенность
print_exp=0
print_cont=1.5


end_a_plus=4
end_a_minus=3
end_b_plus=4
end_b_minus=3'''
@cache_data()
def raw(raw_in) -> np.array:
    with rawpy.imread(raw_in) as raw:
            rgb = raw.postprocess(output_color=rawpy.ColorSpace(0),  demosaic_algorithm=rawpy.DemosaicAlgorithm(3), half_size=False,
                                use_camera_wb=True, highlight_mode=rawpy.HighlightMode(2),#user_wb=(1,1,1,1),
                                output_bps=16,  no_auto_scale=False, auto_bright_thr=0.00001,
                                gamma=(1,1), chromatic_aberration=(1,1),)
    rgb=np.dstack((rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]))
    
    return rgb
@cache_data()
def zoom(rgb,resolution):
    img=ndimage.zoom(rgb,(float(resolution/float(np.max(np.shape(rgb)))),float(resolution/float(np.max(np.shape(rgb)))),1))
    add=np.zeros((200,int(img.shape[1]),3),dtype=np.float32)
    for i in range(img.shape[1]):
        add[:,i,:]=img[int(img.shape[0]-10),i,:]
    img=np.vstack((img,add))
    for i in range(img.shape[1]):
        add[:,i,:]=img[10,i,:]
    img=np.vstack((add,img))
    add2=np.zeros((int(img.shape[0]),200,3),dtype=np.float32)

    for i in range(img.shape[0]):
        add2[i,:,:]=img[i,int(img.shape[1]-10),:]
    img=np.hstack((img,add2))
    for i in range(img.shape[0]):
        add2[i,:,:]=img[i,10,:]
    img=np.hstack((add2,img))

    
    return img
#rgb=raw_zoom(img_in,resolution)


def pre_grain(image,gr_sample) -> np.array:
    image=image[200:int(image.shape[0]-200),200:int(image.shape[1]-200),:]
    if np.shape(image)[0]>np.shape(image)[1]:
        gr_sample=gr_sample.transpose(method=Image.Transpose.ROTATE_90)

    Grain_crop=gr_sample.resize((int(np.shape(image)[1]),int(np.shape(image)[0])),resample=Image.LANCZOS, )
    grain=np.array(Grain_crop,dtype=np.uint8)
    grain=io.convert_bit_depth(grain,"float64")
    grain_0=(grain-np.average(grain))
    return grain_0
#prep_grain=pre_grain(rgb,gr_sample)

@cache_data()
def reduse_sharpness (input,Blur_rad,Halation,Blur_spred) -> np.array:
    
    def blur(size,k):
        #size=10 #радиус размытия
        #k=50  #коэффицент влияет на скорость угасания-чем выше тем быстрее
        x = np.fabs(np.linspace(-(size ), size , size*2+1)/size)#создаем последовательность 1...-0-...1
        x=1-x#гипербола отсюдова:https://habr.com/ru/articles/432622/ 
        x = np.outer(x, x)
        x=1-x
        kernel_2D=(((x-1)**2)*(x*k+k+1))/((k+1)*(k*x+1))  #превращаем его в 2d np.outer(kernel_1D.T, kernel_1D.T) 
        suum=np.sum(kernel_2D)   #сумируем матрицу
        kernel_2D=kernel_2D/suum   #делим нашу матрицу на сумму для приведения в дщиапазон 0-1
        return kernel_2D
    blur_coef=float(np.max(np.shape(input))/1500)
    rgb_blur=np.dstack((
        signal.oaconvolve(input[:,:,0],blur(int(3*Blur_rad*Halation*blur_coef),(2**Blur_spred)*(blur_coef**2)),'same',None),
        signal.oaconvolve(input[:,:,1],blur(int(3*Blur_rad*blur_coef),(2**Blur_spred)*(blur_coef**2)),'same',None),
        signal.oaconvolve(input[:,:,2],blur(int(np.around((1/Halation)*3*Blur_rad*blur_coef)),(2**Blur_spred)*(blur_coef**2)),'same',None)
        ))
    
    rgb_blur=np.array(rgb_blur,dtype=np.float32)
    
    return rgb_blur

#img=blur(rgb, blur_rad, halation, blur_spred)
@cache_data()
def bloom (input,bloom_rad,bloom_Halation,bloom_spred) -> np.array:
    
    def blur(size,k):
        #size=10 #радиус размытия
        #k=50  #коэффицент влияет на скорость угасания-чем выше тем быстрее
        x = np.fabs(np.linspace(-(size ), size , size*2+1)/size)#создаем последовательность 1...-0-...1
        x=1-x#гипербола отсюдова:https://habr.com/ru/articles/432622/ 
        x = np.outer(x, x)
        x=1-x
        kernel_2D=(((x-1)**2)*(x*k+k+1))/((k+1)*(k*x+1))  #превращаем его в 2d np.outer(kernel_1D.T, kernel_1D.T) 
        suum=np.sum(kernel_2D)   #сумируем матрицу
        kernel_2D=kernel_2D/suum   #делим нашу матрицу на сумму для приведения в дщиапазон 0-1
        return kernel_2D
    blur_coef=float((np.max(np.shape(input))/1500))
    
    
    rgb_blur=np.dstack((
        signal.oaconvolve(input[:,:,0],blur(int(3*bloom_rad*bloom_Halation*blur_coef),(2**bloom_spred)*(blur_coef**2)),'same',None),
        signal.oaconvolve(input[:,:,1],blur(int(3*bloom_rad*blur_coef),(2**bloom_spred)*(blur_coef**2)),'same',None),
        signal.oaconvolve(input[:,:,2],blur(int(np.around((1/bloom_Halation)*3*bloom_rad*blur_coef)),(2**bloom_spred)*(blur_coef**2)),'same',None)
        ))
    
    rgb_blur=np.array(rgb_blur,dtype=np.float32)
    return rgb_blur
@cache_data()
def sharpen (input,Sharp_rad,Sharp_spread,amplyfy):
    
        
    def blur(size,k):
        #size=10 #радиус размытия
        #k=50  #коэффицент влияет на скорость угасания-чем выше тем быстрее
        x = np.fabs(np.linspace(-(size ), size , size*2+1)/size)#создаем последовательность 1...-0-...1
        x=1-x#гипербола отсюдова:https://habr.com/ru/articles/432622/ 
        x = np.outer(x, x)
        x=1-x
        kernel_2D=(((x-1)**2)*(x*k+k+1))/((k+1)*(k*x+1))  #превращаем его в 2d np.outer(kernel_1D.T, kernel_1D.T) 
        suum=np.sum(kernel_2D)   #сумируем матрицу
        kernel_2D=kernel_2D/suum   #делим нашу матрицу на сумму для приведения в дщиапазон 0-1
        return kernel_2D
    blur_coef=float(np.max(np.shape(input))/1500)
    rgb_blured=np.dstack((
        signal.fftconvolve(input[:,:,0],blur(int(3*Sharp_rad*blur_coef),(2**Sharp_spread)*(blur_coef**2)),'same',None),
        signal.fftconvolve(input[:,:,1],blur(int(3*Sharp_rad*blur_coef),(2**Sharp_spread)*(blur_coef**2)),'same',None),
        signal.fftconvolve(input[:,:,2],blur(int(3*Sharp_rad*blur_coef),(2**Sharp_spread)*(blur_coef**2)),'same',None)
        ))
    
    rgb_blured=np.array(rgb_blured,dtype=np.float32)
        
    blured=(input-rgb_blured)*(amplyfy/100)
    input+=blured
    input[input>=1]=1
    input[input<=0]=0
    input=input[200:int(input.shape[0]-200),200:int(input.shape[1]-200),:]
    return input
@cache_data
def rgb_mix( r_hue,
            r_sut,
            r_g,
            r_b,

            g_hue,
            g_sut,
            g_r,
            g_b,

            b_hue,
            b_sut,
            b_r,
            b_g
            ):
    hald=LUT3D.linear_table(64)
    #hald/=3 #img/1=вся сцена занимает диапазон0-1 img/10=вся сцена занимает диапазон 0.45-0.55
    
    #                      RGB MIXER
    hald[:,:,:,0]=hald[:,:,:,0]*(1+r_sut)+(hald[:,:,:,1]*(-r_sut/2))+(hald[:,:,:,2]*(-r_sut/2))
    hald[:,:,:,0]=hald[:,:,:,0]+(hald[:,:,:,1]*r_hue)+(hald[:,:,:,2]*(-r_hue)) #r+-
    hald[:,:,:,0]=(hald[:,:,:,0]*(1-r_g))+(hald[:,:,:,1]*r_g)
    hald[:,:,:,0]=(hald[:,:,:,0]*(1-r_b))+(hald[:,:,:,2]*r_b)


    hald[:,:,:,1]=hald[:,:,:,1]*(1+g_sut)+(hald[:,:,:,0]*(-g_sut/2))+(hald[:,:,:,2]*(-g_sut/2))
    hald[:,:,:,1]=hald[:,:,:,1]+(hald[:,:,:,0]*g_hue)+(hald[:,:,:,2]*(-g_hue)) #r+-
    hald[:,:,:,1]=(hald[:,:,:,1]*(1-g_r))+(hald[:,:,:,0]*g_r)
    hald[:,:,:,1]=(hald[:,:,:,1]*(1-g_b))+(hald[:,:,:,2]*g_b)
    
    
    
    hald[:,:,:,2]=hald[:,:,:,2]*(1+b_sut)+(hald[:,:,:,0]*(-b_sut/2))+(hald[:,:,:,1]*(-b_sut/2))
    hald[:,:,:,2]=hald[:,:,:,2]+(hald[:,:,:,0]*b_hue)+(hald[:,:,:,1]*(-b_hue)) #r+-
    hald[:,:,:,2]=(hald[:,:,:,2]*(1-b_r))+(hald[:,:,:,0]*b_r)
    hald[:,:,:,2]=(hald[:,:,:,2]*(1-b_g))+(hald[:,:,:,1]*b_g)
    #hald+=((Exp_shift/9))
    return hald
@cache_data
def apply_film(hald,
            wrong,
            right,
            accuracy

            ) -> np.array:
    
    def ntrls(sourse_tif):
        ntrl_r=np.zeros((1,6,1))
        
    
        for i in range(1,15):
            x=(i*4)-1
            y=sourse_tif[x,:,1]
            y=y.reshape((1,6,1))
            ntrl_r=np.vstack((ntrl_r,y))
        ntrl_r=ntrl_r[1:15,:,:]
        ntrl_r=ntrl_r.reshape((84,)).tolist()
        ntrl_r=np.sort(ntrl_r)
        
        return ntrl_r
    add_0_1=np.array([[0,0,0],[1,1,1]])
    add_0_1=add_0_1.reshape((2,3))
    right=np.nan_to_num(right)
    right_ntrl=ntrls(right)
    right=np.reshape(right, (336,3))
    wrong=np.nan_to_num(wrong)
    wrong_ntrl=ntrls(wrong)
    print(wrong.shape)
    wrong=np.reshape(wrong, (336,3))
    poli=np.polynomial.polynomial.Polynomial.fit(wrong_ntrl,right_ntrl,deg=9)
    film_curve=poli(LUT1D.linear_table(64))
    film_curve=LUT1D(film_curve)
    wrong[:,0]=film_curve.apply(wrong[:,0])
    wrong[:,1]=film_curve.apply(wrong[:,1])
    wrong[:,2]=film_curve.apply(wrong[:,2])

    hald[:,:,:,0]=film_curve.apply(hald[:,:,:,0])
    hald[:,:,:,1]=film_curve.apply(hald[:,:,:,1])
    hald[:,:,:,2]=film_curve.apply(hald[:,:,:,2])


    right=sRGB_to_XYZ(right)
    right=XYZ_to_ProLab(right)
    wrong=sRGB_to_XYZ(wrong)
    wrong=XYZ_to_ProLab(wrong)
    hald=sRGB_to_XYZ(hald)
    hald=XYZ_to_ProLab(hald)
    

    hald=colour_correction(hald, wrong, right, method='Vandermonde', #'Vandermonde', #'Cheung 2004', # 'Finlayson 2015',# #,##, ###'Finlayson 2015',#   #оптимально вандермонд 1гр для скана adobe 1.0 in rgb
        degree=accuracy,
        root_polynomial_expansion=True)                             #vandermond 5deg сработало с с200

    hald[hald[:,:,:,0]<0]=0
    return hald

@cache_data
def ProLab(after_lut,            
           
            sut,
            

            end_a_plus,
            end_a_minus,
            end_b_plus,
            end_b_minus,
            

            a_min_sut,
            a_plus_sut,
            b_min_sut,
            b_plus_sut,


            mask_compress,

            a_m_hue,a_p_hue,b_m_hue,b_p_hue
                            ):
    # MAKE MASKS A/B CHAN
    a_plus_mask=(np.array(2/(1+np.power(10**mask_compress,-after_lut[:,:,:,1])))-1)*100
    a_min_mask=(np.array(2/(1+np.power(10**mask_compress,-after_lut[:,:,:,1])))-1)*100
    b_plus_mask=(np.array(2/(1+np.power(10**mask_compress,-after_lut[:,:,:,2])))-1)*100
    b_min_mask=(np.array(2/(1+np.power(10**mask_compress,-after_lut[:,:,:,2])))-1)*100
    a_plus_mask[a_plus_mask<0]=0
    a_min_mask[a_min_mask>0]=0
    b_plus_mask[b_plus_mask<0]=0
    b_min_mask[b_min_mask>0]=0
    #     ProLab PARAMS
    after_lut[:,:,:,1]=(after_lut[:,:,:,1]*sut*4)
    after_lut[:,:,:,2]=(after_lut[:,:,:,2]*sut*4)
    
    after_lut[:,:,:,1][after_lut[:,:,:,1]<0]=after_lut[:,:,:,1][after_lut[:,:,:,1]<0]*a_min_sut
    after_lut[:,:,:,1][after_lut[:,:,:,1]<0]=((2/(1+(np.power((16**end_a_minus),-(((after_lut[:,:,:,1][after_lut[:,:,:,1]<0])))/100))))-1)*(31/end_a_minus)#*mask[:,:,1][np.where(after_lut[:,:,:,1]<0)]     #a+ ибо обратились к 2 каналу где этот канал больше нуля
    
    #after_lut[:,:,:,0]+=a_min_L*(-a_min_mask/(mask_compress*100))
    
    after_lut[:,:,:,1][after_lut[:,:,:,1]>=0]=after_lut[:,:,:,1][after_lut[:,:,:,1]>=0]*a_plus_sut
    after_lut[:,:,:,1][after_lut[:,:,:,1]>=0]=((2/(1+(np.power((16**end_a_plus),-((after_lut[:,:,:,1][after_lut[:,:,:,1]>=0]))/100))))-1)*(31/end_a_plus)#*mask[:,:,1][np.where(after_lut[:,:,:,1]>=0)]
    
    #after_lut[:,:,:,0]+=a_plus_L*a_plus_mask/(mask_compress*100)
    
    after_lut[:,:,:,2][after_lut[:,:,:,2]<0]=after_lut[:,:,:,2][after_lut[:,:,:,2]<0]*b_min_sut
    after_lut[:,:,:,2][after_lut[:,:,:,2]<0]=((2/(1+(np.power((16**end_b_minus),-((after_lut[:,:,:,2][after_lut[:,:,:,2]<0]))/100))))-1)*(31/end_b_minus)#*mask[:,:,2][np.where(after_lut[:,:,:,2]<0)]
    
    #after_lut[:,:,:,0]+=b_min_mask*(-b_min_L)/(mask_compress*100)
    
    after_lut[:,:,:,2][after_lut[:,:,:,2]>=0]=after_lut[:,:,:,2][after_lut[:,:,:,2]>=0]*b_plus_sut
    after_lut[:,:,:,2][after_lut[:,:,:,2]>=0]=((2/(1+(np.power((16**end_b_plus),-((after_lut[:,:,:,2][after_lut[:,:,:,2]>=0]))/100))))-1)*(31/end_b_plus)#*mask[:,:,2][np.where(after_lut[:,:,:,2]>=0)]                 #((2/1+16**yx)-1)*100/y'''
    
    #after_lut[:,:,:,0]+=b_plus_L*b_plus_mask/(mask_compress*100)

    #HUE MiX
    a_min=np.array(after_lut[:,:,:,1])
    a_plus=np.array(after_lut[:,:,:,1])
    b_min=np.array(after_lut[:,:,:,2])
    b_plus=np.array(after_lut[:,:,:,2])

    a_min=a_min*(1-a_m_hue)+after_lut[:,:,:,2]*a_m_hue
    a_plus=a_plus*(1-a_p_hue)+after_lut[:,:,:,2]*a_p_hue
    b_min=b_min*(1-b_m_hue)+after_lut[:,:,:,1]*b_m_hue
    b_plus=b_plus*(1-b_p_hue)+after_lut[:,:,:,1]*b_p_hue

    a_min[a_min>0]=0
    a_plus[a_plus<0]=0
    b_min[b_min>0]=0
    b_plus[b_plus<0]=0

    a=a_min+a_plus
    b=b_min+b_plus

    after_lut[:,:,:,1]=a
    after_lut[:,:,:,2]=b

    return after_lut
@cache_data
def wheels(after_lut,
            steepness_shad,width_shad,
            steepness_light,width_light,
            shad_a,shad_b,mid_a,mid_b,light_a,light_b,
            neutral_mask,amplyfy_wheel
            ):

    #     MASK FOR WHEELS
    wheel_sut_mask=np.fabs(np.array(after_lut[:,:,:,1]+after_lut[:,:,:,2]))
    wheel_sut_mask/=np.max(wheel_sut_mask)
    wheel_sut_mask=np.stack((wheel_sut_mask,wheel_sut_mask,wheel_sut_mask),axis=-1)
    

    shad_mask=np.array(after_lut[:,:,:,0])/100
    mid_mask=np.array(after_lut[:,:,:,0])/100
    mid_mask=(mid_mask*0)+1
    light_mask=np.array(after_lut[:,:,:,0])/100 
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
    shad=np.array(after_lut)
    mid=np.array(after_lut)
    light=np.array(after_lut)

    shad[:,:,:,1]+=shad_a*(((1-neutral_mask)+1)**2)*amplyfy_wheel
    shad[:,:,:,2]+=shad_b*(((1-neutral_mask)+1)**2)*amplyfy_wheel
    mid[:,:,:,1]+=mid_a*(((1-neutral_mask)+1)**2)*amplyfy_wheel
    mid[:,:,:,2]+=mid_b*(((1-neutral_mask)+1)**2)*amplyfy_wheel
    light[:,:,:,1]+=light_a*(((1-neutral_mask)+1)**2)*amplyfy_wheel
    light[:,:,:,2]+=light_b*(((1-neutral_mask)+1)**2)*amplyfy_wheel

    shad*=shad_mask
    mid*=mid_mask
    light*=light_mask
    wheel_sut_mask=(wheel_sut_mask+10**(1-neutral_mask))/(10**(1-neutral_mask)+1)
    after_wheel=shad+mid+light
    after_wheel*=(1-wheel_sut_mask)
    after_lut*=wheel_sut_mask
    after_lut=after_lut+after_wheel



    return after_lut



def to_rgb(after_lut):
    to_XYZ=ProLab_to_XYZ(after_lut)
    to_XYZ[to_XYZ>1]=1
    to_XYZ[to_XYZ<0]=0
    after_lut=XYZ_to_sRGB(to_XYZ)
    
    #after_lut=(after_lut-0.2-Exp_shift/9)*3
    after_lut[after_lut>1]=1
    after_lut[after_lut<0]=0
    after_lut=np.array(after_lut,dtype=np.float32)
    
    return after_lut


#p_lut=par_lut(wrong,right,exp_shift,sut,end_a_plus,end_a_minus,end_b_plus,end_b_minus)
#@jit(nopython=True)
@cache_data()
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


#gjfbn=colmap(img,p_lut)'''

#@cache_data()
def log(iimg,gamma) -> np.array:
    
    rgb_autoscale=(iimg*((13*(10**gamma))/np.average(iimg)))

    rgb_autoscale+=(200-np.min(rgb_autoscale)) #(np.log(gamma+2.7)-1)
    
    rgblog=(np.log10(rgb_autoscale))
    rgblog=(((rgblog-np.min(rgblog))/(np.max(rgblog)-np.min(rgblog))))
  
    #in_img=(((rgblog-2.3)/(np.max(rgblog)-2.3)))
    return rgblog


@cache_data
def film_em(in_img,Wb_b,Wb_r,print_cont,print_exp,Lut,gamma,light_compr,Wb_r2,Wb_b2) -> np.array:
    '''                                 WB               '''
    

    
    in_img=(((in_img-np.min(in_img))/(np.max(in_img)-np.min(in_img))))
    #in_img=np.nan_to_num(in_img)
    in_img[:,:,0]-=Wb_b/90
    in_img[:,:,1]-=((Wb_b/90)+(Wb_r/90))
    in_img[:,:,2]-=Wb_r/90
    
    qwert=time.perf_counter()
     #table=
    
    #in_img=colmap(in_img,Lut)
    in_img=(in_img-np.average(in_img))+(print_exp/9)
    in_img[in_img>0]=(((10**light_compr)/(-in_img[in_img>0]-(10**light_compr)))+1)*(10**light_compr)
    print(in_img.dtype)
    qwerty=time.perf_counter()
    print(f"Вычисление заняло {qwerty - qwert:0.4f} секунд")
    in_img+=(-0.15)+((gamma-3.1)/18)
    in_img[in_img<-0.6]=-0.6
    gamma=np.sqrt(np.sqrt(gamma))

    print((10**(gamma*print_cont)),'gamma')
    '''                         CONTRAST                    '''

    #img_contrast=1/(1+(np.power(55,(-in_img))))
    img_contrast=1/(1+(np.power((10**(gamma*print_cont)),(-in_img))))
    #img_contrast+=0.16
    print(np.min(img_contrast), "img cont")
    img_contrast=my_interpolation_trilinear(img_contrast,table=Lut)
    
    img_contrast[:,:,0]-=((Wb_b2/90)+(Wb_r2/90))
    img_contrast[:,:,1]-=Wb_b2/90
    img_contrast[:,:,2]-=Wb_r2/90


    return img_contrast


#ticcc=time.perf_counter()
#p_image=film_em(img,WB_b,WB_r,print_cont,print_exp,p_lut,gamma)
#toc = time.perf_counter()
#print(f"Вычисление заняло {toc - ticcc:0.4f} секунд")

def grain(image,grain_curve,prep_grain,amplify,amplify_mask) -> np.array:
    
    Mask_gr=grain_curve.apply(image)
    grain=((prep_grain*amplify*(Mask_gr**amplify_mask)))/(amplify_mask**1.5)+1
    grain=np.array(grain,dtype=np.float32)
    grained=image*grain
    grained[grained>=1]=1
    grained[grained<=0]=0
    #grained=(grained+np.fabs(np.min(grained)))*(1/np.max(grained+np.fabs(np.min(grained))))
    return grained

#p_image=grain(p_image,gr_curve,prep_grain,0.15,6)

def show(image):
    image=io.convert_bit_depth(image,'uint8')
    return image

def plot_zone(
            width_shad,
            steepness_shad,
        
            width_light,
            steepness_light,):
    def curve(img,steepness,width,apex):
        img=(2/(1+np.power(10**steepness,(-width-apex+img))))*(2/(1+np.power(10**steepness,(-width+apex-img))))
        return img
    plot_shad=np.linspace(0,1,100)
    plot_mid=np.linspace(0,1,100)
    plot_mid=(plot_mid*0)+1
    plot_light=np.linspace(0,1,100)

    shad=curve(plot_shad,steepness_shad,width_shad,0)
    shad*=(1/np.max(shad))
    light=curve(plot_light,steepness_light,width_light,1)
    light*=(1/np.max(light))
    mid=(plot_mid-shad-light)
    mid[mid<0]=0

    all={"Shadow": shad,"Mid":mid,"Hilight":light}

    return all

def crop(img,rotate,rotate_thin,crop_sl,y_shift,x_shift,aspect):
    img=ndimage.rotate(img,rotate )
    img=ndimage.rotate(img, rotate_thin)

    
    shape=np.shape(img)
    coef=float(np.max(shape)/1500)
    crop_sl*=(coef/300)
    y_shift*=crop_sl*10
    x_shift*=crop_sl*10
    crop_sl+=1
    y=int(shape[0]/crop_sl)
 
    y_2=shape[0]-y
    x=int(shape[1]/crop_sl)
    x_2=shape[1]-x
    img=img[(y_2+int(y_shift)):(y+int(y_shift)),(x_2+int(x_shift)):(x+int(x_shift)),:]
    

    shape_2=(np.shape(img))
    if shape_2[0]<shape_2[1]:
        z=aspect[0]/aspect[1]
        x_3=int(shape_2[0]*z)
        print(x_3)
        sup=(shape_2[1]-x_3+4)/2
        print(sup)
        img=img[:,int(sup):int(x_3+sup),:]
    else:
        z=aspect[0]/aspect[1]
        
        x_3=shape_2[1]*z
        sup=(shape_2[0]-x_3)/2
        img=img[int(sup):int(x_3+sup),:,:]        


    return img



def done1():
    print("done1")
#show(p_image)



'''    qwert=time.perf_counter()
    qwerty=time.perf_counter()
    print(f"Вычисление заняло {qwerty - qwert:0.4f} секунд")'''

'''def my_interpolation_trilinear(
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
'''
'''
@cache_data()
@jit(nopython=True)
def colmap(img,lut) -> np.array:
    shape=img.shape
    size=lut.shape[0]-1.1
    img=np.reshape(img,((img.shape[0]*img.shape[1]),3))
    img = np.clip(img, 0, 1)
    img*=size
    img_i=img%1
    img_c=np.trunc(img)
    img_c=img_c.astype(np.uint8)
    for i in range(img.shape[0]):
        coef=lut[img_c[i,0]+1,img_c[i,1]+1,img_c[i,2]+1]-lut[img_c[i,0],img_c[i,1],img_c[i,2]]
        #coef=np.array(lut[(int(img_c[i,0])+1),(int(img_c[i,1])+1),(int(img_c[i,2])+1)])-(lut[img_c[i,0],img_c[i,1],img_c[i,2]])
       
        #print(lut[img_c[i,0],img_c[i,1],img_c[i,2]])
        #coef*=10
        
        pers=np.array([img_i[i,0],img_i[i,1],img_i[i,2]])
        #print(pers)
        coef=pers*coef
        #print(coef)
        img[i]=lut[img_c[i,0],img_c[i,1],img_c[i,2]]
        img[i]+=coef
    img=np.reshape(img,shape)
    return img'''


'''
def film_pill(iimg,Wb_b,Wb_r,print_cont,print_exp,Lut,gamma):
    rgb_autoscale=(iimg*(((np.log(gamma+2.7)-1)*10000)/np.average(iimg)))+200
    rgblog=(np.log10(rgb_autoscale))
    in_img=(((rgblog-2.3)/(np.max(rgblog)-2.3)))
    
    
                                     WB               
    in_img[:,:,0]-=Wb_b/90
    in_img[:,:,1]-=((Wb_b/90)+(Wb_r/90))
    in_img[:,:,2]-=Wb_r/90
    
    in_img=((in_img-np.average(in_img)))+0.4
    qwert=time.perf_counter()
    in_img=my_interpolation_trilinear(in_img,Lut)
    in_img=(in_img-np.average(in_img))+(print_exp/9)
    print(in_img.dtype)
    qwerty=time.perf_counter()
    print(f"Вычисление заняло {qwerty - qwert:0.4f} секунд")

    print_cont*=gamma*100
    print_cont=10**print_cont


                             CONTRAST                    

    img_contrast=1/(1+(np.power((print_cont),(2*(-in_img)))))
    

    return img_contrast'''