import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import rawpy
from scipy import ndimage
from colour import io
from numba import njit, prange
from colour import gamma_function
import sys
sys.frozen = True

from PIL import Image

def zoom(q_in_zoom,q_out_zoom):
    while True:
        get_par = q_in_zoom.get()
        zoom_chank = get_par[0]
        resolution = get_par[1]
        gr_sample = get_par[2]
        on_grain = get_par[3]
        shape=[[]]
        for b in zoom_chank:
            shape=zoom_chank[b].shape
        shape3=shape[1]
        shape2=shape[0]

        if shape[0]>shape[1]:
            gr_sample=gr_sample.transpose(method=Image.Transpose.ROTATE_90)

        Grain_crop=gr_sample.resize((int(shape[1]),int(shape[0])),resample=Image.LANCZOS, )
        grain=np.array(Grain_crop,dtype=np.uint8)
        grain=io.convert_bit_depth(grain,"float64")
        grain_0=(grain-np.average(grain))
        grain_0 = grain_0[...,1]


        @njit(fastmath=True, cache=True)
        def grinder(img2,img3,val,size_of_rand,shape2,shape3):
            

            step = 0
            x=0
            for _ in range(1):
                for z in prange(int(shape3-size_of_rand)):
                            step = 0
                            while step<=shape2-size_of_rand:
                                img2[(step):(step+val[x]),z:z+val[x]]=np.average(img2[(step):(step+val[x]),z:z+val[x]])
                                step = step+val[x]
                                x=x+1
                                if x>=10000:
                                    x=0

                    
                for z in prange((shape2-size_of_rand)):
                            step = 0
                            while step<=shape3-size_of_rand:
                                img3[z:z+val[x],(step):(step+val[x])]=np.average(img3[z:z+val[x],(step):(step+val[x])])
                                step = step+val[x]
                                x=x+1
                                if x>=10000:
                                    x=0
            img=(img2+img3)/2

            return img        



        for i in zoom_chank:
            if on_grain != False:
                print(on_grain)
                if shape2>shape3:
                     size_of_rand=shape3
                else:
                     size_of_rand=shape2

                size_of_rand = int(np.around(size_of_rand/1000))
                if size_of_rand<3:
                    size_of_rand = 3

                print(size_of_rand)
                zoom_chank[i] = gamma_function(zoom_chank[i],0.4) #gamma 2.5
                zoom_chank[i] = np.nan_to_num(zoom_chank[i])

                zoom_chank[i]=zoom_chank[i]+(grain_0/30)


                zoom_chank[i][zoom_chank[i]<0]=0
                zoom_chank[i][zoom_chank[i]>1]=1
                zoom_chank[i] = gamma_function(zoom_chank[i],2.5)



                val = np.random.default_rng().integers(low=1,high=size_of_rand,size=10001)
                img2= np.array(zoom_chank[i])
                zoom_chank[i]=grinder(zoom_chank[i],img2,val,size_of_rand,shape2,shape3)




            zoom_chank[i]=ndimage.zoom(zoom_chank[i],(resolution,resolution))

        q_out_zoom.put(zoom_chank)
        q_in_zoom.task_done()



def raw(raw_in,WB,is_half) -> np.array:

    cam=False
    auto=False
    if WB=="In camera WB":
        cam=True
    else:
        auto=True

    with rawpy.imread(raw_in) as raw:
            rgb = raw.postprocess(output_color=rawpy.ColorSpace(0),  demosaic_algorithm=rawpy.DemosaicAlgorithm(3), half_size=False,
                                use_camera_wb=cam, use_auto_wb=auto, highlight_mode=rawpy.HighlightMode(0),#user_wb=(1,1,1,1),
                                output_bps=16,  no_auto_scale=False, no_auto_bright=False,auto_bright_thr=0.000000001,
                                gamma=(1.0,4.5), chromatic_aberration=(1,1),)
    rgb=np.dstack((rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]))
    rgb = io.convert_bit_depth(rgb,"float32")
    return rgb
