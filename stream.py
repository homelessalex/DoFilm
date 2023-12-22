from streamlit import (set_page_config, sidebar,expander,header,slider,file_uploader,number_input,
                       container,image,tabs,selectbox,divider,download_button,button,line_chart,area_chart,
                       select_slider, session_state,toggle)
from PIL import Image, ImageCms
from pathlib import Path
import numpy as np
from colour import read_image,io,sRGB_to_XYZ,XYZ_to_ProLab,read_LUT,write_image
from Do_film import raw,zoom,pre_grain,reduse_sharpness,film_em,grain,show,bloom,sharpen,log,plot_zone,rgb_mix,apply_film,ProLab,wheels,to_rgb,crop
from io import BytesIO
set_page_config(layout="wide")
import time
import os
import signal
import pickle
cameras = pickle.load(open("Cum.pkl",'rb'))
Films = pickle.load(open("Films.pkl",'rb'))





with  sidebar:
    

    
    tab_input ,tab_colour, tab_struct, tab_wheels, tab_crop  = tabs(['In/Out','Colour','Structure','Wheels', 'Crop'])
    with tab_input:
        with  expander("Upload raw image file"):
            uploaded_file =  file_uploader("Upload raw image file",
                                        type=["NEF", "cr3", "cr2", "dng", "tiff"],
                                        label_visibility="hidden",)

        camera = selectbox(
        label='Camera', options=cameras
                            )

        film = selectbox(
        label='Film', options=Films
                            )

        resolution =  number_input("Viewer resolution",min_value=1000,max_value=4000,value=1400)
        divider()
        download = button(
            label="Render for download")
        resolution_on_save=number_input("Resolution on save",min_value=1500,max_value=6000,value=3000)
        

    with tab_colour:

        defoult_colour = button("Set to defoult",use_container_width=True)
        
        if defoult_colour:
                session_state.wbr = 0.
                session_state.wbb = 0.
                session_state.gamma = 0.
                session_state.print_exp = 0.
                session_state.print_cont = 0.7
                session_state.sut = 1.
                session_state.light_compr = -1.
                session_state.wbr_af = 0.
                session_state.wbb_af = 0.

        header("On shot")
        WB_r =  slider("Wight Balance | Red", min_value=-10., max_value=10., value=0., step=.01, key="wbr")
        WB_b =  slider("Wight Balance | Blue", min_value=-10., max_value=10., value=0., step=.01,key="wbb")
        gamma = 3.1+slider("Film exposer", min_value=-3., max_value=6., value=0., step=0.05, key="gamma")
        
        #exp_shif =  slider("Exposure shift (ev)", min_value=-3., max_value=3., value=-1., step=.1)
        divider()
        header("Print")
        
        print_exp = -1+ slider("Print exposure", min_value=-4., max_value=4., value=0., step=.01, key="print_exp")
        print_cont =  slider("Contrast", min_value=0.1, max_value=2., value=0.7, step=.025, key="print_cont")     
        sut = np.log( slider("Saturation", min_value=0.05, max_value=2., value=1., step=.05, key='sut') + 1)
        light_compr= -(slider ('Light compression',min_value=-1.,max_value=1.,value=-1.,step=0.01, key="light_compr"))
        divider()
        WB_r2 =  slider("ColourHead | Magenta", min_value=-10., max_value=10., value=0., step=.01, key="wbr_af")
        WB_b2 =  slider("ColourHead | Yelow", min_value=-10., max_value=10., value=0., step=.01,key="wbb_af")

        header("Advanced")

        with  expander("ProLab channels"):
            mask_compress=slider("Mask compression",min_value=0.01,max_value=1.,value=0.5,step=0.01)
            divider()
            END_A_PLUS =  slider("A+ compression", min_value=0.1, max_value=2., value=0.5,step=.05)
            a_plus_sut = slider("A+ Sut",min_value=0.5,max_value=2.,value=1.,step=0.01)
            a_p_hue = slider('A+/B Mix',min_value=-1.,max_value=1.,value=0.,step=0.01)
            divider()
            END_A_MINUS =  slider("A- compression", min_value=0.1, max_value=2., value=0.5,step=.05)
            a_min_sut= slider("A- Sut",min_value=0.5,max_value=2.,value=1.,step=0.01)
            a_m_hue = slider('A-/B Mix',min_value=-1.,max_value=1.,value=0.,step=0.01)
            divider()
            END_B_PLUS =  slider("B+ compression", min_value=0.1, max_value=2., value=0.5,step=.05)
            b_plus_sut= slider("B+ Sut",min_value=0.5,max_value=2.,value=1.,step=0.01)
            b_p_hue = slider('B+/A Mix',min_value=-1.,max_value=1.,value=0.1,step=0.01)
            divider()
            END_B_MINUS =  slider("B- compression", min_value=0.1, max_value=2., value=0.5,step=.05)
            b_min_sut= slider("B- Sut",min_value=0.5,max_value=2.,value=1.,step=0.01)
            b_m_hue = slider('B-/A Mix',min_value=-1.,max_value=1.,value=0.,step=0.01)
        
        with expander ("RGB Mixer"):
            r_sut = slider("R Sut",min_value=-1.,max_value=1.,value=0.,step=0.01)
            r_hue= slider("R Hue",min_value=-1.,max_value=1.,value=0.,step=0.01)
            r_g = slider("R/G Mix",min_value=-1.,max_value=1.,value=0.,step=0.01)
            r_b = slider("R/B Mix",min_value=-1.,max_value=1.,value=0.,step=0.01)
            divider()
            g_sut = slider("G Sut",min_value=-1.,max_value=1.,value=0.,step=0.01)
            g_hue= slider("G Hue",min_value=-1.,max_value=1.,value=0.,step=0.01)
            g_r = slider("G/R Mix",min_value=-1.,max_value=1.,value=0.,step=0.01)
            g_b = slider("G/B Mix",min_value=-1.,max_value=1.,value=0.,step=0.01)
            divider()
            b_sut = slider("B Sut",min_value=-1.,max_value=1.,value=0.,step=0.01)
            b_hue= slider("B Hue",min_value=-1.,max_value=1.,value=0.,step=0.01)
            b_r = slider("B/R Mix",min_value=-1.,max_value=1.,value=0.,step=0.01)
            b_g = slider("B/G Mix",min_value=-1.,max_value=1.,value=0.,step=0.01)    

        accuracy = select_slider("film repetition accuracy",(1,2,3,4,5,6,7),value=7)        
    with tab_struct:
            
            header("Reduse sharpness")
            blur_rad =  slider("Radius ", min_value=1.1, max_value=5., value=1.8, step=.1)
            blur_spred =  slider("Spread ", min_value=.1, max_value=16., value=8.5, step=.1)
            halation =  slider("Halation ", min_value=1.0, max_value=3., value=1.8, step=.1)

            header("Bloom")
            bloom_rad =  slider("Radius", min_value=10., max_value=300., value=140., step=10.0)
            bloom_spred =  slider("Spread", min_value=.1, max_value=30., value=25., step=.1)
            bloom_Halation =  slider("Halation", min_value=1.0, max_value=2., value=1., step=.025)

            header("Local contrast")
            sharp_rad =  slider("Radius  ", min_value=1.1, max_value=100., value=40., step=.1)
            sharp_spred =  slider("Spread  ", min_value=1., max_value=30., value=15., step=.1)
            sharp_amplif = slider("Amplify  ", min_value=0., max_value=300., value=0., step=1.)

            header("grain")
            AMPLIFY_GRAIN =  slider("Amplify", min_value=0., max_value=10., value=2., step=.1)
            AMPLIFY_GRAIN_MASK =  slider("Amplify mask", min_value=1., max_value=10., value=7., step=.5)
    with tab_wheels:
            
            defoult_wheel = button("Set to defoult",use_container_width=True, key="def_wheel")
        
            if defoult_wheel:
                    session_state.w_shad = 0.25
                    session_state.s_shad = 6.
                    session_state.w_light = 0.25
                    session_state.s_light = 6.
                    session_state.ntrl_msk = 0.
                    session_state.amp_wheel = 10.
                    session_state.shad_a = 0.
                    session_state.shad_b = 0.
                    session_state.mid_a = 0.
                    session_state.mid_b = 0.
                    session_state.light_a = 0.
                    session_state.light_b = 0.


            tab_ms,tab_mh = tabs(['Shadow mask','Highlight mask'])

            with tab_ms:
                width_shad=slider("width shadow",min_value=0.01,max_value=1.,value=0.25,step=0.01,key="w_shad")
                steepness_shad=slider("steepness shadow",min_value=1.,max_value=25.,value=6.,key="s_shad")
            with tab_mh:
                width_light=slider("width highlight",min_value=0.01,max_value=1.,value=0.25,step=0.01, key="w_light")
                steepness_light=slider("steepness highlight",min_value=1.,max_value=25.,value=6. , key="s_light")
            neutral_mask = slider('Neutral involve', min_value=-1.,max_value=1.,value=0.,step=0.01, key="ntrl_msk")
            amply_wheel = slider ("Amplyfy shift", min_value= 1., max_value= 10.,value=10.,step=0.25, key="amp_wheel")
            plot=plot_zone(width_shad,steepness_shad,width_light,steepness_light)
            area_chart(plot,height=200,y=None,color=([200,220,255,255],[128,128,128],[0,0,0,255]))

            shad_a=slider("Shadow A shift",min_value=-3.,max_value=3.,value=0.0,step=0.01, key="shad_a")
            shad_b=slider("Shadow B shift",min_value=-3.,max_value=3.,value=0.0,step=0.01, key="shad_b")
            divider()
            mid_a=slider("Mid A shift",min_value=-3.,max_value=3.,value=0.0,step=0.01, key="mid_a")
            mid_b=slider("Mid B shift",min_value=-3.,max_value=3.,value=0.0,step=0.01, key="mid_b")
            divider()
            light_a=slider("Hilights A shift",min_value=-3.,max_value=3.,value=0.0,step=0.01, key="light_a")
            light_b=slider("Hilights B shift",min_value=-3.,max_value=3.,value=0.0,step=0.01, key="light_b")
            divider()            

    with tab_crop:
        rotate = select_slider("Rotate",([0,90,180,270]))
        aspect = select_slider("Aspect Ratio",([[ 3,2],[1.41,1],[ 4,3],[ 7,6],[ 1,1]]))
        rotate_thin= slider('Horizon',min_value=-10.,max_value=10., value=0.,step=0.01)
        crop_sl = slider('Crop',min_value=0.,max_value=100.,value=0.)
        y_shift = slider('Y Shift',min_value=-100.,max_value=100.,value=0.)
        x_shift = slider('X shift',min_value=-100.,max_value=100.,value=0.)


#Film_curve=read_LUT('do_film/sup/film/portra400/film_curve_porttra_400.spi1d')
Grain_curve=read_LUT('Grain_curve.spi1d')
Gr_sample=Image.open('grain_portra400.tif')



#right=read_image(('do_film/sup/film/c200/fuji_c200.tiff'),'uint16',"ImageIO")
#right=io.convert_bit_depth(right, "float32")
#wrong=read_image(('do_film/sup/cum/sigma_fp.tiff'),'uint16',"ImageIO")
#wrong=io.convert_bit_depth(wrong, "float32")
#right=sRGB_to_XYZ(right)
#right=XYZ_to_ProLab(right)
#right=((np.reshape(right, (216,3))).tolist())








if uploaded_file is not None:
    wrong=cameras[camera]
    right=Films[film]




    img=raw(uploaded_file)
    img=zoom(img,resolution)

    prep_grain=pre_grain(img,Gr_sample)
    img=reduse_sharpness(img, blur_rad, halation, blur_spred)
    img=bloom(img,bloom_rad,bloom_Halation,bloom_spred)
    p_lut=rgb_mix( r_hue,r_sut,r_g,r_b,g_hue,g_sut,g_r,g_b,b_hue,b_sut,b_r,b_g)
    p_lut=apply_film(p_lut,wrong,right,accuracy)
    p_lut=ProLab(p_lut,sut,END_A_PLUS,END_A_MINUS,END_B_PLUS,END_B_MINUS,a_min_sut,a_plus_sut,b_min_sut,b_plus_sut,mask_compress
                 ,a_m_hue,a_p_hue,b_m_hue,b_p_hue)
    p_lut=wheels(p_lut,steepness_shad,width_shad,steepness_light,width_light,shad_a,shad_b,mid_a,mid_b,light_a,light_b,neutral_mask,amply_wheel)
    p_lut=to_rgb(p_lut)
    p_image=log(img,gamma)
    p_image=sharpen(p_image,sharp_rad,sharp_spred,sharp_amplif)
    p_image=film_em(p_image,WB_b,WB_r,print_cont,print_exp,p_lut,gamma,light_compr,WB_r2,WB_b2)
    
    p_image=grain(p_image,Grain_curve,prep_grain,AMPLIFY_GRAIN,AMPLIFY_GRAIN_MASK)
    
    p_image=show(p_image)
    p_image=crop(p_image,rotate,rotate_thin,crop_sl,y_shift,x_shift,aspect)
    
    if p_image.shape[0]>p_image.shape[1]:
        with  container():
             image(p_image,width=int(resolution/2) )
    else:   
        with  container():
             image(p_image, use_column_width="never" )

if download:
    #wrong=cameras[camera]
    #wrong=sRGB_to_XYZ(wrong)
    #wrong=XYZ_to_ProLab(wrong)
    #wrong=((np.reshape(wrong, (216,3))).tolist())


    img=raw(uploaded_file)
    img=zoom(img,resolution_on_save)

    prep_grain=pre_grain(img,Gr_sample)
    img=reduse_sharpness(img, blur_rad, halation, blur_spred)
    img=bloom(img,bloom_rad,bloom_Halation,bloom_spred)
    qwert=time.perf_counter()
    p_lut=rgb_mix( r_hue,r_sut,r_g,r_b,g_hue,g_sut,g_r,g_b,b_hue,b_sut,b_r,b_g)
    p_lut=apply_film(p_lut,wrong,right,accuracy)
    p_lut=ProLab(p_lut,sut,END_A_PLUS,END_A_MINUS,END_B_PLUS,END_B_MINUS,a_min_sut,a_plus_sut,b_min_sut,b_plus_sut,mask_compress
                 ,a_m_hue,a_p_hue,b_m_hue,b_p_hue)
    p_lut=wheels(p_lut,steepness_shad,width_shad,steepness_light,width_light,shad_a,shad_b,mid_a,mid_b,light_a,light_b,neutral_mask,amply_wheel)
    p_lut=to_rgb(p_lut)
    p_image=log(img,gamma)
    p_image=sharpen(p_image,sharp_rad,sharp_spred,sharp_amplif)
    p_image=film_em(p_image,WB_b,WB_r,print_cont,print_exp,p_lut,gamma,light_compr,WB_r2,WB_b2)
    
    p_image=grain(p_image,Grain_curve,prep_grain,AMPLIFY_GRAIN,AMPLIFY_GRAIN_MASK)
    
    p_image=show(p_image)
    p_image=crop(p_image,rotate,rotate_thin,crop_sl,y_shift,x_shift,aspect)
    p_image=show(p_image)
    print(p_image.shape)
    pp_image=Image.fromarray(p_image)
    buffer = BytesIO()
    for_icc = Image.open('IMG_4799.jpg')
    icc = for_icc.info.get('icc_profile')
    pp_image.save(buffer, format="JPEG", quality=95, icc_profile=icc)
    img_bytes = buffer.getvalue()
    download_button(
        label="Download image",
        data=img_bytes,
        mime="image/jpeg",
        use_container_width=True)

exit_app =  sidebar.button("Shut Down")
if exit_app:
    os.kill(os.getppid(), signal.SIGHUP)


def done():
    print("done")

#wrong=read_image(('do_film/sup/cum/sigma_fp.tif'),'uint16',"ImageIO")
#wrong=io.convert_bit_depth(wrong, "float32")