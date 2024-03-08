import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

import cash
import time
import slising
import Not_multi_funk
from Zoom import raw,zoom
from multiprocessing import Process
from blurs import reduse_sharpness
from Lut import lut_main
from apply_all import apply_film
cameras = cash.cameras
Films = cash.Films
inface = cash.inface
Grain_curve = cash.Grain_curve
Gr_sample = cash.Gr_sample
from cash import q_in_zoom, q_out_zoom, q_in_blur, q_out_blur , q_in_lut, q_out_lut, q_in_film , q_out_film 
from multiprocessing import active_children
import os


if cash.close==True:
                    active = active_children()
                    for child in active:
                        child.kill()
                    for child in active:
                        child.join()

                    os._exit()




def magic(resolution,uploaded_file,wight_balance, blur_rad, halation, blur_spred,
            bloom_rad, bloom_Halation, bloom_spred,sharp_rad, sharp_spred, sharp_amplif,
            camera,
           
            
            r_hue,r_sut,r_g,r_b,g_hue,g_sut,g_r,g_b,b_hue,b_sut,b_r,b_g,
            film,accuracy,

            sut,end_a_plus,end_a_minus,end_b_plus,end_b_minus,a_min_sut,a_plus_sut,
            b_min_sut,b_plus_sut,a_m_hue,a_p_hue,b_m_hue,b_p_hue,sat_comp,
            
            steepness_shad,width_shad,steepness_light,width_light,shad_a,shad_b,
            mid_a,mid_b,light_a,light_b,neutral_mask,amplyfy_wheel,
            
            Wb_b,Wb_r,Wb_b2,Wb_r2,print_exp,light_compr,zone,print_cont,gamma,
            
            r_scan,b_scan,amplify_grain,amplify_mask,on_grain,white_point,is_half,rotate,aspect): #, bloom_rad, bloom_Halation, bloom_spred, r_hue, r_sut, r_g, r_b, g_hue, g_sut, g_r, g_b, b_hue, b_sut, b_r, b_g, camera, film, accuracy, sut, END_A_PLUS, END_A_MINUS, END_B_PLUS, END_B_MINUS, a_min_sut, a_plus_sut, b_min_sut, b_plus_sut, a_m_hue, a_p_hue, b_m_hue, b_p_hue, sut_compress, steepness_shad, width_shad, steepness_light, width_light, shad_a, shad_b, mid_a, mid_b, light_a, light_b, neutral_mask, amply_wheel, gamma, sharp_rad, sharp_spred, sharp_amplif, WB_b, WB_r, print_cont, print_exp, light_compr, WB_r2, WB_b2, zone, AMPLIFY_GRAIN, AMPLIFY_GRAIN_MASK):
        #if __name__ == '__main__':
                

                cash.q_in_blur.join()
                cash.q_in_zoom.join()
                cash.q_out_zoom.join()
                cash.q_out_blur.join()
                cash.q_in_lut.join()
                cash.q_out_lut.join()
                cash.q_in_film.join()
                cash.q_out_film.join()

                if cash.zoom_count == 0 :
            
                    zoom_process = [Process(target = zoom,args=(cash.q_in_zoom,cash.q_out_zoom))for i in range(3)]
                    for i in zoom_process:
                        i.start()
                    cash.zoom_count = 1
                    cash.names_zoom_process = zoom_process

                if cash.blur_count == 0:
            
                    blur_process = [Process(target = reduse_sharpness,args=(cash.q_in_blur,cash.q_out_blur))for i in range(3)]
                    for i in blur_process:
                        i.start()
                    cash.blur_count = 1
                    cash.names_blur_process = blur_process
                if cash.lut_count == 0:
            
                    lut_process = [Process(target = lut_main,args=(cash.q_in_lut,cash.q_out_lut))for i in range(4)]
                    for i in lut_process:
                        i.start()
                    cash.lut_count = 1
                    cash.names_lut_process = lut_process
                if cash.film_count == 0:
            
                    film_process = [Process(target = apply_film,args=(cash.q_in_film,cash.q_out_film))for i in range(4)]
                    for i in film_process:
                        i.start()
                    cash.film_count = 1
                    cash.names_film_process = film_process
                wrong=cameras[camera]
                right=Films[film]

                

                
                #RAW
                if cash.uploaded_file_temp!=uploaded_file or cash.WB_t!=wight_balance: 
                    img = raw(uploaded_file,wight_balance,is_half)
                    slises_zoom_tup = slising.slise_for_blur(img) #разделяем для паралельки
                    cash.uploaded_file_temp=uploaded_file
                    cash.WB_t=wight_balance
                    cash.img_raw=slises_zoom_tup
                    cash.resolution_t=None
                elif cash.uploaded_file_temp==uploaded_file and cash.WB_t==wight_balance:
                    slises_zoom_tup=cash.img_raw
                    print("raw yes")
                #ZOOM
                ticZoom = time.time()
                
                if cash.resolution_t!=resolution or cash.on_grain_t!=on_grain:
                    slises_zoom = slises_zoom_tup[0]
                    #slise_zoom_names_keys = img[1]
                    shape_zoom = slises_zoom_tup[1]
                    resolution_coef = float(resolution/float(np.max(shape_zoom)))
                    count = 0
                    for i in slises_zoom:
                        count+=1
                        cash.q_in_zoom.put([slises_zoom[i],resolution_coef,Gr_sample,on_grain])
            
                    cash.q_in_zoom.join()
                    print("done2")  
                    after_zoom = {}
                    for i in range(3):
                        temp = {} 
                        temp[i] = cash.q_out_zoom.get()
                        cash.q_out_zoom.task_done()
                        temp2 = temp.get(i)
                        for y in temp2:
                            after_zoom[y] = temp2[y]
                    print("done3") 
                    cash.q_out_zoom.join()
                    print("done4")
                    after_zoom_img = slising.assembling_after_blur(after_zoom,shape_zoom)
                    cash.resolution_t=resolution
                    cash.img_zoom = after_zoom_img
                    cash.clear_shape = after_zoom_img.shape
                    cash.corners_t=True
                    cash.on_grain_t=on_grain
                elif cash.resolution_t==resolution:
                    after_zoom_img=cash.img_zoom
                    print("zoom yes")


                tocZoom = time.time()
                print(tocZoom-ticZoom,"sec for zoom")


                
                
                #ADD CORNERS
                if cash.corners_t==True:
                    img_whith_corners=Not_multi_funk.add_corners(after_zoom_img)
                    prep_grain = Not_multi_funk.pre_grain(img_whith_corners,Gr_sample)
                    cash.img_corners=img_whith_corners
                    cash.prep_grain=prep_grain
                    cash.corners_t=False
                    cash.blur_rad_t=None
                elif cash.corners_t==False:
                    img_whith_corners=np.array(cash.img_corners)
                    prep_grain=np.array(cash.prep_grain)
                    print("Corners yes")
                

                
                such_mach = time.time()
                
                #REDUSE SHARPNESS
                if cash.blur_rad_t!=blur_rad or cash.halation_t!=halation or cash.blur_spred_t!=blur_spred or cash.sharp_rad_t!=sharp_rad or cash.sharp_spred_t!=sharp_spred or cash.sharp_amplif_t!=sharp_amplif or cash.bloom_rad_t!=bloom_rad or cash.bloom_Halation_t!=bloom_Halation or cash.bloom_spred_t!=bloom_spred:
                    slice_for_blur = slising.slise_for_blur(img_whith_corners)
                    dict_for_blur = slice_for_blur[0]
                    shape_blur = slice_for_blur[1]
                    for i in dict_for_blur:
                        cash.q_in_blur.put([dict_for_blur[i],blur_rad,halation,blur_spred,shape_blur,
                                            bloom_rad,bloom_Halation,bloom_spred,
                                            sharp_rad,sharp_spred,sharp_amplif])
                    
                    cash.q_in_blur.join()

                    after_blur = {}
                    for i in range(3):
                        temp = {} 
                        temp[i] = cash.q_out_blur.get()
                        cash.q_out_blur.task_done()
                        temp2 = temp.get(i)
                        for y in temp2:
                            after_blur[y] = temp2[y]                    
                    cash.q_out_blur.join()

                    img_blured = slising.assembling_after_blur(after_blur,shape_blur)
                    img_blured = img_blured[200:int(img_blured.shape[0]-200),200:int(img_blured.shape[1]-200),:]
                    cash.img_reduse=np.array(img_blured)
                    cash.blur_rad_t=blur_rad
                    cash.halation_t=halation 
                    cash.blur_spred_t=blur_spred
                    cash.bloom_rad_t=bloom_rad
                    cash.bloom_Halation_t=bloom_Halation
                    cash.bloom_spred_t=bloom_spred
                    cash.sharp_rad_t=sharp_rad
                    cash.sharp_spred_t=sharp_spred
                    cash.sharp_amplif_t=sharp_amplif
                    cash.gamma=None
                else:
                    img_blured=np.array(cash.img_reduse)
                    print("reduse yes")
                it_is = time.time()
                print(it_is - such_mach,"sec for blur")

                #prep for Lut
                if  cash.Film_t!=film or cash.camera_t!=camera: 
                    params_lut=Not_multi_funk.preper_lut(wrong,right)
                    cash.lut_film=params_lut
                    cash.Film_t=film
                    cash.camera_t=camera
                    cash.r_hue_t=None
                else:
                    params_lut=cash.lut_film
                    print("lut params yes")

                

                such_mach2 = time.time()
                #RGB MIX
                if (    cash.r_hue_t!=r_hue                     or cash.r_sut_t!=r_sut                  or cash.r_g_t!=r_g
                        or cash.r_b_t!=r_b                  or cash.g_hue_t!=g_hue                  or cash.g_sut_t!=g_sut
                        or cash.g_r_t!=g_r                  or cash.g_b_t!=g_b                      or cash.b_hue_t!=b_hue
                        or cash.b_sut_t!=b_sut              or cash.b_r_t!=b_r                      or cash.b_g_t!=b_g
                        or cash.sut_t!=sut                  or cash.END_A_PLUS_t!=end_b_plus        or cash.END_A_MINUS_t!=end_a_minus
                        or cash.END_B_PLUS_t!=end_b_plus    or cash.END_B_MINUS_t!=end_b_minus      or cash.r_scan!=r_scan
                        or cash.a_min_sut_t!=a_min_sut      or cash.a_plus_sut_t!=a_plus_sut        or cash.b_scan!=b_scan
                        or cash.b_min_sut_t!=b_min_sut      or cash.b_plus_sut_t!=b_plus_sut
                        or cash.a_m_hue_t!=a_m_hue          or cash.a_p_hue_t!=a_p_hue
                        or cash.b_m_hue_t!=b_m_hue          or cash.b_p_hue_t!=b_p_hue
                        or cash.sut_compress_t!=sat_comp    or cash.steepness_shad_t!=steepness_shad
                        or cash.width_shad_t!=width_shad    or cash.steepness_light_t!=steepness_light
                        or cash.width_light_t!=width_light  or cash.shad_a_t!=shad_a
                        or cash.shad_b_t!=shad_b            or cash.mid_a_t!=mid_a                  or cash.mid_b_t!=mid_b
                        or cash.light_a_t!=light_a          or cash.light_b_t!=light_b
                        or cash.neutral_mask_t!=neutral_mask or cash.amply_wheel_t!=amplyfy_wheel):
                    
                    right = np.array(right)


                    right[...,0] *= r_scan
                    right[...,2] *= b_scan
                    right/=np.max(right)
                    
                    
                    
                    lut_slises = slising.lut_slise()
                    lut_chunks =lut_slises[0]
                    lut_chunks_keys = lut_slises[1]
                    for i in lut_chunks_keys:
                        cash.q_in_lut.put([lut_chunks[i],params_lut,
                    r_hue,r_sut,r_g,r_b,g_hue,g_sut,g_r,g_b,b_hue,b_sut,b_r,b_g,
                    right,film,accuracy,

                    sut,end_a_plus,end_a_minus,end_b_plus,end_b_minus,a_min_sut,a_plus_sut,
                    b_min_sut,b_plus_sut,a_m_hue,a_p_hue,b_m_hue,b_p_hue,sat_comp,
                    
                    steepness_shad,width_shad,steepness_light,width_light,shad_a,shad_b,
                    mid_a,mid_b,light_a,light_b,neutral_mask,amplyfy_wheel,
                    ])

                    cash.q_in_lut.join()


                    lut_chanks_new = {}
                    for i in range(64):
                        temp = {} 
                        temp[i] = cash.q_out_lut.get()
                        cash.q_out_lut.task_done()
                        temp2 = temp.get(i)
                        for y in temp2:
                            lut_chanks_new [y] = temp2[y] 

                    cash.q_out_lut.join()

                    lut = slising.lut_assambley(lut_chanks_new ,lut_chunks_keys)

                    lut = np.array(lut,dtype=np.float32)





                    cash.lut_mix=np.array(lut)
                    cash.r_hue_t=r_hue
                    cash.r_sut_t=r_sut
                    cash.r_g_t=r_g 
                    cash.r_b_t=r_b 
                    cash.g_hue_t=g_hue
                    cash.g_sut_t=g_sut
                    cash.g_r_t=g_r
                    cash.g_b_t=g_b
                    cash.b_hue_t=b_hue
                    cash.b_sut_t=b_sut
                    cash.b_r_t=b_r 
                    cash.b_g_t=b_g
                    cash.sut_t=sut
                    cash.END_A_PLUS_t=end_a_plus
                    cash.END_A_MINUS_t=end_a_minus
                    cash.END_B_PLUS_t=end_b_plus
                    cash.END_B_MINUS_t=end_a_minus 
                    cash.a_min_sut_t=a_min_sut
                    cash.a_plus_sut_t=a_plus_sut
                    cash.b_min_sut_t=b_min_sut
                    cash.b_plus_sut_t=b_plus_sut
                    cash.a_m_hue_t=a_m_hue
                    cash.a_p_hue_t=a_p_hue
                    cash.b_m_hue_t=b_m_hue
                    cash.b_p_hue_t=b_p_hue
                    cash.sut_compress_t=sat_comp
                    cash.steepness_shad_t=steepness_shad
                    cash.width_shad_t=width_shad
                    cash.steepness_light_t=steepness_light
                    cash.width_light_t=width_light
                    cash.shad_a_t=shad_a
                    cash.shad_b_t=shad_b
                    cash.mid_a_t=mid_a
                    cash.mid_b_t=mid_b
                    cash.light_a_t=light_a
                    cash.light_b_t=light_b
                    cash.neutral_mask_t=neutral_mask 
                    cash.amply_wheel_t=amplyfy_wheel
                    cash.b_scan=b_scan
                    cash.r_scan=r_scan
                    cash.gamma = None
                else:
                    lut = cash.lut_mix
                tic = time.time()
                """if gamma!=cash.gamma or print_exp!=cash.print_exp or Wb_b!=cash.Wb_b or Wb_r!=cash.Wb_r or Wb_b2!=cash.Wb_b2 or Wb_r2!=cash.Wb_r2 or light_compr!=cash.light_compr or zone!=cash.zone or print_cont!=cash.print_cont or amplify_grain!=cash.amplify_grain or amplify_mask!=cash.amplify_mask or on_grain!=cash.on_grain:"""
                #gamma+=print_exp*1.5
                
                tictic = time.time()
                img_blured-=np.min(img_blured)
                #img_blured/=np.max(img_blured)
                img_blured[...,0]*=(1+Wb_r/10)
                img_blured[...,2]*=(1+Wb_b/10)

                img_blured=(img_blured*(411/np.average(img_blured)))
                img_blured+=(200-np.min(img_blured))
                img_blured=(np.log10(img_blured))
                img_blured=(((img_blured-np.min(img_blured))/(np.max(img_blured)-np.min(img_blured))))

                img_blured/=(np.average(img_blured)*2)
                #img_blured*=(5**((2+gamma+print_exp)/3))
                img_blured*=(5**((2+gamma)/3))
                
                img_blured+=1#(np.log(gamma+2.7)-1)

                img_blured=(np.log10(img_blured))

                img_blured-=np.min(img_blured)  

                if np.max(img_blured)>=0.9:
                    img_blured/=(np.max(img_blured)/0.9)

                

                







                img_blured=np.nan_to_num(img_blured)

                slised_grain = slising.slice_for_filmEm(np.array(prep_grain,dtype=np.float32))
                grain_slices = slised_grain[0]



                
                slised_for_film = slising.slice_for_filmEm(np.array(img_blured,dtype=np.float32))
                film_slices = slised_for_film[0]
                film_slices_keys = slised_for_film[1]
                shape = slised_for_film[2]
                threads = slised_for_film[3]
                
                if "scan" in film:
                     print_cont+=0.3
                     print_exp+=1.5

                
                for f in film_slices:
                    cash.q_in_film.put([film_slices[f],lut,Wb_b,Wb_r,Wb_b2,Wb_r2,
                                        print_exp,light_compr,zone,print_cont,gamma,
                                        Grain_curve,grain_slices[f],amplify_grain,
                                        amplify_mask,on_grain, cash.Paper
                                        ])
                
                cash.q_in_film.join()
                
                
                #cash.q_in_film.close()
                #cash.q_in_film.join_thread()            
                film_chanks = {}
                for i in range(threads+1):
                        temp = {} 
                        temp[i] = cash.q_out_film.get()
                        cash.q_out_film.task_done()
                        temp2 = temp.get(i)
                        for y in temp2:
                            film_chanks [y] = temp2[y] 
                toc = time.time()
                print(toc - tic,"sec film")    
                cash.q_out_film.join()
                toctoc = time.time()
                print(toctoc - tictic,"sec film + conversion") 
                img_film = slising.assembling_filmEm(film_chanks,film_slices_keys,shape,threads)
                #img_film/=np.max(img_film)
                img_film*=white_point
                img_film[img_film>1]=1
                cash.img_film = np.array(img_film)
                cash.gamma = gamma 
                cash.print_exp = print_exp 
                cash.Wb_b = Wb_b 
                cash.Wb_r = Wb_r
                cash.Wb_b2 = Wb_b2
                cash.Wb_r2 = Wb_r2 
                cash.light_compr = light_compr 
                cash.zone = zone 
                cash.print_cont = print_cont
                cash.amplify_grain = amplify_grain 
                cash.amplify_mask = amplify_mask 
                cash.on_grain = on_grain

                if rotate==90.0:
                     img_film = np.rot90(img_film)
                elif rotate==180.0:
                     img_film = np.rot90(img_film,2)
                elif rotate==270.0:
                     img_film = np.rot90(img_film,3)
                else:
                     pass
                
                shape_asp = img_film.shape

                if aspect!="org":
                    aspect = eval(aspect)
                    if shape_asp[0]>shape_asp[1]:
                            
                            length = int(shape_asp[1]*(aspect[0]/aspect[1]))
                            add = int((shape_asp[0]-length)/2)
                            img_film = img_film[add:add+length,:,:]
                    else:
                            length = int(shape_asp[0]*(aspect[0]/aspect[1]))
                            add = int((shape_asp[1]-length)/2)
                            img_film = img_film[:,add:add+length,:]
                #else:
                #    img_film = cash.img_film
        
            # print the difference between start 
                # and end time in milli. secs
                """print("The time of execution of above program is :",
                    (end-start) * 10**3, "ms  ","ms")"""
                print (img_film.dtype)


        
                return img_film

"""finally:
                cash.q_in_zoom.close()
                cash.q_in_zoom.join_thread()
                cash.q_out_zoom.close()
                cash.q_out_zoom.join_thread()
                cash.blur_count = 0

                cash.q_in_blur.close()
                cash.q_in_blur.join_thread()
                cash.q_out_blur.close()
                cash.q_in_blur.join_thread()
                cash.zoom_count = 0

                cash.q_in_lut.close()
                cash.q_in_lut.join_thread()
                cash.q_out_lut.close()
                cash.q_out_lut.join_thread()
                cash.lut_count = 0

                cash.q_in_film.close()
                cash.q_in_film.join_thread()
                cash.q_out_film.close()
                cash.q_out_film.join_thread()
                cash.film_count = 0
                #os.killpg(0, signal.SIGKILL) 

                for i in cash.names_zoom_process:
                                i.close()
                for i in cash.names_blur_process:
                                i.close()
                for i in cash.names_lut_process:
                                i.close()
                for i in cash.names_film_process:
                                i.close()

                print("i am cleen all")"""
                        
                
                
"""blur_rad = 1.3
halation = 1
blur_spread = 4

bloom_rad = 75
bloom_spread = 20
bloom_Halation = 1

sharp_rad = 20
sharp_spread = 10
sharp_amplif = 1


r_hue = 1
r_sut = 1
r_g = 1
r_b = 1
g_hue = 1
g_sut = 1
g_r = 1
g_b = 1
b_hue = 1
b_sut = 1
b_r = 1
b_g = 1
film = "Kodak Vision250d"
accuracy = "extrim"
sut = 1
end_a_plus = 1
end_a_minus = 1
end_b_plus = 1
end_b_minus = 1
a_min_sut = 1
a_plus_sut = 1
b_min_sut = 1
b_plus_sut = 1
a_m_hue = 0
a_p_hue = 0
b_m_hue = 0
b_p_hue = 0
sat_comp = 1
steepness_shad = 0.25
width_shad = 1
steepness_light = 0.25
width_light = 1
shad_a = 0
shad_b = 0
mid_a = 0
mid_b = 0
light_a = 0
light_b = 0
neutral_mask = -1
amplyfy_wheel = 10
Wb_b = 0
Wb_r = 0
Wb_b2 = 0
Wb_r2 = 0
print_exp = 0
light_compr = 0
zone = 0.5
print_cont = 2
gamma = 2




if __name__ == '__main__':
    img = magic(2500,"/Users/aleksejromadin/Desktop/SDIM3367.DNG","In camera WB",
                blur_rad,halation,blur_spread,bloom_rad,bloom_Halation,bloom_spread,sharp_rad,sharp_spread,sharp_amplif,
                "Sigma FP",r_hue,r_sut,r_g,r_b,g_hue,g_sut,g_r,g_b,
                b_hue,b_sut,b_r,b_g,film,accuracy,sut,end_a_plus,end_b_minus,end_a_plus,
                end_b_minus,a_min_sut,a_plus_sut,b_min_sut,b_plus_sut,a_m_hue,a_p_hue,
                b_m_hue,b_p_hue,sat_comp,steepness_shad,width_shad,steepness_light,
                width_light,shad_a,shad_b,mid_a,mid_b,light_a,light_b,
                neutral_mask,amplyfy_wheel,Wb_b,Wb_r,Wb_b2,Wb_r2,print_exp,light_compr,zone,print_cont,gamma)
    img = Image.fromarray(io.convert_bit_depth(img,"uint8")).show()"""