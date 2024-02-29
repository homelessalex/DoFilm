import flet as ft
from PIL import Image
from copy import deepcopy
import cash_flet as cash_flet
from io import BytesIO
from colour import io
import base64
import rawpy
import pickle as pkl
import numpy as np
import pathlib
import cash_flet
import Not_multi_funk
from flet.matplotlib_chart import MatplotlibChart
import Distributor
import time
from multiprocessing.pool import Pool
import os
import glob
from copy import deepcopy
import cash
import os
from cash import q_in_zoom, q_out_zoom, q_in_blur, q_out_blur , q_in_lut, q_out_lut, q_in_film , q_out_film 
from multiprocessing import active_children
import time




if __name__ == "__main__":
    def main(page: ft.Page):
        def close_all(e):
                if e.data == "close":
                    print("drenvglnregvjklrengjkwerlkdefml")
                    cash.close=True

                    all_q=[q_in_zoom,q_out_zoom,q_in_blur,q_out_blur,q_in_lut,q_out_lut,q_in_film,q_out_film]
                    for i in all_q:
                        i.join()
                        i.close()
                    active = active_children()
                    for child in active:
                        child.kill()
                    for child in active:
                        child.join()
                    active = active_children()
                    print(active)
                    os.system("pkill  python")
                    page.window_destroy()
                    os._exit(1)
                    
        
        page.title = "DoFilm"
        page.padding = 0
        page.window_maximized=True
        page.bgcolor = ft.colors.GREY_100
        page.theme = ft.theme.Theme(use_material3 = True)
        page.on_window_event = close_all
        page.window_prevent_close = True
        page.window_min_height = 600
        page.window_min_width = 1280

        shirina = page.window_width
        imgs = pkl.load(open("imgs.pkl",'rb'))


        
        main_img_gall=ft.Column(controls=[ft.Container(ft.Image(src=os.path.abspath("bg.jpeg"),fit=ft.ImageFit.SCALE_DOWN,border_radius=15),border_radius=15,padding=ft.padding.only(5,3,0,20),expand=1)],expand=3,horizontal_alignment=ft.CrossAxisAlignment.CENTER,tight=True)
        main_img_gall_out=ft.Column(controls=[ft.Container(ft.Image(src=os.path.abspath("bg.jpeg"),fit=ft.ImageFit.SCALE_DOWN,border_radius=15),border_radius=15,padding=ft.padding.only(5,3,0,20),expand=1)],expand=3,horizontal_alignment=ft.CrossAxisAlignment.CENTER,tight=True)
        #Gallery And In Controls___________#Gallery And In Controls___________#Gallery And In Controls___________#Gallery And In Controls___________#Gallery And In Controls___________
        #Gallery And In Controls___________#Gallery And In Controls___________#Gallery And In Controls___________#Gallery And In Controls___________#Gallery And In Controls___________
        #Gallery And In Controls___________#Gallery And In Controls___________#Gallery And In Controls___________#Gallery And In Controls___________#Gallery And In Controls___________
        def hover_gallery(e):
              main_img_gall.controls[0].content.src=e.control.content.src
              main_img_gall.update()

        def hover_gallery_out(e):
              main_img_gall_out.controls[0].content.src_base64=cash_flet.pillow_for_grid_string[e.control.content.key]
              main_img_gall_out.update()

        def click_gallery(e):
            imgs = pkl.load(open("imgs.pkl",'rb'))
            name = e.control.content.src[-13:-5]
            if imgs[name][2] == False:
                e.control.margin = 20
                e.control.bgcolor = ft.colors.GREY_500
                e.control.shadow=ft.BoxShadow(
                                                    spread_radius=9,
                                                    blur_radius=10,
                                                    color=ft.colors.GREY_500,
                                                    offset=ft.Offset(0, 0),
                                                    blur_style=ft.ShadowBlurStyle.SOLID,
                                                )
                
                imgs[name][2] = True
                with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)

            
            elif imgs[name][2] == True:
                e.control.margin = 3
                e.control.bgcolor = ft.colors.GREY_100
                e.control.shadow=ft.BoxShadow(
                                                    spread_radius=0,
                                                    blur_radius=0,
                                                    color=ft.colors.GREY_500,
                                                    offset=ft.Offset(0, 0),
                                                    blur_style=ft.ShadowBlurStyle.SOLID,
                                                )
                imgs[name][2] = False
                with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)
            grid_quewe_tup = grid_quewe_rer()
            grid_quewe = grid_quewe_tup[0]
            tabss.tabs[1].content.controls.remove(tabss.tabs[1].content.controls[0])
            tabss.tabs[1].content.controls.append(grid_quewe)
            """grid = grid_rer()

            tabss.tabs[0].content.controls[1].controls.remove(tabss.tabs[0].content.controls[1].controls[1])
            tabss.tabs[0].content.controls[1].controls.append(grid)"""
            tabss.update()
        def select_all(e):
            imgs = pkl.load(open("imgs.pkl",'rb'))
            for i in imgs:
                    if imgs[i][3]==False:
                        imgs[i][2]=True
            for i in range(len(imgs.keys())):
                if tabss.tabs[0].content.controls[1].controls[1].controls[i].disabled==False:
                    tabss.tabs[0].content.controls[1].controls[1].controls[i].margin = 20
                    tabss.tabs[0].content.controls[1].controls[1].controls[i].bgcolor = ft.colors.GREY_500
                    tabss.tabs[0].content.controls[1].controls[1].controls[i].shadow=ft.BoxShadow(
                                                        spread_radius=9,
                                                        blur_radius=10,
                                                        color=ft.colors.GREY_500,
                                                        offset=ft.Offset(0, 0),
                                                        blur_style=ft.ShadowBlurStyle.SOLID,
                                                    )
            
            with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)
            grid_quewe_tup = grid_quewe_rer()
            grid_quewe = grid_quewe_tup[0]
            tabss.tabs[1].content.controls.remove(tabss.tabs[1].content.controls[0])
            tabss.tabs[1].content.controls.append(grid_quewe)
            """grid = grid_rer()

            tabss.tabs[0].content.controls[1].controls.remove(tabss.tabs[0].content.controls[1].controls[1])
            tabss.tabs[0].content.controls[1].controls.append(grid) """

            tabss.update() 
        
        def unselect_all(e):
            imgs = pkl.load(open("imgs.pkl",'rb'))

            for i in imgs:
                    if imgs[i][3]==False:
                        imgs[i][2]=False
            for i in range(len(imgs.keys())):
                if tabss.tabs[0].content.controls[1].controls[1].controls[i].disabled==False:
                    tabss.tabs[0].content.controls[1].controls[1].controls[i].margin = 3
                    tabss.tabs[0].content.controls[1].controls[1].controls[i].bgcolor = ft.colors.GREY_100
                    tabss.tabs[0].content.controls[1].controls[1].controls[i].shadow=ft.BoxShadow(
                                                        spread_radius=0,
                                                        blur_radius=0,
                                                        color=ft.colors.GREY_500,
                                                        offset=ft.Offset(0, 0),
                                                        blur_style=ft.ShadowBlurStyle.SOLID,
                                                    )


            with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)
            grid_quewe_tup = grid_quewe_rer()
            grid_quewe = grid_quewe_tup[0]
            tabss.tabs[1].content.controls.remove(tabss.tabs[1].content.controls[0])
            tabss.tabs[1].content.controls.append(grid_quewe)
            """grid = grid_rer()
            

            tabss.tabs[0].content.controls[1].controls.remove(tabss.tabs[0].content.controls[1].controls[1])
            tabss.tabs[0].content.controls[1].controls.append(grid)"""
            tabss.update() 

        def grid_rer():
            imgs = pkl.load(open("imgs.pkl",'rb'))
            grid = ft.GridView(    
                            expand=1,
                            runs_count=0,
                            max_extent=300,
                            child_aspect_ratio=1.5,
                            spacing=0,
                            run_spacing=0,
                            
                            )
            for i in imgs:
                if imgs[i][2]==False and imgs[i][3]==False:
                        grid.controls.append(ft.Container(content=ft.Image(src=os.path.abspath("imgs/"+imgs[i][1]+".jpeg"),fit=ft.ImageFit.SCALE_DOWN,),
                                                margin=3,bgcolor=ft.colors.GREY_100,on_hover=hover_gallery,on_click=click_gallery,disabled=False,shadow=ft.BoxShadow(
                                                                                                                                                            spread_radius=0,
                                                                                                                                                            blur_radius=0,
                                                                                                                                                            color=ft.colors.GREY_500,
                                                                                                                                                            offset=ft.Offset(0, 0),
                                                                                                                                                            blur_style=ft.ShadowBlurStyle.SOLID,
                                                                                                                                                        )))
                elif imgs[i][2]==True and imgs[i][3]==False:
                        grid.controls.append(ft.Container(content=ft.Image(src=os.path.abspath("imgs/"+imgs[i][1]+".jpeg"),fit=ft.ImageFit.SCALE_DOWN,),
                                                margin=15,bgcolor=ft.colors.GREY_500,on_hover=hover_gallery,on_click=click_gallery,disabled=False,shadow=ft.BoxShadow(
                                                                                                                                                            spread_radius=9,
                                                                                                                                                            blur_radius=10,
                                                                                                                                                            color=ft.colors.GREY_500,
                                                                                                                                                            offset=ft.Offset(0, 0),
                                                                                                                                                            blur_style=ft.ShadowBlurStyle.SOLID,
                                                                                                                                                )))
                elif imgs[i][2]==False and imgs[i][3]==True:
                        grid.controls.append(ft.Container(content=ft.Image(src=os.path.abspath("imgs/"+imgs[i][1]+".jpeg"),fit=ft.ImageFit.SCALE_DOWN,),
                                                margin=35,bgcolor=ft.colors.GREY_800,on_hover=hover_gallery,on_click=click_gallery,disabled=True,shadow=ft.BoxShadow(
                                                                                                                                                            spread_radius=9,
                                                                                                                                                            blur_radius=10,
                                                                                                                                                            color=ft.colors.GREY_800,
                                                                                                                                                            offset=ft.Offset(0, 0),
                                                                                                                                                            blur_style=ft.ShadowBlurStyle.SOLID,
                                                                                                                                                )))
            return grid
        
        grid = grid_rer()

        def in_to_grading(e):
                imgs = pkl.load(open("imgs.pkl",'rb'))
                Preset_dict = pkl.load(open("Presets.pkl",'rb'))
                name = e.control.content.src[-13:-5]
                cash_flet.uploaded_file = imgs[name][0]
                tabss.selected_index = 2
                if imgs[name][4]!={}:
                        param_drop.options.append(ft.dropdown.Option("Last render"))
                        Preset_dict["Last render"]=imgs[name][4]
                        with open('Presets.pkl', 'wb') as fp:
                                pkl.dump(Preset_dict, fp)
                        param_drop.value = param_drop.options[-1].key
                        tabss.update()
                        apply_preset_c(e)

                else:
                    tabss.update()
                go_go(e)

        def grid_quewe_rer():
            imgs = pkl.load(open("imgs.pkl",'rb'))
            names={}
            paths={}
            for i in imgs:
                if imgs[i][2]==True and imgs[i][3]==False:
                    names[str(imgs[i][1])] = imgs[i][1]
                    paths[str(imgs[i][1])] = imgs[i][0]
            grid_quewe = ft.GridView(    
                            expand=1,
                            runs_count=0,
                            max_extent=300,
                            child_aspect_ratio=1.5,
                            spacing=0,
                            run_spacing=0,
                            controls=([ft.Container(content=ft.Image(src=os.path.abspath("imgs/"+names[i]+".jpeg"),fit=ft.ImageFit.SCALE_DOWN,),
                                                margin=3,bgcolor=ft.colors.GREY_100,on_hover=hover_gallery,on_click=in_to_grading) for i in names])
                            )
            
            return grid_quewe,names,paths
        

        grid_quewe_tup = grid_quewe_rer()
        grid_quewe = grid_quewe_tup[0]


        def is_render_it(e):
            imgs = pkl.load(open("imgs.pkl",'rb'))
            #name = e.control.content.src[-13:-5]
            name = e.control.content.key
            print(imgs[name][5])
            if imgs[name][5] == False:
                e.control.margin = 20
                e.control.bgcolor = ft.colors.GREY_500
                e.control.shadow=ft.BoxShadow(
                                                    spread_radius=9,
                                                    blur_radius=10,
                                                    color=ft.colors.GREY_500,
                                                    offset=ft.Offset(0, 0),
                                                    blur_style=ft.ShadowBlurStyle.SOLID,
                                                )
                
                imgs[name][5] = True
                with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)

            
            elif imgs[name][5] == True:
                e.control.margin = 3
                e.control.bgcolor = ft.colors.GREY_100
                e.control.shadow=ft.BoxShadow(
                                                    spread_radius=0,
                                                    blur_radius=0,
                                                    color=ft.colors.GREY_500,
                                                    offset=ft.Offset(0, 0),
                                                    blur_style=ft.ShadowBlurStyle.SOLID,
                                                )
                imgs[name][5] = False
                with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)

            grid = grid_rer()
            tabss.tabs[0].content.controls[1].controls.remove(tabss.tabs[0].content.controls[1].controls[1])
            tabss.tabs[0].content.controls[1].controls.append(grid)
            grid_quewe_tup = grid_quewe_rer()
            grid_quewe = grid_quewe_tup[0]
            tabss.tabs[1].content.controls.remove(tabss.tabs[1].content.controls[0])
            tabss.tabs[1].content.controls.append(grid_quewe)
            tabss.tabs[1].update()
            tabss.tabs[3].update()



        def is_render_select_all(e):
            imgs = pkl.load(open("imgs.pkl",'rb'))

            temp = 0
            for i in imgs:
                if imgs[i][3]==True:
                    temp+=1
                    imgs[i][5]=True

            for i in range(temp):
                tabss.tabs[3].content.controls[1].controls[1].controls[i].margin = 20
                tabss.tabs[3].content.controls[1].controls[1].controls[i].bgcolor = ft.colors.GREY_500
                tabss.tabs[3].content.controls[1].controls[1].controls[i].shadow=ft.BoxShadow(
                                                    spread_radius=9,
                                                    blur_radius=10,
                                                    color=ft.colors.GREY_500,
                                                    offset=ft.Offset(0, 0),
                                                    blur_style=ft.ShadowBlurStyle.SOLID,
                                                )
            
            with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)  

            grid = grid_rer()
            tabss.tabs[0].content.controls[1].controls.remove(tabss.tabs[0].content.controls[1].controls[1])
            tabss.tabs[0].content.controls[1].controls.append(grid)
            grid_quewe_tup = grid_quewe_rer()
            grid_quewe = grid_quewe_tup[0]
            tabss.tabs[1].content.controls.remove(tabss.tabs[1].content.controls[0])
            tabss.tabs[1].content.controls.append(grid_quewe)
            tabss.update() 




        def is_render_unselect_all(e):
            imgs = pkl.load(open("imgs.pkl",'rb'))
            temp=0
            for i in imgs:
                if imgs[i][3]==True:
                    temp+=1
                    imgs[i][5]=False
            for i in range(temp):
                tabss.tabs[3].content.controls[1].controls[1].controls[i].margin = 3
                tabss.tabs[3].content.controls[1].controls[1].controls[i].bgcolor = ft.colors.GREY_100
                tabss.tabs[3].content.controls[1].controls[1].controls[i].shadow=ft.BoxShadow(
                                                    spread_radius=0,
                                                    blur_radius=0,
                                                    color=ft.colors.GREY_500,
                                                    offset=ft.Offset(0, 0),
                                                    blur_style=ft.ShadowBlurStyle.SOLID,
                                                )


            with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)

            grid = grid_rer()
            tabss.tabs[0].content.controls[1].controls.remove(tabss.tabs[0].content.controls[1].controls[1])
            tabss.tabs[0].content.controls[1].controls.append(grid)
            grid_quewe_tup = grid_quewe_rer()
            grid_quewe = grid_quewe_tup[0]
            tabss.tabs[1].content.controls.remove(tabss.tabs[1].content.controls[0])
            tabss.tabs[1].content.controls.append(grid_quewe)
            tabss.update()

        def grid_quewe_out_rer():
            imgs = pkl.load(open("imgs.pkl",'rb'))
            names={}
            paths={}
            for i in imgs:
                if imgs[i][2]==False and imgs[i][3]==True:
                    names[str(imgs[i][1])] = imgs[i][1]
                    paths[str(imgs[i][1])] = imgs[i][0]
            grid_quewe = ft.GridView(    
                            expand=1,
                            runs_count=0,
                            max_extent=300,
                            child_aspect_ratio=1.5,
                            spacing=0,
                            run_spacing=0,
                            
                            )
            for i in names:

                pillow_for_grid = Image.open(os.path.abspath("rendered/"+imgs[i][1]+".jpeg"))
                buff_out = BytesIO()
                pillow_for_grid.save(buff_out, format="JPEG")
                cash_flet.pillow_for_grid_string[imgs[i][1]]=base64.b64encode(buff_out.getvalue()).decode("utf-8")
                if imgs[i][5] == False:

 
                    grid_quewe.controls.append(ft.Container(content=ft.Image(src_base64=cash_flet.pillow_for_grid_string[imgs[i][1]],fit=ft.ImageFit.SCALE_DOWN,key=imgs[i][1]),
                                                margin=3,bgcolor=ft.colors.GREY_100,on_hover=hover_gallery_out,on_click=is_render_it,shadow=ft.BoxShadow(
                                                                                                                                                            spread_radius=0,
                                                                                                                                                            blur_radius=0,
                                                                                                                                                            color=ft.colors.GREY_500,
                                                                                                                                                            offset=ft.Offset(0, 0),
                                                                                                                                                            blur_style=ft.ShadowBlurStyle.SOLID,
                                                                                                                                                        )))
                elif imgs[i][5]==True:
                        grid_quewe.controls.append(ft.Container(content=ft.Image(src=os.path.abspath("rendered/"+imgs[i][1]+".jpeg"),fit=ft.ImageFit.SCALE_DOWN,key=imgs[i][1]),
                                                margin=15,bgcolor=ft.colors.GREY_500,on_hover=hover_gallery_out,on_click=is_render_it,shadow=ft.BoxShadow(
                                                                                                                                                            spread_radius=9,
                                                                                                                                                            blur_radius=10,
                                                                                                                                                            color=ft.colors.GREY_500,
                                                                                                                                                            offset=ft.Offset(0, 0),
                                                                                                                                                            blur_style=ft.ShadowBlurStyle.SOLID,
                                                                                                                                                )))
            
            return grid_quewe,names,paths
        grid_quewe_out_tup = grid_quewe_out_rer()
        grid_quewe_out = grid_quewe_out_tup[0]


        prog_bar = ft.ProgressBar(expand=5)
        prog_bar.value = 0


        prog_bar_out = ft.ProgressBar(expand=True)
        prog_bar_out.value = 0
        def prog_bar_update():
            prog_bar.value = ((cash_flet.count_bar/cash_flet.length_bar))
            prog_bar.update()
        


        def on_dialog_result(e: ft.FilePickerResultEvent):

            global grid

            

                
            if e.path!=None:
                
                files = glob.glob(os.path.abspath('imgs/*'))
                for f in files:
                        os.remove(f)
                files = glob.glob(os.path.abspath('rendered/*'))
                for f in files:
                        os.remove(f)




                path = e.path
                path = pathlib.Path(path)
                dng = list(path.glob("*"))
                count=0


                all={}
                for i in dng:
                    all[str(i)]=str(i)
                    count+=1
                all = {key:all[key] for key in sorted(all.keys())}
                #cash_flet.imgs = {}
                
                count_bar = 0
                length_bar = len(all)
                tic = time.time()
                imgs = {}
                prog_bar.value = None
                prog_bar.update()
                with Pool(processes=4) as pool:
                            it = pool.map(func=Not_multi_funk.read_raw,iterable=[all[i] for i in all])

                for q in it:
                        if q!=None:
                            imgs[q[1]]=q
                toc = time.time()
                print(toc-tic,"Total time")
                imgs = {key:imgs[key] for key in sorted(imgs.keys())}
                """for i in all:
                        #print(all[i])
                        
                        try:    
                                tic = time.time()
                                with rawpy.imread(all[i]) as raw:
                                    rgb = raw.postprocess(output_color=rawpy.ColorSpace(1), half_size=True,
                                                    use_camera_wb=True, highlight_mode=rawpy.HighlightMode(0),demosaic_algorithm=rawpy.DemosaicAlgorithm(0),
                                                    output_bps=8,  no_auto_scale=False, no_auto_bright=False,auto_bright_thr=0.000000001,fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(0),
                                                    gamma=(2.222, 4.5))
                                    
                                    rgb=np.dstack((rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]))

                                    coef=1920/np.max(rgb.shape)
                                    rgb=Image.fromarray(rgb)
                                    rgb=rgb.resize((int(rgb.width*coef),int(rgb.height*coef)),resample=Image.Resampling.NEAREST)
                                    name = str(all[i])
                                    name = name[-12:-4]
                                    rgb.save("imgs/"+name+".jpeg", format="JPEG", quality=75)
                                    imgs[name]=[all[i],name,False]
                                    count_bar+=1

                                    prog_bar.value = ((count_bar/length_bar))

                                    prog_bar.update()
                                toc = time.time()
                                print(toc-tic,"per file")

                        except: # Replace Exception with something more specific.
                                    count_bar+=1 
                                    print("oouuupsss")"""
                
                with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)

                prog_bar.value = 1
                prog_bar.update()
                

                grid = grid_rer()

                tabss.tabs[0].content.controls[1].controls.remove(tabss.tabs[0].content.controls[1].controls[1])
                tabss.tabs[0].content.controls[1].controls.append(grid)
                grid_quewe_tup = grid_quewe_rer()
                grid_quewe = grid_quewe_tup[0]
                tabss.tabs[1].content.controls.remove(tabss.tabs[1].content.controls[0])
                tabss.tabs[1].content.controls.append(grid_quewe)
                grid_quewe_out_tup = grid_quewe_out_rer()
                grid_quewe_out = grid_quewe_out_tup[0]
                tabss.tabs[3].content.controls[1].controls.remove(tabss.tabs[3].content.controls[1].controls[1])
                tabss.tabs[3].content.controls[1].controls.append(grid_quewe_out)
                

                page.update()
            #except:
            #      pass 


        def on_dialog_result_out(e: ft.FilePickerResultEvent):
            imgs = pkl.load(open("imgs.pkl",'rb'))
            if e.path!=None:
                length = 0
                for i in imgs:
                    if imgs[i][5] == True:
                                length+=1
                counter = 0
                for i in imgs:
                    if imgs[i][5] == True:
                            params = imgs[i][4]
                            params["cash_flet.resolution"] = cash_flet.save_resolution

                            p_image = Distributor.magic(
                                            params["cash_flet.resolution"]  ,        params["cash_flet.uploaded_file"]  ,     params["cash_flet.Wb"]  ,            params["cash_flet.blur_rad"],
                                            params["cash_flet.halation"]  ,          params["cash_flet.blur_spred"]  ,        params["cash_flet.bloom_rad"]  ,     params["cash_flet.bloom_Halation"],
                                            params["cash_flet.bloom_spred"]  ,       params["cash_flet.sharp_rad"]  ,         params["cash_flet.sharp_spred"]  ,   params["cash_flet.sharp_amplif"],
                                            
                                            params["cash_flet.camera"]  ,            params["cash_flet.r_hue"]  ,             params["cash_flet.r_sut"]  ,         params["cash_flet.r_g"]  ,           params["cash_flet.r_b"],
                                            params["cash_flet.g_hue"]  ,             params["cash_flet.g_sut"]  ,             params["cash_flet.g_r"]  ,           params["cash_flet.g_b"],
                                            params["cash_flet.b_hue"]  ,             params["cash_flet.b_sut"]  ,             params["cash_flet.b_r"]  ,           params["cash_flet.b_g"],
                                            params["cash_flet.film"],"None"  ,       params["cash_flet.sut"],
                                            params["cash_flet.end_a_plus"]  ,        params["cash_flet.end_a_minus"]  ,       params["cash_flet.end_b_plus"]  ,    params["cash_flet.end_b_minus"],
                                            params["cash_flet.sut_a_minus"]  ,       params["cash_flet.sut_a_plus"]  ,        params["cash_flet.sut_b_minus"]  ,   params["cash_flet.sut_b_plus"],
                                            0,0,0,0  ,                               params["cash_flet.sut_compr"],
                                            params["cash_flet.steepness_shad"]  ,    params["cash_flet.width_shad"]  ,        params["cash_flet.steepness_light"]  , params["cash_flet.width_light"],
                                            params["cash_flet.shad_a"]  ,            params["cash_flet.shad_b"]  ,            params["cash_flet.mid_a"]  ,         params["cash_flet.mid_b"]  ,         params["cash_flet.light_a"]  ,   params["cash_flet.light_b"],
                                            params["cash_flet.neutral_mask"]  ,      params["cash_flet.amplyfy_wheel"]  ,     params["cash_flet.wbb"]  ,           params["cash_flet.wbr"]  , 0,0,      params["cash_flet.print_exp"], 
                                            params["cash_flet.light_compr"]  ,       params["cash_flet.zone"]  ,              params["cash_flet.print_cont"]  ,    params["cash_flet.gamma"]  ,         params["cash_flet.wbr_scan"]  ,  params["cash_flet.wbb_scan"],
                                            params["cash_flet.amplify_grain"]  ,     params["cash_flet.amplify_mask"]  ,      params["cash_flet.on_grain"],        params["cash_flet.white_point"], is_half=False)
                            
                            
                            p_image = Image.fromarray(io.convert_bit_depth(p_image,"uint8"))
                            for_icc = Image.open(os.path.abspath("IMG_4799.jpg"))
                            icc = for_icc.info.get('icc_profile')
                            p_image.save(str(e.path)+"/"+str(imgs[i][1])+".jpeg", format="JPEG", quality=100, icc_profile=icc)
                            counter+=1
                            prog_bar_out.value = counter/length
                            prog_bar_out.update()
                            grid_quewe_out_tup = grid_quewe_out_rer()
                            grid_quewe_out = grid_quewe_out_tup[0]
                            tabss.tabs[3].content.controls[1].controls.remove(tabss.tabs[3].content.controls[1].controls[1])
                            tabss.tabs[3].content.controls[1].controls.append(grid_quewe_out)
                            tabss.update()




        file_picker = ft.FilePicker(on_result=on_dialog_result)
        file_picker_out = ft.FilePicker(on_result=on_dialog_result_out)

        #COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________
        #COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________
        #COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________
        #COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________
        #COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________
        #COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________
        #COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________#COLOUR CONTROLs___________________________

        
        def preset_dict_rer(e):
            Preset_dict = pkl.load(open("Presets.pkl",'rb'))

            param_drop = ft.Dropdown(label="Presets",on_change=cash_flet.camera_c ,focused_border_color=ft.colors.GREY_800,focused_bgcolor=ft.colors.GREY_100,focused_color=ft.colors.GREY_800)
            
            for z in Preset_dict.keys():
                    param_drop.options.append(ft.dropdown.Option(z))  
            param_drop.value = param_drop.options[0].key
            return param_drop
        param_drop = preset_dict_rer(1)
        
        
        def save_preset_c(e):
                
                Preset_dict = pkl.load(open("Presets.pkl",'rb'))

                params = {  "cash_flet.resolution": cash_flet.resolution,               
                        "cash_flet.Wb" : cash_flet.Wb,                              "cash_flet.blur_rad": cash_flet.blur_rad,
                        "cash_flet.halation": cash_flet.halation,                   "cash_flet.blur_spred": cash_flet.blur_spred,
                        "cash_flet.bloom_rad": cash_flet.bloom_rad,                 "cash_flet.bloom_Halation": cash_flet.bloom_Halation,
                        "cash_flet.bloom_spred": cash_flet.bloom_spred,             "cash_flet.sharp_rad": cash_flet.sharp_rad,
                        "cash_flet.sharp_spred": cash_flet.sharp_spred,             "cash_flet.sharp_amplif": cash_flet.sharp_amplif,
                        "cash_flet.camera": cash_flet.camera,                       "cash_flet.r_hue": cash_flet.r_hue,
                        "cash_flet.r_sut": cash_flet.r_sut,                         "cash_flet.r_g": cash_flet.r_g,
                        "cash_flet.r_b": cash_flet.r_b,                             "cash_flet.g_hue": cash_flet.g_hue,
                        "cash_flet.g_sut": cash_flet.g_sut,                         "cash_flet.g_r": cash_flet.g_r,
                        "cash_flet.g_b": cash_flet.g_b,                             "cash_flet.b_hue": cash_flet.b_hue,
                        "cash_flet.b_sut": cash_flet.b_sut,                         "cash_flet.b_r": cash_flet.b_r,
                        "cash_flet.b_g": cash_flet.b_g,                             "cash_flet.film" : cash_flet.film,
                        "None" : "None",                                            "cash_flet.sut": cash_flet.sut,
                        "cash_flet.end_a_plus": cash_flet.end_a_plus,               "cash_flet.end_a_minus": cash_flet.end_a_minus,
                        "cash_flet.end_b_plus": cash_flet.end_b_plus,               "cash_flet.end_b_minus": cash_flet.end_b_minus,
                        "cash_flet.sut_a_minus": cash_flet.sut_a_minus,             "cash_flet.sut_a_plus": cash_flet.sut_a_plus,
                        "cash_flet.sut_b_minus": cash_flet.sut_b_minus,             "cash_flet.sut_b_plus": cash_flet.sut_b_plus,
                        "0": 0,                                      "cash_flet.sut_compr": cash_flet.sut_compr,                                 "cash_flet.steepness_shad": cash_flet.steepness_shad,       "cash_flet.width_shad": cash_flet.width_shad,
                        "cash_flet.steepness_light": cash_flet.steepness_light,     "cash_flet.width_light": cash_flet.width_light,
                        "cash_flet.shad_a": cash_flet.shad_a,                       "cash_flet.shad_b": cash_flet.shad_b,
                        "cash_flet.mid_a": cash_flet.mid_a,                         "cash_flet.mid_b": cash_flet.mid_b,
                        "cash_flet.light_a": cash_flet.light_a,                     "cash_flet.light_b": cash_flet.light_b,
                        "cash_flet.neutral_mask": cash_flet.neutral_mask,           "cash_flet.amplyfy_wheel": cash_flet.amplyfy_wheel,
                        "cash_flet.wbb": cash_flet.wbb,                             "cash_flet.wbr": cash_flet.wbr,
                        "cash_flet.print_exp": cash_flet.print_exp,                 "cash_flet.light_compr": cash_flet.light_compr,
                        "cash_flet.zone": cash_flet.zone,                           "cash_flet.print_cont": cash_flet.print_cont,
                        "cash_flet.gamma": cash_flet.gamma,                         "cash_flet.wbr_scan": cash_flet.wbr_scan,
                        "cash_flet.wbb_scan": cash_flet.wbb_scan,                   "cash_flet.amplify_grain": cash_flet.amplify_grain,
                        "cash_flet.amplify_mask": cash_flet.amplify_mask,           "cash_flet.on_grain": cash_flet.on_grain,
                        "cash_flet.white_point": cash_flet.white_point}
                name_t = None
                if preset_name.value!="":
                     name_t = preset_name.value
                else:
                     name_t = "Unnamed preset"
                Preset_dict[name_t] = params
                with open('Presets.pkl', 'wb') as fp:
                    pkl.dump(Preset_dict, fp)
                param_drop = preset_dict_rer(1)
                Presets = ft.Column(controls=[ft.ExpansionTile(title=ft.Text("Presets"),controls=[ft.Container(content = ft.Column(controls=[ 
                                                                                                        param_drop,apply_preset,save_preset,del_preset_button
                                                                                                            ],horizontal_alignment=ft.CrossAxisAlignment.CENTER),border=ft.border.all(1, ft.colors.GREY_800),border_radius=5,padding=10,margin=10 )
                                                                                        ])],)
                tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls.remove(tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls[0])
                tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls.append(Presets.controls[0])

                
                tabss.update()
        def del_preset_c(e):
                Preset_dict = pkl.load(open("Presets.pkl",'rb'))
                if tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls[0].controls[0].content.controls[0].value!= 'Def':
                    del Preset_dict[tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls[0].controls[0].content.controls[0].value]

                    with open('Presets.pkl', 'wb') as fp:
                        pkl.dump(Preset_dict, fp)
                    param_drop = preset_dict_rer(1)
                    Presets = ft.Column(controls=[ft.ExpansionTile(title=ft.Text("Presets"),initially_expanded=True,controls=[ft.Container(content = ft.Column(controls=[ 
                                                                                                    param_drop,apply_preset,save_preset,del_preset_button
                                                                                                        ],horizontal_alignment=ft.CrossAxisAlignment.CENTER),border=ft.border.all(1, ft.colors.GREY_800),border_radius=5,padding=10,margin=10 )
                                                                                    ])],)
                    tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls.remove(tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls[0])
                    tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls.append(Presets.controls[0])

                    
                    tabss.update()
        def apply_preset_c(e):
            Preset_dict = pkl.load(open("Presets.pkl",'rb'))
            params = Preset_dict[tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls[0].controls[0].content.controls[0].value]
            cash_flet.resolution=params["cash_flet.resolution"]
            cash_flet.Wb=params["cash_flet.Wb"]
            cash_flet.blur_rad=params["cash_flet.blur_rad"]
            cash_flet.halation=params["cash_flet.halation"]
            cash_flet.blur_spred=params["cash_flet.blur_spred"]
            cash_flet.bloom_rad=params["cash_flet.bloom_rad"]
            cash_flet.bloom_Halation=params["cash_flet.bloom_Halation"]
            cash_flet.bloom_spred=params["cash_flet.bloom_spred"]
            cash_flet.sharp_rad=params["cash_flet.sharp_rad"]
            cash_flet.sharp_spred=params["cash_flet.sharp_spred"]
            cash_flet.sharp_amplif=params["cash_flet.sharp_amplif"]
            cash_flet.r_b=params["cash_flet.r_b"]
            cash_flet.print_exp=params["cash_flet.print_exp"]
            cash_flet.wbr_scan=params["cash_flet.wbr_scan"]
            cash_flet.wbb_scan=params["cash_flet.wbb_scan"]
            cash_flet.camera=params["cash_flet.camera"]
            cash_flet.r_hue=params["cash_flet.r_hue"]
            cash_flet.r_sut=params["cash_flet.r_sut"]
            cash_flet.r_g=params["cash_flet.r_g"]           
            cash_flet.g_hue=params["cash_flet.g_hue"]
            cash_flet.g_sut=params["cash_flet.g_sut"]
            cash_flet.g_r=params["cash_flet.g_r"]
            cash_flet.g_b=params["cash_flet.g_b"]
            cash_flet.b_hue=params["cash_flet.b_hue"]
            cash_flet.b_sut=params["cash_flet.b_sut"]
            cash_flet.b_r=params["cash_flet.b_r"]
            cash_flet.b_g=params["cash_flet.b_g"]
            cash_flet.film=params["cash_flet.film"]
            cash_flet.sut=params["cash_flet.sut"]
            cash_flet.light_a=params["cash_flet.light_a"]
            cash_flet.light_b=params["cash_flet.light_b"]
            cash_flet.end_a_plus=params["cash_flet.end_a_plus"]
            cash_flet.end_a_minus=params["cash_flet.end_a_minus"]
            cash_flet.end_b_plus=params["cash_flet.end_b_plus"]
            cash_flet.end_b_minus=params["cash_flet.end_b_minus"]
            cash_flet.sut_a_minus=params["cash_flet.sut_a_minus"]
            cash_flet.sut_a_plus=params["cash_flet.sut_a_plus"]
            cash_flet.sut_b_minus=params["cash_flet.sut_b_minus"]
            cash_flet.sut_b_plus=params["cash_flet.sut_b_plus"]
            cash_flet.sut_compr=params["cash_flet.sut_compr"]
            cash_flet.steepness_shad=params["cash_flet.steepness_shad"]
            cash_flet.width_shad=params["cash_flet.width_shad"]
            cash_flet.steepness_light=params["cash_flet.steepness_light"]
            cash_flet.width_light=params["cash_flet.width_light"]
            cash_flet.shad_a=params["cash_flet.shad_a"]
            cash_flet.shad_b=params["cash_flet.shad_b"]
            cash_flet.mid_a=params["cash_flet.mid_a"]
            cash_flet.mid_b=params["cash_flet.mid_b"]              
            cash_flet.neutral_mask=params["cash_flet.neutral_mask"]
            cash_flet.amplyfy_wheel=params["cash_flet.amplyfy_wheel"]
            cash_flet.wbb=params["cash_flet.wbb"]
            cash_flet.wbr=params["cash_flet.wbr"]       
            cash_flet.light_compr=params["cash_flet.light_compr"]
            cash_flet.zone=params["cash_flet.zone"]
            cash_flet.print_cont=params["cash_flet.print_cont"]
            cash_flet.gamma=params["cash_flet.gamma"]        
            cash_flet.amplify_grain=params["cash_flet.amplify_grain"]
            cash_flet.amplify_mask=params["cash_flet.amplify_mask"]
            cash_flet.on_grain=params["cash_flet.on_grain"]
            cash_flet.white_point=params["cash_flet.white_point"]
            resolution.value = cash_flet.resolution
            wb.value = cash_flet.Wb
            blur_rad_sld.value = cash_flet.blur_rad
            blur_rad_txt.value = cash_flet.blur_rad

            halation_sld.value = cash_flet.halation
            halation_txt.value = cash_flet.halation
            
            blur_spred_sld.value = cash_flet.blur_spred
            blur_spred_txt.value = cash_flet.blur_spred

            bloom_rad_sld.value = cash_flet.bloom_rad
            bloom_rad_txt.value = cash_flet.bloom_rad

            bloom_Halation_sld.value = cash_flet.bloom_Halation
            bloom_Halation_txt.value = cash_flet.bloom_Halation

            bloom_spred_sld.value = cash_flet.bloom_spred
            bloom_spred_txt.value = cash_flet.bloom_spred

            sharp_rad_sld.value = cash_flet.sharp_rad
            sharp_rad_txt.value = cash_flet.sharp_rad

            sharp_spred_sld.value = cash_flet.sharp_spred
            sharp_spred_txt.value = cash_flet.sharp_spred

            sharp_amplif_sld.value = cash_flet.sharp_amplif
            sharp_amplif_txt.value = cash_flet.sharp_amplif

            cam_drop.value = cash_flet.camera

            r_hue_sld.value = cash_flet.r_hue
            r_hue_txt.value = cash_flet.r_hue

            r_sut_sld.value = cash_flet.r_sut
            r_sut_txt.value = cash_flet.r_sut

            r_g_sld.value = cash_flet.r_g
            r_g_txt.value = cash_flet.r_g

            r_b_sld.value = cash_flet.r_b
            r_b_txt.value = cash_flet.r_b

            g_hue_sld.value = cash_flet.g_hue
            g_hue_txt.value = cash_flet.g_hue

            g_sut_sld.value = cash_flet.g_sut
            g_sut_txt.value = cash_flet.g_sut

            g_r_sld.value = cash_flet.g_r
            g_r_txt.value = cash_flet.g_r

            g_b_sld.value = cash_flet.g_b
            g_b_txt.value = cash_flet.g_b

            b_hue_sld.value = cash_flet.b_hue
            b_hue_txt.value = cash_flet.b_hue

            b_sut_sld.value = cash_flet.b_sut
            b_sut_txt.value = cash_flet.b_sut

            b_r_sld.value = cash_flet.b_r
            b_r_txt.value = cash_flet.b_r

            b_g_sld.value = cash_flet.b_g
            b_g_txt.value = cash_flet.b_g

            film_drop.value = cash_flet.film

            sut_sld.value = cash_flet.sut
            sut_txt.value = cash_flet.sut

            end_a_plus_sld.value = cash_flet.end_a_plus
            end_a_plus_txt.value = cash_flet.end_a_plus

            end_a_minus_sld.value = cash_flet.end_a_minus
            end_a_minus_txt.value = cash_flet.end_a_minus

            end_b_plus_sld.value = cash_flet.end_b_plus
            end_b_plus_txt.value = cash_flet.end_b_plus

            end_b_minus_sld.value = cash_flet.end_b_minus
            end_b_minus_txt.value = cash_flet.end_b_minus

            sut_a_minus_sld.value = cash_flet.sut_a_minus
            sut_a_minus_txt.value = cash_flet.sut_a_minus

            sut_a_plus_sld.value = cash_flet.sut_a_plus
            sut_a_plus_txt.value = cash_flet.sut_a_plus

            sut_b_minus_sld.value = cash_flet.sut_b_minus
            sut_b_minus_txt.value = cash_flet.sut_b_minus

            sut_b_plus_sld.value = cash_flet.sut_b_plus
            sut_b_plus_txt.value = cash_flet.sut_b_plus

            sut_compr_sld.value = cash_flet.sut_compr
            sut_compr_txt.value = cash_flet.sut_compr

            steepness_shad_sld.value = cash_flet.steepness_shad
            steepness_shad_txt.value = cash_flet.steepness_shad

            width_shad_sld.value = cash_flet.width_shad
            width_shad_txt.value = cash_flet.width_shad

            steepness_light_sld.value = cash_flet.steepness_light
            steepness_light_txt.value = cash_flet.steepness_light

            width_light_sld.value = cash_flet.width_light
            width_light_txt.value = cash_flet.width_light

            shad_a_sld.value = cash_flet.shad_a
            shad_a_txt.value = cash_flet.shad_a

            shad_b_sld.value = cash_flet.shad_b
            shad_b_txt.value = cash_flet.shad_b

            mid_a_sld.value = cash_flet.mid_a
            mid_a_txt.value = cash_flet.mid_a

            mid_b_sld.value = cash_flet.mid_b
            mid_b_txt.value = cash_flet.mid_b

            light_a_sld.value = cash_flet.light_a
            light_a_txt.value = cash_flet.light_a

            light_b_sld.value = cash_flet.light_b
            light_b_txt.value = cash_flet.light_b

            neutral_mask_sld.value = cash_flet.neutral_mask
            neutral_mask_txt.value = cash_flet.neutral_mask

            amplyfy_wheel_sld.value = cash_flet.amplyfy_wheel
            amplyfy_wheel_txt.value = cash_flet.amplyfy_wheel

            wbb_sld.value = cash_flet.wbb
            wbb_txt.value = cash_flet.wbb

            wbr_sld.value = cash_flet.wbr
            wbr_txt.value = cash_flet.wbr

            print_exp_sld.value = cash_flet.print_exp
            print_exp_txt.value = cash_flet.print_exp

            light_compr_sld.value = cash_flet.light_compr
            light_compr_txt.value = cash_flet.light_compr

            zone_sld.value = cash_flet.zone
            zone_txt.value = cash_flet.zone

            print_cont_sld.value = cash_flet.print_cont
            print_cont_txt.value = cash_flet.print_cont

            gamma_sld.value = cash_flet.gamma
            gamma_txt.value = cash_flet.gamma

            wbr_scan_sld.value = cash_flet.wbr_scan
            wbr_scan_txt.value = cash_flet.wbr_scan

            wbb_scan_sld.value = cash_flet.wbb_scan
            wbb_scan_txt.value = cash_flet.wbb_scan

            amplify_grain_sld.value = cash_flet.amplify_grain
            amplify_grain_txt.value = cash_flet.amplify_grain

            amplify_mask_sld.value = cash_flet.amplify_mask
            amplify_mask_txt.value = cash_flet.amplify_mask

            on_grain.value = cash_flet.on_grain
            white_point_sld.value = cash_flet.white_point
            white_point_txt.value = cash_flet.white_point

            tabss.update()
            go_go(e)
        
        preset_name = ft.TextField(label="Name of preset:")

        apply_preset = ft.ElevatedButton(text="Apply preset",on_click=apply_preset_c,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800))
        save_preset_button = ft.ElevatedButton(text="Save preset",on_click=save_preset_c,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800))
        del_preset_button = ft.ElevatedButton(text="Delete preset",on_click=del_preset_c,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800))
        save_preset = ft.ExpansionTile(title=ft.Text("Save current"),controls=[
                                                                                preset_name,save_preset_button
             
                                                                                ])


        Presets = ft.Column(controls=[ft.ExpansionTile(title=ft.Text("Presets"),controls=[ft.Container(content = ft.Column(controls=[ 
                                                                                                param_drop,apply_preset,save_preset,del_preset_button
                                                                                                    ],horizontal_alignment=ft.CrossAxisAlignment.CENTER),border=ft.border.all(1, ft.colors.GREY_800),border_radius=5,padding=10,margin=10 )
                                                                                ])],)
        def film_drop_c (e):
                        cash_flet.film = e.control.value   
                        go_go(e)      
        def cam_drop_c (e):
                        cash_flet.camera = e.control.value   
                        go_go(e)                             


        #=====================CAM_DROP==============================================
        cam_drop = ft.Dropdown(label="Camera",on_change=cam_drop_c ,focused_border_color=ft.colors.GREY_800,focused_bgcolor=ft.colors.GREY_100,focused_color=ft.colors.GREY_800)
        for x in cash_flet.cameras1:
                cam_drop.options.append(ft.dropdown.Option(x))
        cam_drop.value = cam_drop.options[0].key
        #=======================FILM_DROP============================================
        film_drop = ft.Dropdown(label="Film",on_change=film_drop_c,focused_border_color=ft.colors.GREY_800,focused_bgcolor=ft.colors.GREY_100,focused_color=ft.colors.GREY_800)
        for x in cash_flet.Films1:
                film_drop.options.append(ft.dropdown.Option(x))
        film_drop.value = film_drop.options[0].key
        #===============WB_CAMERA_OR_AUTO====================================================
        wb = ft.RadioGroup(ft.Row([
                                        ft.Radio(value="In camera WB",label="In camera WB",active_color=ft.colors.GREY_800),
                                        ft.Radio(value="Auto WB",label="Auto WB",active_color=ft.colors.GREY_800),
                                    ]),
                            on_change=cash_flet.WB_c,value="In camera WB",     
                            )
        #===================RESOLUTION================================================
        def resolution_c(e):

                cash_flet.resolution = e.control.value
                go_go(e)

        resolution = ft.Slider(min=1000, max=3000, divisions=20, label="{value}px", on_change_end=resolution_c,value=1600,active_color=ft.colors.GREY_500)
        Params = ft.ExpansionTile(title=ft.Text("Params"),controls=[wb,
                                                                    ft.Text("Viewer resolution:"),
                                                                    resolution

                                                                    ])
        
        #===================RESOLUTION================================================
        def save_resolution_c(e):

                cash_flet.save_resolution = e.control.value

        save_resolution = ft.Slider(min=1000, max=6000, divisions=20, label="{value}px", on_change_end=save_resolution_c,value=3000,active_color=ft.colors.GREY_500)
  
        #COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________
        #COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________
        #COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________
        #COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________
        #COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________
        #COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________
        #COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________#COLOUR_______________
        def sldr_hover(e):
            color = e.control.content.controls[0].active_color
            gr700 = color[0:-3]+"700"
            gr300 = color[0:-3]+"300"
            e.control.content.controls[0].active_color=gr700 if e.data == "true" else   gr300
            cash_flet.selected = e.control.key if e.data == "true" else None  
            print(cash_flet.selected)          
            e.control.update()        
        def on_hover(e):
                
                e.control.bgcolor =  ft.colors.GREY_900 if e.data == "true" else   ft.colors.WHITE
                e.control.update()

        def out_to_in(e):
                                
                e.control.text = "       Long press       " if e.data == "true" else   "Selested| Out -> In "
                e.control.update()
        def on_hover_add(e):
                
                e.control.bgcolor =  ft.colors.GREY_900 if e.data == "true" else   ft.colors.GREY_100
                e.control.update()
        def go_go(e):
            if cash_flet.uploaded_file!=None:
                tic = time.time()
                sldrs=[resolution,wb,blur_rad,halation,blur_spred,bloom_rad,bloom_Halation,bloom_spred,sharp_rad,sharp_spred,sharp_amplif,
                cam_drop,r_hue,r_sut,r_g,r_b,g_hue,g_sut,g_r,g_b,b_hue,b_sut,b_r,b_g,film_drop,sut,end_a_plus,end_a_minus,end_b_plus,end_b_minus,sut_a_minus,sut_a_plus,sut_b_minus,sut_b_plus,sut_compr,
                steepness_shad,width_shad,steepness_light,width_light,shad_a,shad_b,mid_a,mid_b,light_a,light_b,neutral_mask,amplyfy_wheel,wbb,wbr,print_exp,light_compr,zone,print_cont,gamma,wbr_scan,wbb_scan,amplify_grain,amplify_mask,on_grain,white_point]
                for sld in sldrs:
                    sld.disabled = True
                    sld.update()
                cash_flet.blok_spase = True


                p_image = Distributor.magic(cash_flet.resolution,cash_flet.uploaded_file,cash_flet.Wb,cash_flet.blur_rad,
                                            cash_flet.halation,cash_flet.blur_spred,cash_flet.bloom_rad,cash_flet.bloom_Halation,
                                            cash_flet.bloom_spred,cash_flet.sharp_rad,cash_flet.sharp_spred,cash_flet.sharp_amplif,
                                            
                                            cash_flet.camera,cash_flet.r_hue,cash_flet.r_sut,cash_flet.r_g,cash_flet.r_b,
                                            cash_flet.g_hue,cash_flet.g_sut,cash_flet.g_r,cash_flet.g_b,
                                            cash_flet.b_hue,cash_flet.b_sut,cash_flet.b_r,cash_flet.b_g,
                                            cash_flet.film,"None",cash_flet.sut,
                                            cash_flet.end_a_plus,cash_flet.end_a_minus,cash_flet.end_b_plus,cash_flet.end_b_minus,
                                            cash_flet.sut_a_minus,cash_flet.sut_a_plus,cash_flet.sut_b_minus,cash_flet.sut_b_plus,
                                            0,0,0,0,cash_flet.sut_compr,
                                            cash_flet.steepness_shad,cash_flet.width_shad,cash_flet.steepness_light,cash_flet.width_light,
                                            cash_flet.shad_a,cash_flet.shad_b,cash_flet.mid_a,cash_flet.mid_b,cash_flet.light_a,cash_flet.light_b,
                                            cash_flet.neutral_mask,cash_flet.amplyfy_wheel,cash_flet.wbb,cash_flet.wbr,0,0,cash_flet.print_exp,
                                            cash_flet.light_compr,cash_flet.zone,cash_flet.print_cont,cash_flet.gamma,cash_flet.wbr_scan,cash_flet.wbb_scan,
                                            cash_flet.amplify_grain,cash_flet.amplify_mask,cash_flet.on_grain,cash_flet.white_point,is_half=True)

                for sld in sldrs:
                    sld.disabled = False
                    sld.update()
                cash_flet.blok_spase = False
                tac = time.time()
                print(tac-tic,"for disable")
                arr = io.convert_bit_depth(p_image,"uint8")
                pil_img = Image.fromarray(arr)
                cash_flet.pil_img = np.array(pil_img)
                buff = BytesIO()
                pil_img.save(buff, format="JPEG",quality=100)
                cash_flet.image_string = base64.b64encode(buff.getvalue()).decode("utf-8") 
                img.src_base64 = cash_flet.image_string
                img.update()  
        #===================wbr================================================
        #==================wbr=================================================
        def wbr_c_0(e):
            cash_flet.wbr = 0
            wbr_txt.value =" " + str(round(0.00,1)) 
            wbr_sld.value = 0
            go_go(e)
            tabss.update()
        def wbr_c(e):
            cash_flet.wbr = e.control.value
            wbr_txt.value = " "+str(round(e.control.value,2)) 
            wbr.update()

        
        wbr_txt = ft.Text(" " + str(round(cash_flet.wbr,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        wbr_sld = ft.Slider(min=-10, max=10, divisions=1000, value=0.0, on_change_end=go_go,on_change=wbr_c,expand=True,active_color=ft.colors.RED_300) 
        wbr = ft.Column(controls=[ft.Container(ft.Text("Source Wight Balance",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="wbr_sld",on_hover=sldr_hover,content=ft.Row([ wbr_sld ,  ft.Container(wbr_txt,on_click=wbr_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======wbb================wbb==============wbb===============================
        def wbb_c_0(e):
            cash_flet.wbb = 0
            wbb_txt.value =" " + str(round(0.00,1)) 
            wbb_sld.value = 0
            go_go(e)
            tabss.tabs[2].update()

        def wbb_c(e):
            cash_flet.wbb = e.control.value
            wbb_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        wbb_txt = ft.Text(" " + str(round(cash_flet.wbr,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        wbb_sld = ft.Slider(min=-10, max=10, divisions=1000, value=0.0, on_change_end=go_go,on_change=wbb_c,expand=True,active_color=ft.colors.BLUE_300) 
        wbb = ft.Column(controls=[#ft.Container(ft.Text("Wight Balance | Blue",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="wbb_sld",on_hover=sldr_hover,content=ft.Row([ wbb_sld ,  ft.Container(wbb_txt,on_click=wbb_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #=====wbr_scan===================wbr_scan=============================wbr_scan==============
        def wbr_scan_c_0(e):
            cash_flet.wbr_scan = 1.0
            wbr_scan_txt.value =" " + str(round(1.0,2)) 
            wbr_scan_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def wbr_scan_c(e):
            cash_flet.wbr_scan = e.control.value
            wbr_scan_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        wbr_scan_txt = ft.Text(" " + str(round(cash_flet.wbr_scan,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        wbr_scan_sld = ft.Slider(min=0.5, max=1.5, value=1.0, divisions=1000, on_change_end=go_go,on_change=wbr_scan_c,expand=True,active_color=ft.colors.RED_300, ) 
        wbr_scan = ft.Column(controls=[ft.Container(ft.Text("Scan WB",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="wbr_scan_sld",on_hover=sldr_hover,content=ft.Row([ wbr_scan_sld ,  ft.Container(wbr_scan_txt,on_click=wbr_scan_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #=======wbb_scan================wbb_scan===========================wbb_scan=================
        def wbb_scan_c_0(e):
            cash_flet.wbb_scan = 1.0
            wbb_scan_txt.value =" " + str(round(1.0,2)) 
            wbb_scan_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def wbb_scan_c(e):
            cash_flet.wbb_scan = e.control.value
            wbb_scan_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        wbb_scan_txt = ft.Text(" " + str(round(cash_flet.wbb_scan,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        wbb_scan_sld = ft.Slider(min=0.5, max=1.5,value=1.0, divisions=1000, on_change_end=go_go,on_change=wbb_scan_c,expand=True,active_color=ft.colors.BLUE_300, ) 
        wbb_scan = ft.Column(controls=[#ft.Container(ft.Text("Scan WB | Blue",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="wbb_scan_sld",on_hover=sldr_hover,content=ft.Row([ wbb_scan_sld ,  ft.Container(wbb_scan_txt,on_click=wbb_scan_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #========film_prop=================film_prop========================film_prop==================
        film_prop = ft.Container(ft.Column(controls=[
                                        cam_drop,film_drop,wbr_scan,wbb_scan
             

                                        ]),border=ft.border.all(1, ft.colors.GREY_800),border_radius=5,padding=5
                                )
        #====gamma================gamma==========================gamma=====================
        def gamma_c_0(e):
            cash_flet.gamma = 0
            gamma_txt.value =" " + str(round(0.00,1)) 
            gamma_sld.value = 0
            go_go(e)
            tabss.tabs[2].update()

        def gamma_c(e):
            cash_flet.gamma = e.control.value
            gamma_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        gamma_txt = ft.Text(" " + str(round(cash_flet.gamma,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        gamma_sld = ft.Slider(min=-4, max=8, divisions=1000, value=0.0, on_change_end=go_go,on_change=gamma_c,expand=True,active_color=ft.colors.GREY_300 ) 
        gamma = ft.Column(controls=[ft.Container(ft.Text("Film under/over exposer",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="gamma_sld",on_hover=sldr_hover,content=ft.Row([ gamma_sld ,  ft.Container(gamma_txt,on_click=gamma_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #=====print_exp===============print_exp=======================print_exp========================
        def print_exp_c_0(e):
            cash_flet.print_exp = 0
            print_exp_txt.value =" " + str(round(0.00,1)) 
            print_exp_sld.value = 0
            go_go(e)
            tabss.tabs[2].update()

        def print_exp_c(e):
            cash_flet.print_exp = e.control.value
            print_exp_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()



        print_exp_txt = ft.Text(" " + str(round(cash_flet.print_exp,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        print_exp_sld = ft.Slider(min=-4, max=4, divisions=1000, value=0.0, on_change_end=go_go,on_change=print_exp_c,expand=True,active_color=ft.colors.GREY_300,) 
        print_exp = ft.Column(controls=[ft.Container(ft.Text("Print exposer",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key = "print_exp_sld", on_hover=sldr_hover,content=ft.Row([ print_exp_sld ,  ft.Container(print_exp_txt,on_click=print_exp_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #=======print_cont===================print_cont=====================print_cont====================
        def print_cont_c_0(e):
            cash_flet.print_cont = 0.70
            print_cont_txt.value =" " + str(round(0.70,1)) 
            print_cont_sld.value = 0.70
            go_go(e)
            tabss.tabs[2].update()

        def print_cont_c(e):
            cash_flet.print_cont = e.control.value
            print_cont_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        print_cont_txt = ft.Text(" " + str(round(cash_flet.print_cont,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        print_cont_sld = ft.Slider(min=0.1, max=2,value=0.70, divisions=100, on_change_end=go_go,on_change=print_cont_c,expand=True,active_color=ft.colors.GREY_300, ) 
        print_cont = ft.Column(controls=[ft.Container(ft.Text("Print Contrast",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="print_cont_sld",on_hover=sldr_hover,content=ft.Row([ print_cont_sld ,  ft.Container(print_cont_txt,on_click=print_cont_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        
        #======sut=================sut========================sut====================
        def sut_c_0(e):
            cash_flet.sut = 1.0
            sut_txt.value =" " + str(round(1.0,1)) 
            sut_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def sut_c(e):
            cash_flet.sut = e.control.value
            sut_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        sut_txt = ft.Text(" " + str(round(cash_flet.sut,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        sut_sld = ft.Slider(min=0.1, max=3,value=1.0, divisions=300, on_change_end=go_go,on_change=sut_c,expand=True,active_color=ft.colors.GREY_300, ) 
        sut = ft.Column(controls=[ft.Container(ft.Text("Saturation",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="sut_sld",on_hover=sldr_hover,content=ft.Row([ sut_sld ,  ft.Container(sut_txt,on_click=sut_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #=====white_point===========white_point===================white_point================================
        def white_point_c_0(e):
            cash_flet.white_point = 1.0
            white_point_txt.value =" " + str(round(1.0,1)) 
            white_point_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def white_point_c(e):
            cash_flet.white_point = e.control.value
            white_point_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        white_point_txt = ft.Text(" " + str(round(cash_flet.white_point,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        white_point_sld = ft.Slider(min=0.1, max=2.0,value=1.0, divisions=100, on_change_end=go_go,on_change=white_point_c,expand=True,active_color=ft.colors.GREY_300, ) 
        white_point = ft.Column(controls=[ft.Container(ft.Text("White point",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="white_point_sld", on_hover=sldr_hover,content=ft.Row([ white_point_sld ,  ft.Container(white_point_txt,on_click=white_point_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #=====light_compr==============light_compr=========================light_compr=======================
        def light_compr_c_0(e):
            cash_flet.light_compr = 0.0
            light_compr_txt.value =" " + str(round(0.0,1)) 
            light_compr_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def light_compr_c(e):
            cash_flet.light_compr = e.control.value
            light_compr_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        light_compr_txt = ft.Text(" " + str(round(cash_flet.light_compr,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        light_compr_sld = ft.Slider(min=-5.0, max=5.0,value=0.0, divisions=100, on_change_end=go_go,on_change=light_compr_c,expand=True,active_color=ft.colors.GREY_300, ) 
        light_compr = ft.Column(controls=[ft.Container(ft.Text("Light compression value",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="light_compr_sld", on_hover=sldr_hover,content=ft.Row([ light_compr_sld ,  ft.Container(light_compr_txt,on_click=light_compr_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #=======zone====================zone=====================zone===================
        def zone_c_0(e):
            cash_flet.zone = 0.00
            zone_txt.value =" " + str(round(0.00,2)) 
            zone_sld.value = 0.00
            go_go(e)
            tabss.tabs[2].update()

        def zone_c(e):
            cash_flet.zone = e.control.value
            zone_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        zone_txt = ft.Text(" " + str(round(cash_flet.zone,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        zone_sld = ft.Slider(min=0.0, max=1.0,value=0.00, divisions=100, on_change_end=go_go,on_change=zone_c,expand=True,active_color=ft.colors.GREY_300) 
        zone = ft.Column(controls=[ft.Container(ft.Text("Light compression zone",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="zone_sld", on_hover=sldr_hover,content=ft.Row([ zone_sld ,  ft.Container(zone_txt,on_click=zone_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________
        #PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________
        #PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________
        #PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________
        #PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________
        #PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________
        #PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________
        #PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________#PROLAB__________________
        def sut_compr_c_0(e):
            cash_flet.sut_compr = 0.85
            sut_compr_txt.value =" " + str(round(0.85,2)) 
            sut_compr_sld.value = 0.85
            go_go(e)
            tabss.tabs[2].update()

        def sut_compr_c(e):
            cash_flet.sut_compr = e.control.value
            sut_compr_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        sut_compr_txt = ft.Text(" " + str(round(cash_flet.sut_compr,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        sut_compr_sld = ft.Slider(min=0.1, max=5.0,value=0.85, divisions=100, on_change_end=go_go,on_change=sut_compr_c,expand=True,active_color=ft.colors.GREY_300) 
        sut_compr = ft.Column(controls=[ft.Container(ft.Text("Saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="sut_compr_sld", on_hover=sldr_hover,content=ft.Row([ sut_compr_sld ,  ft.Container(sut_compr_txt,on_click=sut_compr_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #======end_a_plus=========end_a_plus=========================end_a_plus==========================
        def end_a_plus_c_0(e):
            cash_flet.end_a_plus = 1.0
            end_a_plus_txt.value =" " + str(round(1.0,2)) 
            end_a_plus_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def end_a_plus_c(e):
            cash_flet.end_a_plus = e.control.value
            end_a_plus_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        end_a_plus_txt = ft.Text(" " + str(round(cash_flet.end_a_plus,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        end_a_plus_sld = ft.Slider(min=0.1, max=3.0,value=1.0, divisions=100, on_change_end=go_go,on_change=end_a_plus_c,expand=True,active_color=ft.colors.PINK_300) 
        end_a_plus = ft.Column(controls=[ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="end_a_plus_sld", on_hover=sldr_hover,content=ft.Row([ end_a_plus_sld ,  ft.Container(end_a_plus_txt,on_click=end_a_plus_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #======end_a_minus=================end_a_minus=================end_a_minus===========================
        def end_a_minus_c_0(e):
            cash_flet.end_a_minus = 1.0
            end_a_minus_txt.value =" " + str(round(1.0,2)) 
            end_a_minus_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def end_a_minus_c(e):
            cash_flet.end_a_minus = e.control.value
            end_a_minus_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        end_a_minus_txt = ft.Text(" " + str(round(cash_flet.end_a_minus,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        end_a_minus_sld = ft.Slider(min=0.1, max=3.0,value=1.0, divisions=100, on_change_end=go_go,on_change=end_a_minus_c,expand=True,active_color=ft.colors.GREEN_300) 
        end_a_minus = ft.Column(controls=[#ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="end_a_minus_sld", on_hover=sldr_hover,content=ft.Row([ end_a_minus_sld ,  ft.Container(end_a_minus_txt,on_click=end_a_minus_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #=======end_b_plus==================end_b_plus=====================end_b_plus====================
        def end_b_plus_c_0(e):
            cash_flet.end_b_plus = 1.0
            end_b_plus_txt.value =" " + str(round(1.0,2)) 
            end_b_plus_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def end_b_plus_c(e):
            cash_flet.end_b_plus = e.control.value
            end_b_plus_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        end_b_plus_txt = ft.Text(" " + str(round(cash_flet.end_b_plus,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        end_b_plus_sld = ft.Slider(min=0.1, max=3.0,value=1.0, divisions=100, on_change_end=go_go,on_change=end_b_plus_c,expand=True,active_color=ft.colors.YELLOW_300) 
        end_b_plus = ft.Column(controls=[#ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="end_b_plus_sld", on_hover=sldr_hover,content=ft.Row([ end_b_plus_sld ,  ft.Container(end_b_plus_txt,on_click=end_b_plus_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #========end_b_minus=================end_b_minus=======================end_b_minus===================
        def end_b_minus_c_0(e):
            cash_flet.end_b_minus = 1.0
            end_b_minus_txt.value =" " + str(round(1.0,2)) 
            end_b_minus_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def end_b_minus_c(e):
            cash_flet.end_b_minus = e.control.value
            end_b_minus_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        end_b_minus_txt = ft.Text(" " + str(round(cash_flet.end_b_minus,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        end_b_minus_sld = ft.Slider(min=0.1, max=3.0,value=1.0, divisions=100, on_change_end=go_go,on_change=end_b_minus_c,expand=True,active_color=ft.colors.CYAN_300) 
        end_b_minus = ft.Column(controls=[#ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="end_b_minus_sld", on_hover=sldr_hover,content=ft.Row([ end_b_minus_sld ,  ft.Container(end_b_minus_txt,on_click=end_b_minus_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)

        #======sut_a_plus=============sut_a_plus===========================sut_a_plus=====================
        def sut_a_plus_c_0(e):
            cash_flet.sut_a_plus = 1.0
            sut_a_plus_txt.value =" " + str(round(1.0,2)) 
            sut_a_plus_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def sut_a_plus_c(e):
            cash_flet.sut_a_plus = e.control.value
            sut_a_plus_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        sut_a_plus_txt = ft.Text(" " + str(round(cash_flet.sut_a_plus,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        sut_a_plus_sld = ft.Slider(min=0.1, max=3.0,value=1.0, divisions=100, on_change_end=go_go,on_change=sut_a_plus_c,expand=True,active_color=ft.colors.PINK_300) 
        sut_a_plus = ft.Column(controls=[ft.Container(ft.Text("Chanels saturation",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="sut_a_plus_sld", on_hover=sldr_hover,content=ft.Row([ sut_a_plus_sld ,  ft.Container(sut_a_plus_txt,on_click=sut_a_plus_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #========sut_a_minus=======================sut_a_minus===================sut_a_minus=================
        def sut_a_minus_c_0(e):
            cash_flet.sut_a_minus = 1.0
            sut_a_minus_txt.value =" " + str(round(1.0,2)) 
            sut_a_minus_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def sut_a_minus_c(e):
            cash_flet.sut_a_minus = e.control.value
            sut_a_minus_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        sut_a_minus_txt = ft.Text(" " + str(round(cash_flet.sut_a_minus,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        sut_a_minus_sld = ft.Slider(min=0.1, max=3.0,value=1.0, divisions=100, on_change_end=go_go,on_change=sut_a_minus_c,expand=True,active_color=ft.colors.GREEN_300) 
        sut_a_minus = ft.Column(controls=[#ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="sut_a_minus_sld", on_hover=sldr_hover,content=ft.Row([ sut_a_minus_sld ,  ft.Container(sut_a_minus_txt,on_click=sut_a_minus_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #======sut_b_plus=====================sut_b_plus========================sut_b_plus================
        def sut_b_plus_c_0(e):
            cash_flet.sut_b_plus = 1.0
            sut_b_plus_txt.value =" " + str(round(1.0,2)) 
            sut_b_plus_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def sut_b_plus_c(e):
            cash_flet.sut_b_plus = e.control.value
            sut_b_plus_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        sut_b_plus_txt = ft.Text(" " + str(round(cash_flet.sut_b_plus,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        sut_b_plus_sld = ft.Slider(min=0.1, max=3.0,value=1.0, divisions=100, on_change_end=go_go,on_change=sut_b_plus_c,expand=True,active_color=ft.colors.YELLOW_300) 
        sut_b_plus = ft.Column(controls=[#ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="sut_b_plus_sld", on_hover=sldr_hover,content=ft.Row([ sut_b_plus_sld ,  ft.Container(sut_b_plus_txt,on_click=sut_b_plus_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        #======sut_b_minus==================sut_b_minus======================sut_b_minus=====================
        def sut_b_minus_c_0(e):
            cash_flet.sut_b_minus = 1.0
            sut_b_minus_txt.value =" " + str(round(1.0,2)) 
            sut_b_minus_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def sut_b_minus_c(e):
            cash_flet.sut_b_minus = e.control.value
            sut_b_minus_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        sut_b_minus_txt = ft.Text(" " + str(round(cash_flet.sut_b_minus,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        sut_b_minus_sld = ft.Slider(min=0.1, max=3.0,value=1.0, divisions=100, on_change_end=go_go,on_change=sut_b_minus_c,expand=True,active_color=ft.colors.CYAN_300) 
        sut_b_minus = ft.Column(controls=[#ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="sut_b_minus_sld", on_hover=sldr_hover,content=ft.Row([ sut_b_minus_sld ,  ft.Container(sut_b_minus_txt,on_click=sut_b_minus_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)





        ProLab = ft.ExpansionTile(title=ft.Text("ProLab"),controls=[
                                                                    sut_compr,end_a_plus,end_a_minus,end_b_plus,end_b_minus,
                                                                    sut_a_plus,sut_a_minus,sut_b_plus,sut_b_minus
                                                                    ])

        #RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________
        #RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________
        #RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________
        #RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________
        #RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________
        #RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________
        #RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________
        #RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________#RGB_MIXER__________________

        #======r_sut==================r_sut======================r_sut=====================
        def r_sut_c_0(e):
            cash_flet.r_sut = 0.0
            r_sut_txt.value =" " + str(round(0.0,2)) 
            r_sut_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def r_sut_c(e):
            cash_flet.r_sut = e.control.value
            r_sut_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        r_sut_txt = ft.Text(" " + str(round(cash_flet.r_sut,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        r_sut_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=r_sut_c,expand=True,active_color=ft.colors.RED_300) 
        r_sut = ft.Column(controls=[ft.Container(ft.Text("Saturation",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="r_sut_sld", on_hover=sldr_hover,content=ft.Row([ r_sut_sld ,  ft.Container(r_sut_txt,on_click=r_sut_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        
        #======g_sut==================g_sut======================g_sut=====================
        def g_sut_c_0(e):
            cash_flet.g_sut = 0.0
            g_sut_txt.value =" " + str(round(0.0,2)) 
            g_sut_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def g_sut_c(e):
            cash_flet.g_sut = e.control.value
            g_sut_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        g_sut_txt = ft.Text(" " + str(round(cash_flet.g_sut,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        g_sut_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=g_sut_c,expand=True,active_color=ft.colors.GREEN_300) 
        g_sut = ft.Column(controls=[#ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="g_sut_sld", on_hover=sldr_hover,content=ft.Row([ g_sut_sld ,  ft.Container(g_sut_txt,on_click=g_sut_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)        
       
        
        #======b_sut==================b_sut======================b_sut=====================
        def b_sut_c_0(e):
            cash_flet.b_sut = 0.0
            b_sut_txt.value =" " + str(round(0.0,2)) 
            b_sut_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def b_sut_c(e):
            cash_flet.b_sut = e.control.value
            b_sut_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        b_sut_txt = ft.Text(" " + str(round(cash_flet.b_sut,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        b_sut_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=b_sut_c,expand=True,active_color=ft.colors.BLUE_300) 
        b_sut = ft.Column(controls=[#ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="b_sut_sld", on_hover=sldr_hover,content=ft.Row([ b_sut_sld ,  ft.Container(b_sut_txt,on_click=b_sut_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)      
        
        
        #======r_hue==================r_hue======================r_hue=====================
        def r_hue_c_0(e):
            cash_flet.r_hue = 0.0
            r_hue_txt.value =" " + str(round(0.0,2)) 
            r_hue_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def r_hue_c(e):
            cash_flet.r_hue = e.control.value
            r_hue_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        r_hue_txt = ft.Text(" " + str(round(cash_flet.r_hue,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        r_hue_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=r_hue_c,expand=True,active_color=ft.colors.RED_300) 
        r_hue = ft.Column(controls=[ft.Container(ft.Text("HUE",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="r_hue_sld", on_hover=sldr_hover,content=ft.Row([ r_hue_sld ,  ft.Container(r_hue_txt,on_click=r_hue_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)
        
        #======g_hue==================g_hue======================g_hue=====================
        def g_hue_c_0(e):
            cash_flet.g_hue = 0.0
            g_hue_txt.value =" " + str(round(0.0,2)) 
            g_hue_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def g_hue_c(e):
            cash_flet.g_hue = e.control.value
            g_hue_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        g_hue_txt = ft.Text(" " + str(round(cash_flet.g_hue,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        g_hue_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=g_hue_c,expand=True,active_color=ft.colors.GREEN_300) 
        g_hue = ft.Column(controls=[#ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="g_hue_sld", on_hover=sldr_hover,content=ft.Row([ g_hue_sld ,  ft.Container(g_hue_txt,on_click=g_hue_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)        
       
        
        #======b_hue==================b_hue======================b_hue=====================
        def b_hue_c_0(e):
            cash_flet.b_hue = 0.0
            b_hue_txt.value =" " + str(round(0.0,2)) 
            b_hue_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def b_hue_c(e):
            cash_flet.b_hue = e.control.value
            b_hue_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        b_hue_txt = ft.Text(" " + str(round(cash_flet.b_hue,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        b_hue_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=b_hue_c,expand=True,active_color=ft.colors.BLUE_300) 
        b_hue = ft.Column(controls=[#ft.Container(ft.Text("Chanels saturation compression",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="b_hue_sld", on_hover=sldr_hover,content=ft.Row([ b_hue_sld ,  ft.Container(b_hue_txt,on_click=b_hue_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)              

        #======r_g==================r_g======================r_g=====================
        def r_g_c_0(e):
            cash_flet.r_g = 0.0
            r_g_txt.value =" " + str(round(0.0,2)) 
            r_g_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def r_g_c(e):
            cash_flet.r_g = e.control.value
            r_g_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        r_g_txt = ft.Text(" " + str(round(cash_flet.r_g,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        r_g_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=r_g_c,expand=True,active_color=ft.colors.GREEN_300,inactive_color=ft.colors.RED_300) 
        r_g = ft.Column(controls=[ft.Container(ft.Text("Mixer",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="r_g_sld", on_hover=sldr_hover,content=ft.Row([ r_g_sld ,  ft.Container(r_g_txt,on_click=r_g_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)              

        #======r_b==================r_b======================r_b=====================
        def r_b_c_0(e):
            cash_flet.r_b = 0.0
            r_b_txt.value =" " + str(round(0.0,2)) 
            r_b_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def r_b_c(e):
            cash_flet.r_b = e.control.value
            r_b_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        r_b_txt = ft.Text(" " + str(round(cash_flet.r_b,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        r_b_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=r_b_c,expand=True,active_color=ft.colors.BLUE_300,inactive_color=ft.colors.RED_300) 
        r_b = ft.Column(controls=[#ft.Container(ft.Text("Mixer",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="r_b_sld", on_hover=sldr_hover,content=ft.Row([ r_b_sld ,  ft.Container(r_b_txt,on_click=r_b_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)     

        #======g_r==================g_r======================g_r=====================
        def g_r_c_0(e):
            cash_flet.g_r = 0.0
            g_r_txt.value =" " + str(round(0.0,2)) 
            g_r_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def g_r_c(e):
            cash_flet.g_r = e.control.value
            g_r_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        g_r_txt = ft.Text(" " + str(round(cash_flet.g_r,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        g_r_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=g_r_c,expand=True,active_color=ft.colors.RED_300,inactive_color=ft.colors.GREEN_300) 
        g_r = ft.Column(controls=[#ft.Container(ft.Text("Mixer",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="g_r_sld", on_hover=sldr_hover,content=ft.Row([ g_r_sld ,  ft.Container(g_r_txt,on_click=g_r_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)              

        #======g_b==================g_b======================g_b=====================
        def g_b_c_0(e):
            cash_flet.g_b = 0.0
            g_b_txt.value =" " + str(round(0.0,2)) 
            g_b_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def g_b_c(e):
            cash_flet.g_b = e.control.value
            g_b_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        g_b_txt = ft.Text(" " + str(round(cash_flet.g_b,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        g_b_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=g_b_c,expand=True,active_color=ft.colors.BLUE_300,inactive_color=ft.colors.GREEN_300) 
        g_b = ft.Column(controls=[#ft.Container(ft.Text("Mixer",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="g_b_sld", on_hover=sldr_hover,content=ft.Row([ g_b_sld ,  ft.Container(g_b_txt,on_click=g_b_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)         
        
        #======b_r==================b_r======================b_r=====================
        def b_r_c_0(e):
            cash_flet.b_r = 0.0
            b_r_txt.value =" " + str(round(0.0,2)) 
            b_r_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def b_r_c(e):
            cash_flet.b_r = e.control.value
            b_r_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        b_r_txt = ft.Text(" " + str(round(cash_flet.b_r,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        b_r_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=b_r_c,expand=True,active_color=ft.colors.RED_300,inactive_color=ft.colors.BLUE_300) 
        b_r = ft.Column(controls=[#ft.Container(ft.Text("Mixer",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="b_r_sld", on_hover=sldr_hover,content=ft.Row([ b_r_sld ,  ft.Container(b_r_txt,on_click=b_r_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)              

        #======b_g==================b_g======================b_g=====================
        def b_g_c_0(e):
            cash_flet.b_g = 0.0
            b_g_txt.value =" " + str(round(0.0,2)) 
            b_g_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def b_g_c(e):
            cash_flet.b_g = e.control.value
            b_g_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        b_g_txt = ft.Text(" " + str(round(cash_flet.b_g,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        b_g_sld = ft.Slider(min=-1.0, max=1.0,value=0.0, divisions=100, on_change_end=go_go,on_change=b_g_c,expand=True,active_color=ft.colors.GREEN_300,inactive_color=ft.colors.BLUE_300) 
        b_g = ft.Column(controls=[#ft.Container(ft.Text("Mixer",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="b_g_sld", on_hover=sldr_hover,content=ft.Row([ b_g_sld ,  ft.Container(b_g_txt,on_click=b_g_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)         
        
        rgb_mix = ft.ExpansionTile(title=ft.Text("RGB Mixer"),controls=[
                                                                        r_sut,g_sut,b_sut,
                                                                        r_hue,g_hue,b_hue,
                                                                        r_g,r_b,g_r,g_b,b_r,b_g
                                                                        ])
        #SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________
        #SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________
        #SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________
        #SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________
        #SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________
        #SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________
        #SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________
        #SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________
        #SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________
        #SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________#SRUCTURE_________________________
        def on_grain_c(e):
             cash_flet.on_grain = e.control.value
             go_go(e)

        #======on_grain==================on_grain======================on_grain=====================
        on_grain = ft.Container(ft.Switch(label="Grain",value=False,on_change=on_grain_c,label_position=ft.LabelPosition.LEFT,active_color=ft.colors.GREY_500),padding=shirina/50)     
        #======amplify_grain==================amplify_grain======================amplify_grain=====================
        def amplify_grain_c_0(e):
            cash_flet.amplify_grain = 0.2
            amplify_grain_txt.value =" " + str(round(0.2,2)) 
            amplify_grain_sld.value = 0.2
            go_go(e)
            tabss.tabs[2].update()

        def amplify_grain_c(e):
            cash_flet.amplify_grain = e.control.value
            amplify_grain_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        amplify_grain_txt = ft.Text(" " + str(round(cash_flet.amplify_grain,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        amplify_grain_sld = ft.Slider(min=0.0, max=1,value=0.2, divisions=100, on_change_end=go_go,on_change=amplify_grain_c,expand=True,active_color=ft.colors.GREY_300) 
        amplify_grain = ft.Column(controls=[ft.Container(ft.Text("Grain Amplify",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="amplify_grain_sld", on_hover=sldr_hover,content=ft.Row([ amplify_grain_sld ,  ft.Container(amplify_grain_txt,on_click=amplify_grain_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)    
        #======amplify_mask==================amplify_mask======================amplify_mask=====================
        def amplify_mask_c_0(e):
            cash_flet.amplify_mask = 7.0
            amplify_mask_txt.value =" " + str(round(7.0,2)) 
            amplify_mask_sld.value = 7.0
            go_go(e)
            tabss.tabs[2].update()

        def amplify_mask_c(e):
            cash_flet.amplify_mask = e.control.value
            amplify_mask_txt.value = " "+str(round(e.control.value,2)) 
            tabss.tabs[2].update()


        amplify_mask_txt = ft.Text(" " + str(round(cash_flet.amplify_mask,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        amplify_mask_sld = ft.Slider(min=1.0, max=10.0,value=7.0, divisions=100, on_change_end=go_go,on_change=amplify_mask_c,expand=True,active_color=ft.colors.GREY_300) 
        amplify_mask = ft.Column(controls=[ft.Container(ft.Text("Grain mask Amplify",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="amplify_mask_sld", on_hover=sldr_hover,content=ft.Row([ amplify_mask_sld ,  ft.Container(amplify_mask_txt,on_click=amplify_mask_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)    
        #======blur_rad==================blur_rad======================blur_rad=====================
        def blur_rad_c_0(e):
            cash_flet.blur_rad = 1.0
            blur_rad_txt.value =" " + str(round(1.0,1)) 
            blur_rad_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def blur_rad_c(e):
            cash_flet.blur_rad = e.control.value
            blur_rad_txt.value = " "+str(round(e.control.value,1)) 
            tabss.tabs[2].update()


        blur_rad_txt = ft.Text(" " + str(round(cash_flet.blur_rad,1)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        blur_rad_sld = ft.Slider(min=0.1, max=5.0,value=1.0, divisions=100, on_change_end=go_go,on_change=blur_rad_c,expand=True,active_color=ft.colors.GREY_300) 
        blur_rad = ft.Column(controls=[ft.Container(ft.Text("Radius",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="blur_rad_sld", on_hover=sldr_hover,content=ft.Row([ blur_rad_sld ,  ft.Container(blur_rad_txt,on_click=blur_rad_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)   
        #======blur_spred==================blur_spred======================blur_spred=====================
        def blur_spred_c_0(e):
            cash_flet.blur_spred = 9.0
            blur_spred_txt.value =" " + str(round(9.0,1)) 
            blur_spred_sld.value = 9.0
            go_go(e)
            tabss.tabs[2].update()

        def blur_spred_c(e):
            cash_flet.blur_spred = e.control.value
            blur_spred_txt.value = " "+str(round(e.control.value,1)) 
            tabss.tabs[2].update()


        blur_spred_txt = ft.Text(" " + str(round(cash_flet.blur_spred,1)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        blur_spred_sld = ft.Slider(min=0.1, max=16.0,value=9.0, divisions=100, on_change_end=go_go,on_change=blur_spred_c,expand=True,active_color=ft.colors.GREY_300) 
        blur_spred = ft.Column(controls=[ft.Container(ft.Text("Spread",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="blur_spred_sld", on_hover=sldr_hover,content=ft.Row([ blur_spred_sld ,  ft.Container(blur_spred_txt,on_click=blur_spred_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======halation==================halation======================halation=====================
        def halation_c_0(e):
            cash_flet.halation = 1.7
            halation_txt.value =" " + str(round(1.7,1)) 
            halation_sld.value = 1.7
            go_go(e)
            tabss.tabs[2].update()

        def halation_c(e):
            cash_flet.halation = e.control.value
            halation_txt.value = " "+str(round(e.control.value,1)) 
            tabss.tabs[2].update()


        halation_txt = ft.Text(" " + str(round(cash_flet.halation,2)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        halation_sld = ft.Slider(min=1.0, max=3.0,value=1.7, divisions=100, on_change_end=go_go,on_change=halation_c,expand=True,active_color=ft.colors.GREY_300) 
        halation = ft.Column(controls=[ft.Container(ft.Text("Halation",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="halation_sld", on_hover=sldr_hover,content=ft.Row([ halation_sld ,  ft.Container(halation_txt,on_click=halation_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 

        #======bloom_rad==================bloom_rad======================bloom_rad=====================
        def bloom_rad_c_0(e):
            cash_flet.bloom_rad = 140.
            bloom_rad_txt.value =" " + str(round(140.0)) 
            bloom_rad_sld.value = 140.0
            go_go(e)
            tabss.tabs[2].update()

        def bloom_rad_c(e):
            cash_flet.bloom_rad = e.control.value
            bloom_rad_txt.value = " "+str(round(e.control.value)) 
            tabss.tabs[2].update()


        bloom_rad_txt = ft.Text(" " + str(round(cash_flet.bloom_rad)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        bloom_rad_sld = ft.Slider(min=10.0, max=300.0,value=140.0, divisions=100, on_change_end=go_go,on_change=bloom_rad_c,expand=True,active_color=ft.colors.GREY_300) 
        bloom_rad = ft.Column(controls=[ft.Container(ft.Text("Radius",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="bloom_rad_sld", on_hover=sldr_hover,content=ft.Row([ bloom_rad_sld ,  ft.Container(bloom_rad_txt,on_click=bloom_rad_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)   
        #======bloom_spred==================bloom_spred======================bloom_spred=====================
        def bloom_spred_c_0(e):
            cash_flet.bloom_spred = 21.0
            bloom_spred_txt.value =" " + str(round(21.0,1)) 
            bloom_spred_sld.value = 21.0
            go_go(e)
            tabss.tabs[2].update()

        def bloom_spred_c(e):
            cash_flet.bloom_spred = e.control.value
            bloom_spred_txt.value = " "+str(round(e.control.value,1)) 
            tabss.tabs[2].update()


        bloom_spred_txt = ft.Text(" " + str(round(cash_flet.bloom_spred,1)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        bloom_spred_sld = ft.Slider(min=0.1, max=30.0,value=21.0, divisions=100, on_change_end=go_go,on_change=bloom_spred_c,expand=True,active_color=ft.colors.GREY_300) 
        bloom_spred = ft.Column(controls=[ft.Container(ft.Text("Spread",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="bloom_spred_sld", on_hover=sldr_hover,content=ft.Row([ bloom_spred_sld ,  ft.Container(bloom_spred_txt,on_click=bloom_spred_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======bloom_Halation==================bloom_Halation======================bloom_Halation=====================
        def bloom_Halation_c_0(e):
            cash_flet.bloom_Halation = 1.0
            bloom_Halation_txt.value =" " + str(round(1.0,1)) 
            bloom_Halation_sld.value = 1.0
            go_go(e)
            tabss.tabs[2].update()

        def bloom_Halation_c(e):
            cash_flet.bloom_Halation = e.control.value
            bloom_Halation_txt.value = " "+str(round(e.control.value,1)) 
            tabss.tabs[2].update()


        bloom_Halation_txt = ft.Text(" " + str(round(cash_flet.bloom_Halation,1)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        bloom_Halation_sld = ft.Slider(min=1.0, max=3.0,value=1.0, divisions=100, on_change_end=go_go,on_change=bloom_Halation_c,expand=True,active_color=ft.colors.GREY_300) 
        bloom_Halation = ft.Column(controls=[ft.Container(ft.Text("Halation",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="bloom_Halation_sld", on_hover=sldr_hover,content=ft.Row([ bloom_Halation_sld ,  ft.Container(bloom_Halation_txt,on_click=bloom_Halation_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======sharp_rad==================sharp_rad======================sharp_rad=====================
        def sharp_rad_c_0(e):
            cash_flet.sharp_rad = 40.0
            sharp_rad_txt.value =" " + str(round(40.0)) 
            sharp_rad_sld.value = 40.0
            go_go(e)
            tabss.tabs[2].update()

        def sharp_rad_c(e):
            cash_flet.sharp_rad = e.control.value
            sharp_rad_txt.value = " "+str(round(e.control.value)) 
            tabss.tabs[2].update()


        sharp_rad_txt = ft.Text(" " + str(round(cash_flet.sharp_rad)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        sharp_rad_sld = ft.Slider(min=1.1, max=100.0,value=40.0, divisions=100, on_change_end=go_go,on_change=sharp_rad_c,expand=True,active_color=ft.colors.GREY_300) 
        sharp_rad = ft.Column(controls=[ft.Container(ft.Text("Radius",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="sharp_rad_sld", on_hover=sldr_hover,content=ft.Row([ sharp_rad_sld ,  ft.Container(sharp_rad_txt,on_click=sharp_rad_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0)   
        #======sharp_spred==================sharp_spred======================sharp_spred=====================
        def sharp_spred_c_0(e):
            cash_flet.sharp_spred = 15.0
            sharp_spred_txt.value =" " + str(round(15.0,1)) 
            sharp_spred_sld.value = 15.0
            go_go(e)
            tabss.tabs[2].update()

        def sharp_spred_c(e):
            cash_flet.sharp_spred = e.control.value
            sharp_spred_txt.value = " "+str(round(e.control.value,1)) 
            tabss.tabs[2].update()


        sharp_spred_txt = ft.Text(" " + str(round(cash_flet.sharp_spred,1)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        sharp_spred_sld = ft.Slider(min=1.0, max=30.0,value=15.0, divisions=100, on_change_end=go_go,on_change=sharp_spred_c,expand=True,active_color=ft.colors.GREY_300) 
        sharp_spred = ft.Column(controls=[ft.Container(ft.Text("Spread",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="sharp_spred_sld", on_hover=sldr_hover,content=ft.Row([ sharp_spred_sld ,  ft.Container(sharp_spred_txt,on_click=sharp_spred_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======sharp_amplif==================sharp_amplif======================sharp_amplif=====================
        def sharp_amplif_c_0(e):
            cash_flet.sharp_amplif = 0.0
            sharp_amplif_txt.value =" " + str(round(0.0)) 
            sharp_amplif_sld.value = 0.0
            go_go(e)
            tabss.tabs[2].update()

        def sharp_amplif_c(e):
            cash_flet.sharp_amplif = e.control.value
            sharp_amplif_txt.value = " "+str(round(e.control.value)) 
            tabss.tabs[2].update()


        sharp_amplif_txt = ft.Text(" " + str(round(cash_flet.sharp_amplif)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        sharp_amplif_sld = ft.Slider(min=0.0, max=300.0,value=0.0, divisions=300, on_change_end=go_go,on_change=sharp_amplif_c,expand=True,active_color=ft.colors.GREY_300) 
        sharp_amplif = ft.Column(controls=[ft.Container(ft.Text("Amplify",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="sharp_amplif_sld", on_hover=sldr_hover,content=ft.Row([ sharp_amplif_sld ,  ft.Container(sharp_amplif_txt,on_click=sharp_amplif_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        
        #Weels______________#Weels______________#Weels______________#Weels______________#Weels______________#Weels______________
        #Weels______________#Weels______________#Weels______________#Weels______________#Weels______________#Weels______________
        #Weels______________#Weels______________#Weels______________#Weels______________#Weels______________#Weels______________
        #Weels______________#Weels______________#Weels______________#Weels______________#Weels______________#Weels______________
        #Weels______________#Weels______________#Weels______________#Weels______________#Weels______________#Weels______________
        #Weels______________#Weels______________#Weels______________#Weels______________#Weels______________#Weels______________
        #Weels______________#Weels______________#Weels______________#Weels______________#Weels______________#Weels______________
        
        def update_plot():
            cash_flet.all = Not_multi_funk.plot_zone(cash_flet.width_shad,cash_flet.steepness_shad,
                                                     cash_flet.width_light,cash_flet.steepness_light)
        #======width_shad==================width_shad======================width_shad=====================
        def width_shad_c_0(e):
            cash_flet.width_shad = 0.25
            width_shad_txt.value =" " + str((0.25)) 
            width_shad_sld.value = 0.25
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def width_shad_c(e):
            cash_flet.width_shad = e.control.value
            width_shad_txt.value = " "+str((e.control.value)) 
            update_plot()
            tabss.tabs[2].update()


        width_shad_txt = ft.Text(" " + str((cash_flet.width_shad)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        width_shad_sld = ft.Slider(min=0.0, max=2.0,value=0.25, divisions=50, on_change_end=go_go,on_change=width_shad_c,expand=True,active_color=ft.colors.GREY_300) 
        width_shad = ft.Column(controls=[ft.Container(ft.Text("Width shadow",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="width_shad_sld", on_hover=sldr_hover,content=ft.Row([ width_shad_sld ,  ft.Container(width_shad_txt,on_click=width_shad_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======steepness_shad==================steepness_shad======================steepness_shad=====================
        def steepness_shad_c_0(e):
            cash_flet.steepness_shad = 6.0
            steepness_shad_txt.value =" " + str((6.0)) 
            steepness_shad_sld.value = 6.0
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def steepness_shad_c(e):
            cash_flet.steepness_shad = e.control.value
            steepness_shad_txt.value = " "+str((e.control.value)) 
            update_plot()
            tabss.tabs[2].update()


        steepness_shad_txt = ft.Text(" " + str((cash_flet.steepness_shad)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        steepness_shad_sld = ft.Slider(min=1.0, max=25.0,value=6.0, divisions=50, on_change_end=go_go,on_change=steepness_shad_c,expand=True,active_color=ft.colors.GREY_300) 
        steepness_shad = ft.Column(controls=[ft.Container(ft.Text("Steepness shadow",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="steepness_shad_sld", on_hover=sldr_hover,content=ft.Row([ steepness_shad_sld ,  ft.Container(steepness_shad_txt,on_click=steepness_shad_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======width_light==================width_light======================width_light=====================
        def width_light_c_0(e):
            cash_flet.width_light = 0.25
            width_light_txt.value =" " + str((0.25)) 
            width_light_sld.value = 0.25
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def width_light_c(e):
            cash_flet.width_light = e.control.value
            width_light_txt.value = " "+str((e.control.value)) 
            update_plot()
            tabss.tabs[2].update()
        width_light_txt = ft.Text(" " + str((cash_flet.width_light)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        width_light_sld = ft.Slider(min=0.0, max=2.0,value=0.25, divisions=50, on_change_end=go_go,on_change=width_light_c,expand=True,active_color=ft.colors.GREY_300) 
        width_light = ft.Column(controls=[ft.Container(ft.Text("Width highlight",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="width_light_sld", on_hover=sldr_hover,content=ft.Row([ width_light_sld ,  ft.Container(width_light_txt,on_click=width_light_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======steepness_light==================steepness_light======================steepness_light=====================
        def steepness_light_c_0(e):
            cash_flet.steepness_light = 6.0
            steepness_light_txt.value =" " + str((6.0)) 
            steepness_light_sld.value = 6.0
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def steepness_light_c(e):
            cash_flet.steepness_light = e.control.value
            steepness_light_txt.value = " "+str((e.control.value)) 
            update_plot()
            tabss.tabs[2].update()


        steepness_light_txt = ft.Text(" " + str((cash_flet.steepness_light)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        steepness_light_sld = ft.Slider(min=1.0, max=25.0,value=6.0, divisions=50, on_change_end=go_go,on_change=steepness_light_c,expand=True,active_color=ft.colors.GREY_300) 
        steepness_light = ft.Column(controls=[ft.Container(ft.Text("Steepness highlight",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="steepness_light_sld", on_hover=sldr_hover,content=ft.Row([ steepness_light_sld ,  ft.Container(steepness_light_txt,on_click=steepness_light_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 


        
        plote = MatplotlibChart(cash_flet.all,expand=True,transparent = True)
        mask_weels = ft.Tabs(tabs=[
                                    ft.Tab("Shadow mask",content=ft.Column(controls=[width_shad,steepness_shad])),
                                    ft.Tab("Light mask",content=ft.Column(controls=[width_light,steepness_light]))
                                    ])

        #======neutral_mask==================neutral_mask======================neutral_mask=====================
        def neutral_mask_c_0(e):
            cash_flet.neutral_mask = -1.0
            neutral_mask_txt.value =" " + str((-1.0)) 
            neutral_mask_sld.value = -1.0
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def neutral_mask_c(e):
            cash_flet.neutral_mask = e.control.value
            neutral_mask_txt.value = " "+str((e.control.value)) 
            update_plot()
            tabss.tabs[2].update()


        neutral_mask_txt = ft.Text(" " + str((cash_flet.neutral_mask)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        neutral_mask_sld = ft.Slider(min=-2.0, max=1.0,value=-1.0, divisions=50, on_change_end=go_go,on_change=neutral_mask_c,expand=True,active_color=ft.colors.GREY_300) 
        neutral_mask = ft.Column(controls=[ft.Container(ft.Text("Neutral involve",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="neutral_mask_sld", on_hover=sldr_hover,content=ft.Row([ neutral_mask_sld ,  ft.Container(neutral_mask_txt,on_click=neutral_mask_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======amplyfy_wheel==================amplyfy_wheel======================amplyfy_wheel=====================
        def amplyfy_wheel_c_0(e):
            cash_flet.amplyfy_wheel = 10.0
            amplyfy_wheel_txt.value =" " + str((10.0)) 
            amplyfy_wheel_sld.value = 10.0
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def amplyfy_wheel_c(e):
            cash_flet.amplyfy_wheel = e.control.value
            amplyfy_wheel_txt.value = " "+str((e.control.value)) 
            update_plot()
            tabss.tabs[2].update()


        amplyfy_wheel_txt = ft.Text(" " + str((cash_flet.amplyfy_wheel)),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        amplyfy_wheel_sld = ft.Slider(min=1.0, max=20.0,value=10.0, divisions=50, on_change_end=go_go,on_change=amplyfy_wheel_c,expand=True,active_color=ft.colors.GREY_300) 
        amplyfy_wheel = ft.Column(controls=[ft.Container(ft.Text("Amplyfy shift",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="amplyfy_wheel_sld", on_hover=sldr_hover,content=ft.Row([ amplyfy_wheel_sld ,  ft.Container(amplyfy_wheel_txt,on_click=amplyfy_wheel_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======shad_a==================shad_a======================shad_a=====================
        def shad_a_c_0(e):
            cash_flet.shad_a = 0.0
            shad_a_txt.value =" " + str((0.0)) 
            shad_a_sld.value = 0.0
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def shad_a_c(e):
            cash_flet.shad_a = e.control.value
            shad_a_txt.value = " "+str((round(e.control.value,2))) 
            update_plot()
            tabss.tabs[2].update()


        shad_a_txt = ft.Text(" " + str((round(cash_flet.shad_a,2))),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        shad_a_sld = ft.Slider(min=-3.0, max=3.0,value=0.0, divisions=300, on_change_end=go_go,on_change=shad_a_c,expand=True,active_color=ft.colors.GREEN_300,inactive_color=ft.colors.PINK_300) 
        shad_a = ft.Column(controls=[ft.Container(ft.Text("Shadow shift",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="shad_a_sld", on_hover=sldr_hover,content=ft.Row([ shad_a_sld ,  ft.Container(shad_a_txt,on_click=shad_a_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======shad_b==================shad_b======================shad_b=====================
        def shad_b_c_0(e):
            cash_flet.shad_b = 0.0
            shad_b_txt.value =" " + str((0.0)) 
            shad_b_sld.value = 0.0
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def shad_b_c(e):
            cash_flet.shad_b = e.control.value
            shad_b_txt.value = " "+str((round(e.control.value,2))) 
            update_plot()
            tabss.tabs[2].update()


        shad_b_txt = ft.Text(" " + str((round(cash_flet.shad_b,2))),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        shad_b_sld = ft.Slider(min=-3.0, max=3.0,value=0.0, divisions=300, on_change_end=go_go,on_change=shad_b_c,expand=True,active_color=ft.colors.CYAN_300,inactive_color=ft.colors.YELLOW_300) 
        shad_b = ft.Column(controls=[#ft.Container(ft.Text("Shadow shift",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="shad_b_sld", on_hover=sldr_hover,content=ft.Row([ shad_b_sld ,  ft.Container(shad_b_txt,on_click=shad_b_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======mid_a==================mid_a======================mid_a=====================
        def mid_a_c_0(e):
            cash_flet.mid_a = 0.0
            mid_a_txt.value =" " + str((0.0)) 
            mid_a_sld.value = 0.0
            update_plot()
            tabss.tabs[2].update()

        def mid_a_c(e):
            cash_flet.mid_a = e.control.value
            mid_a_txt.value = " "+str((round(e.control.value,2))) 
            update_plot()
            tabss.tabs[2].update()


        mid_a_txt = ft.Text(" " + str((round(cash_flet.mid_a,2))),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        mid_a_sld = ft.Slider(min=-3.0, max=3.0,value=0.0, divisions=300, on_change_end=go_go,on_change=mid_a_c,expand=True,active_color=ft.colors.GREEN_300,inactive_color=ft.colors.PINK_300) 
        mid_a = ft.Column(controls=[ft.Container(ft.Text("Mid shift",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="mid_a_sld", on_hover=sldr_hover,content=ft.Row([ mid_a_sld ,  ft.Container(mid_a_txt,on_click=mid_a_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======mid_b==================mid_b======================mid_b=====================
        def mid_b_c_0(e):
            cash_flet.mid_b = 0.0
            mid_b_txt.value =" " + str((0.0)) 
            mid_b_sld.value = 0.0
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def mid_b_c(e):
            cash_flet.mid_b = e.control.value
            mid_b_txt.value = " "+str((round(e.control.value,2))) 
            update_plot()
            tabss.tabs[2].update()


        mid_b_txt = ft.Text(" " + str((round(cash_flet.mid_b,2))),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        mid_b_sld = ft.Slider(min=-3.0, max=3.0,value=0.0, divisions=300, on_change_end=go_go,on_change=mid_b_c,expand=True,active_color=ft.colors.CYAN_300,inactive_color=ft.colors.YELLOW_300) 
        mid_b = ft.Column(controls=[#ft.Container(ft.Text("Shadow shift",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="mid_b_sld", on_hover=sldr_hover,content=ft.Row([ mid_b_sld ,  ft.Container(mid_b_txt,on_click=mid_b_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======light_a==================light_a======================light_a=====================
        def light_a_c_0(e):
            cash_flet.light_a = 0.0
            light_a_txt.value =" " + str((0.0)) 
            light_a_sld.value = 0.0
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def light_a_c(e):
            cash_flet.light_a = e.control.value
            light_a_txt.value = " "+str((round(e.control.value,2))) 
            update_plot()
            tabss.tabs[2].update()


        light_a_txt = ft.Text(" " + str((round(cash_flet.light_a,2))),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        light_a_sld = ft.Slider(min=-3.0, max=3.0,value=0.0, divisions=300, on_change_end=go_go,on_change=light_a_c,expand=True,active_color=ft.colors.GREEN_300,inactive_color=ft.colors.PINK_300) 
        light_a = ft.Column(controls=[ft.Container(ft.Text("Light shift",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="light_a_sld", on_hover=sldr_hover,content=ft.Row([ light_a_sld ,  ft.Container(light_a_txt,on_click=light_a_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        #======light_b==================light_b======================light_b=====================
        def light_b_c_0(e):
            cash_flet.light_b = 0.0
            light_b_txt.value =" " + str((0.0)) 
            light_b_sld.value = 0.0
            go_go(e)
            update_plot()
            tabss.tabs[2].update()

        def light_b_c(e):
            cash_flet.light_b = e.control.value
            light_b_txt.value = " "+str((round(e.control.value,2))) 
            update_plot()
            tabss.tabs[2].update()


        light_b_txt = ft.Text(" " + str((round(cash_flet.light_b,2))),overflow=ft.TextOverflow.CLIP,width=50,max_lines=1,theme_style=ft.TextThemeStyle.TITLE_SMALL,color=ft.colors.GREY_700)
        light_b_sld = ft.Slider(min=-3.0, max=3.0,value=0.0, divisions=300, on_change_end=go_go,on_change=light_b_c,expand=True,active_color=ft.colors.CYAN_300,inactive_color=ft.colors.YELLOW_300) 
        light_b = ft.Column(controls=[#ft.Container(ft.Text("Shadow shift",),padding=ft.padding.only(left=shirina/50,top=shirina/50)),
                                  ft.Container(key="light_b_sld", on_hover=sldr_hover,content=ft.Row([ light_b_sld ,  ft.Container(light_b_txt,on_click=light_b_c_0,on_hover = on_hover,border_radius=10,bgcolor=ft.colors.WHITE)],spacing=0))
                                  ],spacing=0) 
        def back_to_in(e):
            imgs = pkl.load(open("imgs.pkl",'rb'))
            for i in imgs:
                  if imgs[i][5] == True:
                        imgs[i][3] = False
                        imgs[i][2] = True
                        imgs[i][5] = False
                        os.remove(os.path.abspath("rendered/"+imgs[i][1]+".jpeg"))
            with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)


            grid = grid_rer()
            tabss.tabs[0].content.controls[1].controls.remove(tabss.tabs[0].content.controls[1].controls[1])
            tabss.tabs[0].content.controls[1].controls.append(grid)
            grid_quewe_tup = grid_quewe_rer()
            grid_quewe = grid_quewe_tup[0]
            tabss.tabs[1].content.controls.remove(tabss.tabs[1].content.controls[0])
            tabss.tabs[1].content.controls.append(grid_quewe)
            grid_quewe_out_tup = grid_quewe_out_rer()
            grid_quewe_out = grid_quewe_out_tup[0]
            tabss.tabs[3].content.controls[1].controls.remove(tabss.tabs[3].content.controls[1].controls[1])
            tabss.tabs[3].content.controls[1].controls.append(grid_quewe_out)
            tabss.update()


        def add_to_render(e):
            Preset_dict = pkl.load(open("Presets.pkl",'rb'))
            imgs = pkl.load(open("imgs.pkl",'rb'))
            name = cash_flet.uploaded_file[-12:-4]
            imgs[name][2] = False
            imgs[name][3] = True
            params = {  "cash_flet.resolution": cash_flet.resolution,               "cash_flet.uploaded_file": cash_flet.uploaded_file,
                        "cash_flet.Wb" : cash_flet.Wb,                              "cash_flet.blur_rad": cash_flet.blur_rad,
                        "cash_flet.halation": cash_flet.halation,                   "cash_flet.blur_spred": cash_flet.blur_spred,
                        "cash_flet.bloom_rad": cash_flet.bloom_rad,                 "cash_flet.bloom_Halation": cash_flet.bloom_Halation,
                        "cash_flet.bloom_spred": cash_flet.bloom_spred,             "cash_flet.sharp_rad": cash_flet.sharp_rad,
                        "cash_flet.sharp_spred": cash_flet.sharp_spred,             "cash_flet.sharp_amplif": cash_flet.sharp_amplif,
                        "cash_flet.camera": cash_flet.camera,                       "cash_flet.r_hue": cash_flet.r_hue,
                        "cash_flet.r_sut": cash_flet.r_sut,                         "cash_flet.r_g": cash_flet.r_g,
                        "cash_flet.r_b": cash_flet.r_b,                             "cash_flet.g_hue": cash_flet.g_hue,
                        "cash_flet.g_sut": cash_flet.g_sut,                         "cash_flet.g_r": cash_flet.g_r,
                        "cash_flet.g_b": cash_flet.g_b,                             "cash_flet.b_hue": cash_flet.b_hue,
                        "cash_flet.b_sut": cash_flet.b_sut,                         "cash_flet.b_r": cash_flet.b_r,
                        "cash_flet.b_g": cash_flet.b_g,                             "cash_flet.film" : cash_flet.film,
                        "None" : "None",                                            "cash_flet.sut": cash_flet.sut,
                        "cash_flet.end_a_plus": cash_flet.end_a_plus,               "cash_flet.end_a_minus": cash_flet.end_a_minus,
                        "cash_flet.end_b_plus": cash_flet.end_b_plus,               "cash_flet.end_b_minus": cash_flet.end_b_minus,
                        "cash_flet.sut_a_minus": cash_flet.sut_a_minus,             "cash_flet.sut_a_plus": cash_flet.sut_a_plus,
                        "cash_flet.sut_b_minus": cash_flet.sut_b_minus,             "cash_flet.sut_b_plus": cash_flet.sut_b_plus,
                        "0": 0,                                      "cash_flet.sut_compr": cash_flet.sut_compr,                                 "cash_flet.steepness_shad": cash_flet.steepness_shad,       "cash_flet.width_shad": cash_flet.width_shad,
                        "cash_flet.steepness_light": cash_flet.steepness_light,     "cash_flet.width_light": cash_flet.width_light,
                        "cash_flet.shad_a": cash_flet.shad_a,                       "cash_flet.shad_b": cash_flet.shad_b,
                        "cash_flet.mid_a": cash_flet.mid_a,                         "cash_flet.mid_b": cash_flet.mid_b,
                        "cash_flet.light_a": cash_flet.light_a,                     "cash_flet.light_b": cash_flet.light_b,
                        "cash_flet.neutral_mask": cash_flet.neutral_mask,           "cash_flet.amplyfy_wheel": cash_flet.amplyfy_wheel,
                        "cash_flet.wbb": cash_flet.wbb,                             "cash_flet.wbr": cash_flet.wbr,
                        "cash_flet.print_exp": cash_flet.print_exp,                 "cash_flet.light_compr": cash_flet.light_compr,
                        "cash_flet.zone": cash_flet.zone,                           "cash_flet.print_cont": cash_flet.print_cont,
                        "cash_flet.gamma": cash_flet.gamma,                         "cash_flet.wbr_scan": cash_flet.wbr_scan,
                        "cash_flet.wbb_scan": cash_flet.wbb_scan,                   "cash_flet.amplify_grain": cash_flet.amplify_grain,
                        "cash_flet.amplify_mask": cash_flet.amplify_mask,           "cash_flet.on_grain": cash_flet.on_grain,
                        "cash_flet.white_point": cash_flet.white_point}



            imgs[name][4] = deepcopy(params)
            print (imgs[name][4])
            to_save = Image.fromarray(cash_flet.pil_img)
            to_save.save("rendered/"+name+".jpeg", format="JPEG", quality=100)

            
            param_drop = preset_dict_rer(1)
            Presets = ft.Column(controls=[ft.ExpansionTile(title=ft.Text("Presets"),initially_expanded=True,controls=[ft.Container(content = ft.Column(controls=[ 
                                                                                                    param_drop,apply_preset,save_preset,del_preset_button
                                                                                                        ],horizontal_alignment=ft.CrossAxisAlignment.CENTER),border=ft.border.all(1, ft.colors.GREY_800),border_radius=5,padding=10,margin=10 )
                                                                                    ])],)
            tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls.remove(tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls[0])
            tabss.tabs[2].content.controls[0].controls[0].tabs[0].content.controls[0].controls.append(Presets.controls[0])



            with open('imgs.pkl', 'wb') as fp:
                    pkl.dump(imgs, fp)

            grid = grid_rer()
            tabss.tabs[0].content.controls[1].controls.remove(tabss.tabs[0].content.controls[1].controls[1])
            tabss.tabs[0].content.controls[1].controls.append(grid)
            grid_quewe_tup = grid_quewe_rer()
            grid_quewe = grid_quewe_tup[0]
            tabss.tabs[1].content.controls.remove(tabss.tabs[1].content.controls[0])
            tabss.tabs[1].content.controls.append(grid_quewe)
            grid_quewe_out_tup = grid_quewe_out_rer()
            grid_quewe_out = grid_quewe_out_tup[0]
            tabss.tabs[3].content.controls[1].controls.remove(tabss.tabs[3].content.controls[1].controls[1])
            tabss.tabs[3].content.controls[1].controls.append(grid_quewe_out)
            tabss.selected_index = 1
            tabss.update()

        plote = MatplotlibChart(cash_flet.all,expand=True,transparent = True)
        mask_weels = ft.Tabs(tabs=[
                                    ft.Tab("Shadow mask",content=ft.Column(controls=[width_shad,steepness_shad])),
                                    ft.Tab("Light mask",content=ft.Column(controls=[width_light,steepness_light]))
                                    ])

        def grab_style_c(e):
             cash_flet.grab_style_image = deepcopy(cash_flet.image_string)
             cash_flet.grab_style_params = deepcopy({  "cash_flet.resolution": cash_flet.resolution,               
                        "cash_flet.Wb" : cash_flet.Wb,                              "cash_flet.blur_rad": cash_flet.blur_rad,
                        "cash_flet.halation": cash_flet.halation,                   "cash_flet.blur_spred": cash_flet.blur_spred,
                        "cash_flet.bloom_rad": cash_flet.bloom_rad,                 "cash_flet.bloom_Halation": cash_flet.bloom_Halation,
                        "cash_flet.bloom_spred": cash_flet.bloom_spred,             "cash_flet.sharp_rad": cash_flet.sharp_rad,
                        "cash_flet.sharp_spred": cash_flet.sharp_spred,             "cash_flet.sharp_amplif": cash_flet.sharp_amplif,
                        "cash_flet.camera": cash_flet.camera,                       "cash_flet.r_hue": cash_flet.r_hue,
                        "cash_flet.r_sut": cash_flet.r_sut,                         "cash_flet.r_g": cash_flet.r_g,
                        "cash_flet.r_b": cash_flet.r_b,                             "cash_flet.g_hue": cash_flet.g_hue,
                        "cash_flet.g_sut": cash_flet.g_sut,                         "cash_flet.g_r": cash_flet.g_r,
                        "cash_flet.g_b": cash_flet.g_b,                             "cash_flet.b_hue": cash_flet.b_hue,
                        "cash_flet.b_sut": cash_flet.b_sut,                         "cash_flet.b_r": cash_flet.b_r,
                        "cash_flet.b_g": cash_flet.b_g,                             "cash_flet.film" : cash_flet.film,
                        "None" : "None",                                            "cash_flet.sut": cash_flet.sut,
                        "cash_flet.end_a_plus": cash_flet.end_a_plus,               "cash_flet.end_a_minus": cash_flet.end_a_minus,
                        "cash_flet.end_b_plus": cash_flet.end_b_plus,               "cash_flet.end_b_minus": cash_flet.end_b_minus,
                        "cash_flet.sut_a_minus": cash_flet.sut_a_minus,             "cash_flet.sut_a_plus": cash_flet.sut_a_plus,
                        "cash_flet.sut_b_minus": cash_flet.sut_b_minus,             "cash_flet.sut_b_plus": cash_flet.sut_b_plus,
                        "0": 0,                                      "cash_flet.sut_compr": cash_flet.sut_compr,                                 "cash_flet.steepness_shad": cash_flet.steepness_shad,       "cash_flet.width_shad": cash_flet.width_shad,
                        "cash_flet.steepness_light": cash_flet.steepness_light,     "cash_flet.width_light": cash_flet.width_light,
                        "cash_flet.shad_a": cash_flet.shad_a,                       "cash_flet.shad_b": cash_flet.shad_b,
                        "cash_flet.mid_a": cash_flet.mid_a,                         "cash_flet.mid_b": cash_flet.mid_b,
                        "cash_flet.light_a": cash_flet.light_a,                     "cash_flet.light_b": cash_flet.light_b,
                        "cash_flet.neutral_mask": cash_flet.neutral_mask,           "cash_flet.amplyfy_wheel": cash_flet.amplyfy_wheel,
                        "cash_flet.wbb": cash_flet.wbb,                             "cash_flet.wbr": cash_flet.wbr,
                        "cash_flet.print_exp": cash_flet.print_exp,                 "cash_flet.light_compr": cash_flet.light_compr,
                        "cash_flet.zone": cash_flet.zone,                           "cash_flet.print_cont": cash_flet.print_cont,
                        "cash_flet.gamma": cash_flet.gamma,                         "cash_flet.wbr_scan": cash_flet.wbr_scan,
                        "cash_flet.wbb_scan": cash_flet.wbb_scan,                   "cash_flet.amplify_grain": cash_flet.amplify_grain,
                        "cash_flet.amplify_mask": cash_flet.amplify_mask,           "cash_flet.on_grain": cash_flet.on_grain,
                        "cash_flet.white_point": cash_flet.white_point})

        def aplply_style(e):
                if cash_flet.grab_style_params!=None:
                    params =cash_flet.grab_style_params
                    cash_flet.resolution=params["cash_flet.resolution"]
                    cash_flet.Wb=params["cash_flet.Wb"]
                    cash_flet.blur_rad=params["cash_flet.blur_rad"]
                    cash_flet.halation=params["cash_flet.halation"]
                    cash_flet.blur_spred=params["cash_flet.blur_spred"]
                    cash_flet.bloom_rad=params["cash_flet.bloom_rad"]
                    cash_flet.bloom_Halation=params["cash_flet.bloom_Halation"]
                    cash_flet.bloom_spred=params["cash_flet.bloom_spred"]
                    cash_flet.sharp_rad=params["cash_flet.sharp_rad"]
                    cash_flet.sharp_spred=params["cash_flet.sharp_spred"]
                    cash_flet.sharp_amplif=params["cash_flet.sharp_amplif"]
                    cash_flet.r_b=params["cash_flet.r_b"]
                    cash_flet.print_exp=params["cash_flet.print_exp"]
                    cash_flet.wbr_scan=params["cash_flet.wbr_scan"]
                    cash_flet.wbb_scan=params["cash_flet.wbb_scan"]
                    cash_flet.camera=params["cash_flet.camera"]
                    cash_flet.r_hue=params["cash_flet.r_hue"]
                    cash_flet.r_sut=params["cash_flet.r_sut"]
                    cash_flet.r_g=params["cash_flet.r_g"]           
                    cash_flet.g_hue=params["cash_flet.g_hue"]
                    cash_flet.g_sut=params["cash_flet.g_sut"]
                    cash_flet.g_r=params["cash_flet.g_r"]
                    cash_flet.g_b=params["cash_flet.g_b"]
                    cash_flet.b_hue=params["cash_flet.b_hue"]
                    cash_flet.b_sut=params["cash_flet.b_sut"]
                    cash_flet.b_r=params["cash_flet.b_r"]
                    cash_flet.b_g=params["cash_flet.b_g"]
                    cash_flet.film=params["cash_flet.film"]
                    cash_flet.sut=params["cash_flet.sut"]
                    cash_flet.light_a=params["cash_flet.light_a"]
                    cash_flet.light_b=params["cash_flet.light_b"]
                    cash_flet.end_a_plus=params["cash_flet.end_a_plus"]
                    cash_flet.end_a_minus=params["cash_flet.end_a_minus"]
                    cash_flet.end_b_plus=params["cash_flet.end_b_plus"]
                    cash_flet.end_b_minus=params["cash_flet.end_b_minus"]
                    cash_flet.sut_a_minus=params["cash_flet.sut_a_minus"]
                    cash_flet.sut_a_plus=params["cash_flet.sut_a_plus"]
                    cash_flet.sut_b_minus=params["cash_flet.sut_b_minus"]
                    cash_flet.sut_b_plus=params["cash_flet.sut_b_plus"]
                    cash_flet.sut_compr=params["cash_flet.sut_compr"]
                    cash_flet.steepness_shad=params["cash_flet.steepness_shad"]
                    cash_flet.width_shad=params["cash_flet.width_shad"]
                    cash_flet.steepness_light=params["cash_flet.steepness_light"]
                    cash_flet.width_light=params["cash_flet.width_light"]
                    cash_flet.shad_a=params["cash_flet.shad_a"]
                    cash_flet.shad_b=params["cash_flet.shad_b"]
                    cash_flet.mid_a=params["cash_flet.mid_a"]
                    cash_flet.mid_b=params["cash_flet.mid_b"]              
                    cash_flet.neutral_mask=params["cash_flet.neutral_mask"]
                    cash_flet.amplyfy_wheel=params["cash_flet.amplyfy_wheel"]
                    cash_flet.wbb=params["cash_flet.wbb"]
                    cash_flet.wbr=params["cash_flet.wbr"]       
                    cash_flet.light_compr=params["cash_flet.light_compr"]
                    cash_flet.zone=params["cash_flet.zone"]
                    cash_flet.print_cont=params["cash_flet.print_cont"]
                    cash_flet.gamma=params["cash_flet.gamma"]        
                    cash_flet.amplify_grain=params["cash_flet.amplify_grain"]
                    cash_flet.amplify_mask=params["cash_flet.amplify_mask"]
                    cash_flet.on_grain=params["cash_flet.on_grain"]
                    cash_flet.white_point=params["cash_flet.white_point"]
                    resolution.value = cash_flet.resolution
                    wb.value = cash_flet.Wb
                    blur_rad_sld.value = cash_flet.blur_rad
                    blur_rad_txt.value = cash_flet.blur_rad

                    halation_sld.value = cash_flet.halation
                    halation_txt.value = cash_flet.halation
                    
                    blur_spred_sld.value = cash_flet.blur_spred
                    blur_spred_txt.value = cash_flet.blur_spred

                    bloom_rad_sld.value = cash_flet.bloom_rad
                    bloom_rad_txt.value = cash_flet.bloom_rad

                    bloom_Halation_sld.value = cash_flet.bloom_Halation
                    bloom_Halation_txt.value = cash_flet.bloom_Halation

                    bloom_spred_sld.value = cash_flet.bloom_spred
                    bloom_spred_txt.value = cash_flet.bloom_spred

                    sharp_rad_sld.value = cash_flet.sharp_rad
                    sharp_rad_txt.value = cash_flet.sharp_rad

                    sharp_spred_sld.value = cash_flet.sharp_spred
                    sharp_spred_txt.value = cash_flet.sharp_spred

                    sharp_amplif_sld.value = cash_flet.sharp_amplif
                    sharp_amplif_txt.value = cash_flet.sharp_amplif

                    cam_drop.value = cash_flet.camera

                    r_hue_sld.value = cash_flet.r_hue
                    r_hue_txt.value = cash_flet.r_hue

                    r_sut_sld.value = cash_flet.r_sut
                    r_sut_txt.value = cash_flet.r_sut

                    r_g_sld.value = cash_flet.r_g
                    r_g_txt.value = cash_flet.r_g

                    r_b_sld.value = cash_flet.r_b
                    r_b_txt.value = cash_flet.r_b

                    g_hue_sld.value = cash_flet.g_hue
                    g_hue_txt.value = cash_flet.g_hue

                    g_sut_sld.value = cash_flet.g_sut
                    g_sut_txt.value = cash_flet.g_sut

                    g_r_sld.value = cash_flet.g_r
                    g_r_txt.value = cash_flet.g_r

                    g_b_sld.value = cash_flet.g_b
                    g_b_txt.value = cash_flet.g_b

                    b_hue_sld.value = cash_flet.b_hue
                    b_hue_txt.value = cash_flet.b_hue

                    b_sut_sld.value = cash_flet.b_sut
                    b_sut_txt.value = cash_flet.b_sut

                    b_r_sld.value = cash_flet.b_r
                    b_r_txt.value = cash_flet.b_r

                    b_g_sld.value = cash_flet.b_g
                    b_g_txt.value = cash_flet.b_g

                    film_drop.value = cash_flet.film

                    sut_sld.value = cash_flet.sut
                    sut_txt.value = cash_flet.sut

                    end_a_plus_sld.value = cash_flet.end_a_plus
                    end_a_plus_txt.value = cash_flet.end_a_plus

                    end_a_minus_sld.value = cash_flet.end_a_minus
                    end_a_minus_txt.value = cash_flet.end_a_minus

                    end_b_plus_sld.value = cash_flet.end_b_plus
                    end_b_plus_txt.value = cash_flet.end_b_plus

                    end_b_minus_sld.value = cash_flet.end_b_minus
                    end_b_minus_txt.value = cash_flet.end_b_minus

                    sut_a_minus_sld.value = cash_flet.sut_a_minus
                    sut_a_minus_txt.value = cash_flet.sut_a_minus

                    sut_a_plus_sld.value = cash_flet.sut_a_plus
                    sut_a_plus_txt.value = cash_flet.sut_a_plus

                    sut_b_minus_sld.value = cash_flet.sut_b_minus
                    sut_b_minus_txt.value = cash_flet.sut_b_minus

                    sut_b_plus_sld.value = cash_flet.sut_b_plus
                    sut_b_plus_txt.value = cash_flet.sut_b_plus

                    sut_compr_sld.value = cash_flet.sut_compr
                    sut_compr_txt.value = cash_flet.sut_compr

                    steepness_shad_sld.value = cash_flet.steepness_shad
                    steepness_shad_txt.value = cash_flet.steepness_shad

                    width_shad_sld.value = cash_flet.width_shad
                    width_shad_txt.value = cash_flet.width_shad

                    steepness_light_sld.value = cash_flet.steepness_light
                    steepness_light_txt.value = cash_flet.steepness_light

                    width_light_sld.value = cash_flet.width_light
                    width_light_txt.value = cash_flet.width_light

                    shad_a_sld.value = cash_flet.shad_a
                    shad_a_txt.value = cash_flet.shad_a

                    shad_b_sld.value = cash_flet.shad_b
                    shad_b_txt.value = cash_flet.shad_b

                    mid_a_sld.value = cash_flet.mid_a
                    mid_a_txt.value = cash_flet.mid_a

                    mid_b_sld.value = cash_flet.mid_b
                    mid_b_txt.value = cash_flet.mid_b

                    light_a_sld.value = cash_flet.light_a
                    light_a_txt.value = cash_flet.light_a

                    light_b_sld.value = cash_flet.light_b
                    light_b_txt.value = cash_flet.light_b

                    neutral_mask_sld.value = cash_flet.neutral_mask
                    neutral_mask_txt.value = cash_flet.neutral_mask

                    amplyfy_wheel_sld.value = cash_flet.amplyfy_wheel
                    amplyfy_wheel_txt.value = cash_flet.amplyfy_wheel

                    wbb_sld.value = cash_flet.wbb
                    wbb_txt.value = cash_flet.wbb

                    wbr_sld.value = cash_flet.wbr
                    wbr_txt.value = cash_flet.wbr

                    print_exp_sld.value = cash_flet.print_exp
                    print_exp_txt.value = cash_flet.print_exp

                    light_compr_sld.value = cash_flet.light_compr
                    light_compr_txt.value = cash_flet.light_compr

                    zone_sld.value = cash_flet.zone
                    zone_txt.value = cash_flet.zone

                    print_cont_sld.value = cash_flet.print_cont
                    print_cont_txt.value = cash_flet.print_cont

                    gamma_sld.value = cash_flet.gamma
                    gamma_txt.value = cash_flet.gamma

                    wbr_scan_sld.value = cash_flet.wbr_scan
                    wbr_scan_txt.value = cash_flet.wbr_scan

                    wbb_scan_sld.value = cash_flet.wbb_scan
                    wbb_scan_txt.value = cash_flet.wbb_scan

                    amplify_grain_sld.value = cash_flet.amplify_grain
                    amplify_grain_txt.value = cash_flet.amplify_grain

                    amplify_mask_sld.value = cash_flet.amplify_mask
                    amplify_mask_txt.value = cash_flet.amplify_mask

                    on_grain.value = cash_flet.on_grain
                    white_point_sld.value = cash_flet.white_point
                    white_point_txt.value = cash_flet.white_point
                    grab_style.controls[0].content.disabled = False
                    tabss.update()
                    go_go(e)
        

        def show_grabbed (e):
             if cash_flet.grab_style_image!=None:
                    
                    sldrs=[resolution,wb,blur_rad,halation,blur_spred,bloom_rad,bloom_Halation,bloom_spred,sharp_rad,sharp_spred,sharp_amplif,
                    cam_drop,r_hue,r_sut,r_g,r_b,g_hue,g_sut,g_r,g_b,b_hue,b_sut,b_r,b_g,film_drop,sut,end_a_plus,end_a_minus,end_b_plus,end_b_minus,sut_a_minus,sut_a_plus,sut_b_minus,sut_b_plus,sut_compr,
                    steepness_shad,width_shad,steepness_light,width_light,shad_a,shad_b,mid_a,mid_b,light_a,light_b,neutral_mask,amplyfy_wheel,wbb,wbr,print_exp,light_compr,zone,print_cont,gamma,wbr_scan,wbb_scan,amplify_grain,amplify_mask,on_grain,white_point,grab_style.controls[0].content]
                    
                    if img.src_base64 == cash_flet.image_string and cash_flet.image_string!=cash_flet.grab_style_image:
                        img.src_base64 = cash_flet.grab_style_image
                        img.update()
                        for sld in sldrs:
                            sld.disabled = True
                            sld.update()
                    else:
                        img.src_base64 = cash_flet.image_string
                        img.update()
                        for sld in sldrs:
                            sld.disabled = False
                            sld.update()
                         

        grab_style = ft.Row(controls=[  ft.Container(content=ft.ElevatedButton(text="Grab style",on_click=grab_style_c,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800)),margin=ft.margin.only(0,5,0,5)),
                                        ft.Container(content=ft.ElevatedButton(text="Apply style",on_click=aplply_style,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800)),margin=ft.margin.only(0,5,0,5))
                            ],alignment=ft.MainAxisAlignment.CENTER)
        img = ft.Image(src_base64=cash_flet.image_string,fit=ft.ImageFit.FIT_WIDTH,border_radius=10)


        tabss = ft.Tabs(tabs=[
                                    ft.Tab  ("           Gallery           ",content=ft.Column(controls=[
                                      
                                                ft.Row(controls=[
                                                                file_picker,
                                                                ft.Container(content=ft.ElevatedButton(text="Set path",on_click= lambda _: file_picker.get_directory_path(),
                                                                                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800)),margin=10,alignment = ft.alignment.center,expand=1),
                                                                prog_bar,
                                                                ft.ElevatedButton(text="Select all",on_click=select_all,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800),expand=1),
                                                                ft.ElevatedButton(text="Unselect all",on_click=unselect_all,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800),expand=1)
                                    
                                                                    ],tight=True),

                                                ft.Row(controls=[
                                                                main_img_gall,
                                                                grid
                                                

                                                                    ],expand=1,tight=True)
                                                                                    ],expand=1,)),
                                                                
                    
                    
                                    ft.Tab  ("           In            ",content=ft.Row(controls=[
                                                                                                 grid_quewe
                                                                                                 ])
                                            ),
                                    ft.Tab  ("         Grading         ",content=ft.Row(controls=[
                                                                                ft.Column(
                                                                                            controls=[ft.Tabs(tabs=[
                                                                                                                        ft.Tab("Colour",content=ft.Column(controls=[
                                                                                                                                                                    Presets,grab_style,film_prop,wbr,wbb,
                                                                                                                                                                    ft.Divider(height=1, color=ft.colors.GREY_800)
                                                                                                                                                                    ,gamma,print_exp,print_cont,sut,
                                                                                                                                                                    ft.Divider(height=1, color=ft.colors.GREY_800),
                                                                                                                                                                    white_point,light_compr,zone,ProLab,rgb_mix,
                                                                                                                                                                    Params
                                                                                                                                                                    ],spacing=0,horizontal_alignment=ft.CrossAxisAlignment.CENTER)),
                                                                                                                        ft.Tab("Structure",content=ft.Column(controls=[
                                                                                                                                                                    on_grain,amplify_grain,amplify_mask,
                                                                                                                                                                    ft.Divider(height=1, color=ft.colors.GREY_800),
                                                                                                                                                                    ft.Container(ft.Text("Reduse sharpness",weight = ft.FontWeight.W_500,size=18),padding=10),
                                                                                                                                                                    blur_rad,blur_spred,halation,
                                                                                                                                                                    ft.Divider(height=1, color=ft.colors.GREY_800),
                                                                                                                                                                    ft.Container(ft.Text("Bloom",weight = ft.FontWeight.W_500,size=18),padding=10),
                                                                                                                                                                    bloom_rad,bloom_spred,bloom_Halation,
                                                                                                                                                                    ft.Divider(height=1, color=ft.colors.GREY_800),
                                                                                                                                                                    ft.Container(ft.Text("Local contrast",weight = ft.FontWeight.W_500,size=18),padding=10),
                                                                                                                                                                    sharp_rad,sharp_spred,sharp_amplif
                                                                                                                                                                    ],spacing=0,horizontal_alignment=ft.CrossAxisAlignment.CENTER)),
                                                                                                                        ft.Tab("Wheels",content=ft.Column(controls=[
                                                                                                                                                                    
                                                                                                                                                                    ft.Container(plote,bgcolor=ft.colors.GREY_300,padding=0,border_radius=15),
                                                                                                                                                                    ft.Container(mask_weels,height=shirina/3.3),
                                                                                                                                                                    ft.Divider(height=1, color=ft.colors.GREY_800),
                                                                                                                                                                    neutral_mask,amplyfy_wheel,
                                                                                                                                                                    ft.Divider(height=1, color=ft.colors.GREY_800),
                                                                                                                                                                    shad_a,shad_b,mid_a,mid_b,light_a,light_b
                                                                                                                                                                        ],spacing=0,horizontal_alignment=ft.CrossAxisAlignment.CENTER))                                                                                                                                                                   
                                                                                                                    ],label_color=ft.colors.GREY_800,indicator_color=ft.colors.GREY_800,animation_duration=500,tab_alignment=ft.TabAlignment.CENTER
                                                                                                        )],expand=17,scroll=ft.ScrollMode.HIDDEN
                                                                                        ),
                                                                                ft.Container(img,alignment=ft.alignment.center,expand=88,on_click=show_grabbed,padding=ft.padding.only(5,0,5,0)),
                                                                                ft.Column([ft.Container(ft.Icon(name=ft.icons.ARROW_FORWARD_IOS,color=ft.colors.GREY_800),
                                                                                                        on_hover=on_hover_add,on_click=add_to_render,
                                                                                                        expand=1,border=ft.border.all(1, ft.colors.GREY_800),bgcolor=ft.colors.GREY_100,border_radius=5,padding=10,alignment=ft.alignment.center_left)],expand=2)
                                                                                ],expand=True,alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
                                            ),
                                    ft.Tab  ("           Out           ",content=ft.Column(controls=[
                                      
                                                ft.Row(controls=[
                                                                    ft.Row(controls=[
                                                                                file_picker_out,
                                                                                ft.Container(content=ft.ElevatedButton(text="Set path and save selected",on_click= lambda _: file_picker_out.get_directory_path(),
                                                                                                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800)),margin=10,alignment = ft.alignment.center),
                                                                                prog_bar_out,
                                                                                ft.Text("   Resolution on save:"),save_resolution
                                                                                        ],tight=True,expand=1),
                                                                    ft.Row(controls=[            
                                                                                ft.ElevatedButton(text="Selested| Out -> In ",on_long_press=back_to_in,on_hover=out_to_in,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800)),
                                                                                
                                                                                ft.ElevatedButton(text="Select all",on_click=is_render_select_all,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800)),
                                                                                ft.ElevatedButton(text="Unselect all",on_click=is_render_unselect_all,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),color=ft.colors.GREY_800))
                                                                                ])
                                                                ],expand=0.3),

                                                ft.Row(controls=[
                                                                main_img_gall_out,
                                                                grid_quewe_out
                                                

                                                                    ],expand=1)
                                                                                    ],expand=1)),

                                                     




                                    ],expand=1,animation_duration=500,tab_alignment=ft.TabAlignment.CENTER,label_color=ft.colors.GREY_800,indicator_color=ft.colors.GREY_800,indicator_tab_size=True
                            )
        

        def on_keyboard(e: ft.KeyboardEvent):
                            """if tabss.selected_index == 2:
                        def every_to_def():
                                    wbr_scan_sld.active_color = ft.colors.RED_300
                                    wbr_scan.update()
                                    wbb_scan_sld.active_color = ft.colors.BLUE_300
                                    wbb_scan.update()
                                    wbr_sld.active_color = ft.colors.RED_300
                                    wbr.update()
                                    wbb_sld.active_color = ft.colors.BLUE_300
                                    wbb.update()
                                    gamma_sld.active_color = ft.colors.GREY_300
                                    gamma_sld.update()
                                    print_exp_sld.active_color = ft.colors.GREY_300
                                    print_exp_sld.update()
                                    print_cont_sld.active_color = ft.colors.GREY_300
                                    print_cont_sld.update()
                                    sut_sld.active_color = ft.colors.GREY_300
                                    sut_sld.update()
                                    white_point_sld.active_color = ft.colors.GREY_300
                                    white_point_sld.update()
                                    on_grain.content.thumb_color = ft.colors.GREY_500
                                    on_grain.update()
                                    amplify_grain_sld.active_color = ft.colors.GREY_300
                                    amplify_grain_sld.update()
                                    amplify_mask_sld.active_color = ft.colors.GREY_300
                                    amplify_mask_sld.update()
                                    blur_rad_sld.active_color = ft.colors.GREY_300
                                    blur_rad_sld.update()
                                    blur_spred_sld.active_color = ft.colors.GREY_300
                                    blur_spred_sld.update()
                                    halation_sld.active_color = ft.colors.GREY_300
                                    halation_sld.update()
                                    bloom_spred_sld.active_color = ft.colors.GREY_300
                                    bloom_spred_sld.update()
                                    bloom_Halation_sld.active_color = ft.colors.GREY_300
                                    bloom_Halation_sld.update()
                                    bloom_rad_sld.active_color = ft.colors.GREY_300
                                    bloom_rad_sld.update()
                                    sharp_rad_sld.active_color = ft.colors.GREY_300
                                    sharp_rad_sld.update()
                                    sharp_spred_sld.active_color = ft.colors.GREY_300
                                    sharp_spred_sld.update()
                                    sharp_amplif_sld.active_color = ft.colors.GREY_300
                                    sharp_amplif_sld.update()
                                    
                        
                        if e.key == "Q":
                                every_to_def()
                                if cash_flet.selected!="q":
                                    cash_flet.selected = "q"
                                    wbr_scan_sld.active_color = ft.colors.RED_800
                                    wbr_scan.update()
                                    on_grain.content.thumb_color = ft.colors.BLACK
                                    on_grain.update()
                                elif cash_flet.selected=="q":
                                    cash_flet.selected = None
                                    wbr_scan_sld.active_color = ft.colors.RED_300
                                    wbr_scan.update()
                                    on_grain.content.thumb_color = ft.colors.GREY_500
                                    on_grain.update()
                        if e.key == "W":
                                every_to_def()
                                if cash_flet.selected!="w":
                                    cash_flet.selected = "w"
                                    wbb_scan_sld.active_color = ft.colors.BLUE_800
                                    wbb_scan.update()
                                    amplify_grain_sld.active_color = ft.colors.GREY_800
                                    amplify_grain_sld.update()
                                elif cash_flet.selected=="w":
                                    cash_flet.selected = None
                                    wbb_scan_sld.active_color = ft.colors.BLUE_300
                                    wbb_scan.update()
                                    amplify_grain_sld.active_color = ft.colors.GREY_300
                                    amplify_grain_sld.update()
                        if e.key == "E":
                                every_to_def()
                                if cash_flet.selected!="e":
                                    cash_flet.selected = "e"
                                    wbr_sld.active_color = ft.colors.RED_800
                                    wbr.update()
                                    amplify_mask_sld.active_color = ft.colors.GREY_800
                                    amplify_mask_sld.update()
                                elif cash_flet.selected=="e":
                                    cash_flet.selected = None
                                    wbr_sld.active_color = ft.colors.RED_300
                                    wbr.update()
                                    amplify_mask_sld.active_color = ft.colors.GREY_300
                                    amplify_mask_sld.update()

                        if e.key == "R":
                                every_to_def()
                                if cash_flet.selected!="r":
                                    cash_flet.selected = "r"
                                    wbb_sld.active_color = ft.colors.BLUE_800
                                    wbb.update()
                                    sharp_rad_sld.active_color = ft.colors.GREY_800
                                    sharp_rad_sld.update()
                                elif cash_flet.selected=="r":
                                    cash_flet.selected = None
                                    wbb_sld.active_color = ft.colors.BLUE_300
                                    sharp_rad_sld.active_color = ft.colors.GREY_300
                                    sharp_rad_sld.update()
                                    wbb.update()
                        if e.key == "A":
                                every_to_def()
                                if cash_flet.selected!="a":
                                    cash_flet.selected = "a"
                                    gamma_sld.active_color = ft.colors.GREY_800
                                    gamma_sld.update()
                                    blur_rad_sld.active_color = ft.colors.GREY_800
                                    blur_rad_sld.update()
                                elif cash_flet.selected=="a":
                                    cash_flet.selected = None
                                    gamma_sld.active_color = ft.colors.GREY_300
                                    gamma_sld.update()
                                    blur_rad_sld.active_color = ft.colors.GREY_300
                                    blur_rad_sld.update()
                        if e.key == "S":
                                every_to_def()
                                if cash_flet.selected!="s":
                                    cash_flet.selected = "s"
                                    print_exp_sld.active_color = ft.colors.GREY_800
                                    print_exp_sld.update()
                                    blur_spred_sld.active_color = ft.colors.GREY_800
                                    blur_spred_sld.update()
                                elif cash_flet.selected=="s":
                                    cash_flet.selected = None
                                    print_exp_sld.active_color = ft.colors.GREY_300
                                    print_exp_sld.update()
                                    blur_spred_sld.active_color = ft.colors.GREY_300
                                    blur_spred_sld.update()
                        if e.key == "D":
                                every_to_def()
                                if cash_flet.selected!="d":
                                    cash_flet.selected = "d"
                                    print_cont_sld.active_color = ft.colors.GREY_800
                                    print_cont_sld.update()
                                    halation_sld.active_color = ft.colors.GREY_800
                                    halation_sld.update()
                                elif cash_flet.selected=="d":
                                    cash_flet.selected = None
                                    print_cont_sld.active_color = ft.colors.GREY_300
                                    print_cont_sld.update()
                                    halation_sld.active_color = ft.colors.GREY_300
                                    halation_sld.update()
                        if e.key == "F":
                                every_to_def()
                                if cash_flet.selected!="f":
                                    cash_flet.selected = "f"
                                    sut_sld.active_color = ft.colors.GREY_800
                                    sut_sld.update()
                                    sharp_spred_sld.active_color = ft.colors.GREY_800
                                    sharp_spred_sld.update()
                                elif cash_flet.selected=="f":
                                    cash_flet.selected = None
                                    sut_sld.active_color = ft.colors.GREY_300
                                    sut_sld.update()
                                    sharp_spred_sld.active_color = ft.colors.GREY_300
                                    sharp_spred_sld.update()

                        if e.key == "Z":
                                every_to_def()
                                if cash_flet.selected!="z":
                                    cash_flet.selected = "z"
                                    white_point_sld.active_color = ft.colors.GREY_800
                                    white_point_sld.update()
                                    bloom_rad_sld.active_color = ft.colors.GREY_800
                                    bloom_rad_sld.update()
                                elif cash_flet.selected=="z":
                                    cash_flet.selected = None
                                    white_point_sld.active_color = ft.colors.GREY_300
                                    white_point_sld.update()
                                    bloom_rad_sld.active_color = ft.colors.GREY_300
                                    bloom_rad_sld.update()

                        if e.key == "X":
                                every_to_def()
                                if cash_flet.selected!="x":
                                    cash_flet.selected = "x"
                                    bloom_spred_sld.active_color = ft.colors.GREY_800
                                    bloom_spred_sld.update()
                                elif cash_flet.selected=="x":
                                    cash_flet.selected = None
                                    bloom_spred_sld.active_color = ft.colors.GREY_300
                                    bloom_spred_sld.update()

                        if e.key == "C":
                                every_to_def()
                                if cash_flet.selected!="c":
                                    cash_flet.selected = "c"
                                    bloom_Halation_sld.active_color = ft.colors.GREY_800
                                    bloom_Halation_sld.update()
                                elif cash_flet.selected=="c":
                                    cash_flet.selected = None
                                    bloom_Halation_sld.active_color = ft.colors.GREY_300
                                    bloom_Halation_sld.update()

                        if e.key == "V":
                                every_to_def()
                                if cash_flet.selected!="v":
                                    cash_flet.selected = "v"
                                    sharp_amplif_sld.active_color = ft.colors.GREY_800
                                    sharp_amplif_sld.update()
                                elif cash_flet.selected=="v":
                                    cash_flet.selected = None
                                    sharp_amplif_sld.active_color = ft.colors.GREY_300
                                    sharp_amplif_sld.update()"""


                    
                            #__+___+____+____+___+____+___+___+____+____+___+___+__+
                            if e.key == "W": #  >>>>
                                if cash_flet.selected == "wbr_scan_sld":
                                        cash_flet.wbr_scan+=0.01
                                        if cash_flet.wbr_scan>wbr_scan_sld.max:
                                            cash_flet.wbr_scan=wbr_scan_sld.max
                                        wbr_scan_sld.value=cash_flet.wbr_scan
                                        wbr_scan_sld.update()
                                        wbr_scan_txt.value=round(cash_flet.wbr_scan,2)
                                        wbr_scan_txt.update()

                                if cash_flet.selected == "wbb_scan_sld":
                                        cash_flet.wbb_scan+=0.01
                                        if cash_flet.wbb_scan>wbb_scan_sld.max:
                                            cash_flet.wbb_scan=wbb_scan_sld.max
                                        wbb_scan_sld.value=cash_flet.wbb_scan
                                        wbb_scan_sld.update()
                                        wbb_scan_txt.value=round(cash_flet.wbb_scan,2)
                                        wbb_scan_txt.update()                                    

                                if cash_flet.selected == "wbr_sld":
                                        cash_flet.wbr+=0.1
                                        if cash_flet.wbr>wbr_sld.max:
                                            cash_flet.wbr=wbr_sld.max
                                        wbr_sld.value=cash_flet.wbr
                                        wbr_sld.update()
                                        wbr_txt.value=round(cash_flet.wbr,2)
                                        wbr_txt.update()   

                                if cash_flet.selected == "wbb_sld":
                                        cash_flet.wbb+=0.1
                                        if cash_flet.wbb>wbb_sld.max:
                                            cash_flet.wbb=wbb_sld.max
                                        wbb_sld.value=cash_flet.wbb
                                        wbb_sld.update()
                                        wbb_txt.value=round(cash_flet.wbb,2)
                                        wbb_txt.update()   


                                #___________________________________________
                                    
                                if cash_flet.selected == "gamma_sld":
                                        cash_flet.gamma+=0.1
                                        if cash_flet.gamma>gamma_sld.max:
                                            cash_flet.gamma=gamma_sld.max
                                        gamma_sld.value=cash_flet.gamma
                                        gamma_sld.update()
                                        gamma_txt.value=round(cash_flet.gamma,2)
                                        gamma_txt.update()   

                                if cash_flet.selected == "print_exp_sld":
                                        cash_flet.print_exp+=0.1
                                        if cash_flet.print_exp>print_exp_sld.max:
                                            cash_flet.print_exp=print_exp_sld.max
                                        print_exp_sld.value=cash_flet.print_exp
                                        print_exp_sld.update()
                                        print_exp_txt.value=round(cash_flet.print_exp,2)
                                        print_exp_txt.update()   

                                if cash_flet.selected == "print_cont_sld":
                                        cash_flet.print_cont+=0.02
                                        if cash_flet.print_cont>print_cont_sld.max:
                                            cash_flet.print_cont=print_cont_sld.max
                                        print_cont_sld.value=cash_flet.print_cont
                                        print_cont_sld.update()
                                        print_cont_txt.value=round(cash_flet.print_cont,2)
                                        print_cont_txt.update()  

                                if cash_flet.selected == "sut_sld":
                                        cash_flet.sut+=0.05
                                        if cash_flet.sut>sut_sld.max:
                                            cash_flet.sut=sut_sld.max
                                        sut_sld.value=cash_flet.sut
                                        sut_sld.update()
                                        sut_txt.value=round(cash_flet.sut,2)
                                        sut_txt.update()  
                                    

                                #_________________________________________
                                        
                                if cash_flet.selected == "white_point_sld":
                                        cash_flet.white_point+=0.01
                                        if cash_flet.white_point>white_point_sld.max:
                                            cash_flet.white_point=white_point_sld.max
                                        white_point_sld.value=cash_flet.white_point
                                        white_point_sld.update()
                                        white_point_txt.value=round(cash_flet.white_point,2)
                                        white_point_txt.update()  

                                if cash_flet.selected == "light_compr_sld":
                                        cash_flet.light_compr+=0.1
                                        if cash_flet.light_compr>light_compr_sld.max:
                                            cash_flet.light_compr=light_compr_sld.max
                                        light_compr_sld.value=cash_flet.light_compr
                                        light_compr_sld.update()
                                        light_compr_txt.value=round(cash_flet.light_compr,2)
                                        light_compr_txt.update()

                                if cash_flet.selected == "zone_sld":
                                        cash_flet.zone+=0.01
                                        if cash_flet.zone>zone_sld.max:
                                            cash_flet.zone=zone_sld.max
                                        zone_sld.value=cash_flet.zone
                                        zone_sld.update()
                                        zone_txt.value=round(cash_flet.zone,2)
                                        zone_txt.update()

                                    #_____________________________________________
                                    
                                if cash_flet.selected == "amplify_grain_sld":
                                        cash_flet.amplify_grain+=0.01
                                        if cash_flet.amplify_grain>amplify_grain_sld.max:
                                            cash_flet.amplify_grain=amplify_grain_sld.max
                                        amplify_grain_sld.value=cash_flet.amplify_grain
                                        amplify_grain_sld.update()
                                        amplify_grain_txt.value=round(cash_flet.amplify_grain,2)
                                        amplify_grain_txt.update()

                                if cash_flet.selected == "amplify_mask_sld":
                                        cash_flet.amplify_mask+=0.1
                                        if cash_flet.amplify_mask>amplify_mask_sld.max:
                                            cash_flet.amplify_mask=amplify_mask_sld.max
                                        amplify_mask_sld.value=cash_flet.amplify_mask
                                        amplify_mask_sld.update()
                                        amplify_mask_txt.value=round(cash_flet.amplify_mask,2)
                                        amplify_mask_txt.update()

                                if cash_flet.selected == "blur_rad_sld":
                                        cash_flet.blur_rad+=0.1
                                        if cash_flet.blur_rad>blur_rad_sld.max:
                                            cash_flet.blur_rad=blur_rad_sld.max
                                        blur_rad_sld.value=cash_flet.blur_rad
                                        blur_rad_sld.update()
                                        blur_rad_txt.value=round(cash_flet.blur_rad,2)
                                        blur_rad_txt.update()

                                if cash_flet.selected == "blur_spred_sld":
                                        cash_flet.blur_spred+=0.1
                                        if cash_flet.blur_spred>blur_spred_sld.max:
                                            cash_flet.blur_spred=blur_spred_sld.max
                                        blur_spred_sld.value=cash_flet.blur_spred
                                        blur_spred_sld.update()
                                        blur_spred_txt.value=round(cash_flet.blur_spred,2)
                                        blur_spred_txt.update()

                                if cash_flet.selected == "halation_sld":
                                        cash_flet.halation+=0.1
                                        if cash_flet.halation>halation_sld.max:
                                            cash_flet.halation=halation_sld.max
                                        halation_sld.value=cash_flet.halation
                                        halation_sld.update()
                                        halation_txt.value=round(cash_flet.halation,2)
                                        halation_txt.update()

                                if cash_flet.selected == "bloom_rad_sld":
                                        cash_flet.bloom_rad+=1
                                        if cash_flet.bloom_rad>bloom_rad_sld.max:
                                            cash_flet.bloom_rad=bloom_rad_sld.max
                                        bloom_rad_sld.value=cash_flet.bloom_rad
                                        bloom_rad_sld.update()
                                        bloom_rad_txt.value=round(cash_flet.bloom_rad,2)
                                        bloom_rad_txt.update()


                                if cash_flet.selected == "bloom_spred_sld":
                                    cash_flet.bloom_spred+=0.1
                                    if cash_flet.bloom_spred > bloom_spred_sld.max: 
                                            cash_flet.bloom_spred=bloom_spred_sld.max
                                    bloom_spred_sld.value=cash_flet.bloom_spred
                                    bloom_spred_sld.update()
                                    bloom_spred_txt.value=round(cash_flet.bloom_spred,2)
                                    bloom_spred_txt.update()

                                if cash_flet.selected == "bloom_Halation_sld":
                                    cash_flet.bloom_Halation+=0.1
                                    if cash_flet.bloom_Halation > bloom_Halation_sld.max: 
                                            cash_flet.bloom_Halation=bloom_Halation_sld.max
                                    bloom_Halation_sld.value=cash_flet.bloom_Halation
                                    bloom_Halation_sld.update()
                                    bloom_Halation_txt.value=round(cash_flet.bloom_Halation,2)
                                    bloom_Halation_txt.update()

                                if cash_flet.selected == "sharp_rad_sld":
                                    cash_flet.sharp_rad+=1
                                    if cash_flet.sharp_rad > sharp_rad_sld.max: 
                                            cash_flet.sharp_rad=sharp_rad_sld.max
                                    sharp_rad_sld.value=cash_flet.sharp_rad
                                    sharp_rad_sld.update()
                                    sharp_rad_txt.value=round(cash_flet.sharp_rad,2)
                                    sharp_rad_txt.update()

                                if cash_flet.selected == "sharp_spred_sld":
                                    cash_flet.sharp_spred+=0.1
                                    if cash_flet.sharp_spred > sharp_spred_sld.max: 
                                            cash_flet.sharp_spred=sharp_spred_sld.max
                                    sharp_spred_sld.value=cash_flet.sharp_spred
                                    sharp_spred_sld.update()
                                    sharp_spred_txt.value=round(cash_flet.sharp_spred,2)
                                    sharp_spred_txt.update()

                                if cash_flet.selected == "sharp_amplif_sld":
                                    cash_flet.sharp_amplif+=1
                                    if cash_flet.sharp_amplif > sharp_amplif_sld.max: 
                                            cash_flet.sharp_amplif=sharp_amplif_sld.max
                                    sharp_amplif_sld.value=cash_flet.sharp_amplif
                                    sharp_amplif_sld.update()
                                    sharp_amplif_txt.value=round(cash_flet.sharp_amplif,2)
                                    sharp_amplif_txt.update()


                            #____-_____-_____-_____-_____-_____-_____
                            if e.key == "S": #  <<<<
                                if cash_flet.selected == "wbr_scan_sld":
                                        cash_flet.wbr_scan-=0.01
                                        if cash_flet.wbr_scan<wbr_scan_sld.min:
                                            cash_flet.wbr_scan=wbr_scan_sld.min
                                        wbr_scan_sld.value=cash_flet.wbr_scan
                                        wbr_scan_sld.update()
                                        wbr_scan_txt.value=round(cash_flet.wbr_scan,2)
                                        wbr_scan_txt.update()

                                if cash_flet.selected == "wbb_scan_sld":
                                        cash_flet.wbb_scan-=0.01
                                        if cash_flet.wbb_scan<wbb_scan_sld.min:
                                            cash_flet.wbb_scan=wbb_scan_sld.min
                                        wbb_scan_sld.value=cash_flet.wbb_scan
                                        wbb_scan_sld.update()
                                        wbb_scan_txt.value=round(cash_flet.wbb_scan,2)
                                        wbb_scan_txt.update()                                    

                                if cash_flet.selected == "wbr_sld":
                                        cash_flet.wbr-=0.1
                                        if cash_flet.wbr<wbr_sld.min:
                                            cash_flet.wbr=wbr_sld.min
                                        wbr_sld.value=cash_flet.wbr
                                        wbr_sld.update()
                                        wbr_txt.value=round(cash_flet.wbr,2)
                                        wbr_txt.update()   

                                if cash_flet.selected == "wbb_sld":
                                        cash_flet.wbb-=0.1
                                        if cash_flet.wbb<wbb_sld.min:
                                            cash_flet.wbb=wbb_sld.min
                                        wbb_sld.value=cash_flet.wbb
                                        wbb_sld.update()
                                        wbb_txt.value=round(cash_flet.wbb,2)
                                        wbb_txt.update()   


                                #___________________________________________
                                    
                                if cash_flet.selected == "gamma_sld":
                                        cash_flet.gamma-=0.1
                                        if cash_flet.gamma<gamma_sld.min:
                                            cash_flet.gamma=gamma_sld.min
                                        gamma_sld.value=cash_flet.gamma
                                        gamma_sld.update()
                                        gamma_txt.value=round(cash_flet.gamma,2)
                                        gamma_txt.update()   

                                if cash_flet.selected == "print_exp_sld":
                                        cash_flet.print_exp-=0.1
                                        if cash_flet.print_exp<print_exp_sld.min:
                                            cash_flet.print_exp=print_exp_sld.min
                                        print_exp_sld.value=cash_flet.print_exp
                                        print_exp_sld.update()
                                        print_exp_txt.value=round(cash_flet.print_exp,2)
                                        print_exp_txt.update()   

                                if cash_flet.selected == "print_cont_sld":
                                        cash_flet.print_cont-=0.02
                                        if cash_flet.print_cont<print_cont_sld.min:
                                            cash_flet.print_cont=print_cont_sld.min
                                        print_cont_sld.value=cash_flet.print_cont
                                        print_cont_sld.update()
                                        print_cont_txt.value=round(cash_flet.print_cont,2)
                                        print_cont_txt.update()  

                                if cash_flet.selected == "sut_sld":
                                        cash_flet.sut-=0.05
                                        if cash_flet.sut<sut_sld.min:
                                            cash_flet.sut=sut_sld.min
                                        sut_sld.value=cash_flet.sut
                                        sut_sld.update()
                                        sut_txt.value=round(cash_flet.sut,2)
                                        sut_txt.update()  
                                    

                                #_________________________________________
                                        
                                if cash_flet.selected == "white_point_sld":
                                        cash_flet.white_point-=0.01
                                        if cash_flet.white_point<white_point_sld.min:
                                            cash_flet.white_point=white_point_sld.min
                                        white_point_sld.value=cash_flet.white_point
                                        white_point_sld.update()
                                        white_point_txt.value=round(cash_flet.white_point,2)
                                        white_point_txt.update()  

                                if cash_flet.selected == "light_compr_sld":
                                        cash_flet.light_compr-=0.1
                                        if cash_flet.light_compr<light_compr_sld.min:
                                            cash_flet.light_compr=light_compr_sld.min
                                        light_compr_sld.value=cash_flet.light_compr
                                        light_compr_sld.update()
                                        light_compr_txt.value=round(cash_flet.light_compr,2)
                                        light_compr_txt.update()

                                if cash_flet.selected == "zone_sld":
                                        cash_flet.zone-=0.01
                                        if cash_flet.zone<zone_sld.min:
                                            cash_flet.zone=zone_sld.min
                                        zone_sld.value=cash_flet.zone
                                        zone_sld.update()
                                        zone_txt.value=round(cash_flet.zone,2)
                                        zone_txt.update()

                                    #_____________________________________________
                                    
                                if cash_flet.selected == "amplify_grain_sld":
                                        cash_flet.amplify_grain-=0.01
                                        if cash_flet.amplify_grain<amplify_grain_sld.min:
                                            cash_flet.amplify_grain=amplify_grain_sld.min
                                        amplify_grain_sld.value=cash_flet.amplify_grain
                                        amplify_grain_sld.update()
                                        amplify_grain_txt.value=round(cash_flet.amplify_grain,2)
                                        amplify_grain_txt.update()

                                if cash_flet.selected == "amplify_mask_sld":
                                        cash_flet.amplify_mask-=0.1
                                        if cash_flet.amplify_mask<amplify_mask_sld.min:
                                            cash_flet.amplify_mask=amplify_mask_sld.min
                                        amplify_mask_sld.value=cash_flet.amplify_mask
                                        amplify_mask_sld.update()
                                        amplify_mask_txt.value=round(cash_flet.amplify_mask,2)
                                        amplify_mask_txt.update()

                                if cash_flet.selected == "blur_rad_sld":
                                        cash_flet.blur_rad-=0.1
                                        if cash_flet.blur_rad<blur_rad_sld.min:
                                            cash_flet.blur_rad=blur_rad_sld.min
                                        blur_rad_sld.value=cash_flet.blur_rad
                                        blur_rad_sld.update()
                                        blur_rad_txt.value=round(cash_flet.blur_rad,2)
                                        blur_rad_txt.update()

                                if cash_flet.selected == "blur_spred_sld":
                                        cash_flet.blur_spred-=0.1
                                        if cash_flet.blur_spred<blur_spred_sld.min:
                                            cash_flet.blur_spred=blur_spred_sld.min
                                        blur_spred_sld.value=cash_flet.blur_spred
                                        blur_spred_sld.update()
                                        blur_spred_txt.value=round(cash_flet.blur_spred,2)
                                        blur_spred_txt.update()

                                if cash_flet.selected == "halation_sld":
                                        cash_flet.halation-=0.1
                                        if cash_flet.halation<halation_sld.min:
                                            cash_flet.halation=halation_sld.min
                                        halation_sld.value=cash_flet.halation
                                        halation_sld.update()
                                        halation_txt.value=round(cash_flet.halation,2)
                                        halation_txt.update()

                                if cash_flet.selected == "bloom_rad_sld":
                                        cash_flet.bloom_rad-=1
                                        if cash_flet.bloom_rad<bloom_rad_sld.min:
                                            cash_flet.bloom_rad=bloom_rad_sld.min
                                        bloom_rad_sld.value=cash_flet.bloom_rad
                                        bloom_rad_sld.update()
                                        bloom_rad_txt.value=round(cash_flet.bloom_rad,2)
                                        bloom_rad_txt.update()


                                if cash_flet.selected == "bloom_spred_sld":
                                    cash_flet.bloom_spred-=0.1
                                    if cash_flet.bloom_spred < bloom_spred_sld.min: 
                                            cash_flet.bloom_spred=bloom_spred_sld.min
                                    bloom_spred_sld.value=cash_flet.bloom_spred
                                    bloom_spred_sld.update()
                                    bloom_spred_txt.value=round(cash_flet.bloom_spred,2)
                                    bloom_spred_txt.update()

                                if cash_flet.selected == "bloom_Halation_sld":
                                    cash_flet.bloom_Halation-=0.1
                                    if cash_flet.bloom_Halation < bloom_Halation_sld.min: 
                                            cash_flet.bloom_Halation=bloom_Halation_sld.min
                                    bloom_Halation_sld.value=cash_flet.bloom_Halation
                                    bloom_Halation_sld.update()
                                    bloom_Halation_txt.value=round(cash_flet.bloom_Halation,2)
                                    bloom_Halation_txt.update()

                                if cash_flet.selected == "sharp_rad_sld":
                                    cash_flet.sharp_rad-=1
                                    if cash_flet.sharp_rad < sharp_rad_sld.min: 
                                            cash_flet.sharp_rad=sharp_rad_sld.min
                                    sharp_rad_sld.value=cash_flet.sharp_rad
                                    sharp_rad_sld.update()
                                    sharp_rad_txt.value=round(cash_flet.sharp_rad,2)
                                    sharp_rad_txt.update()

                                if cash_flet.selected == "sharp_spred_sld":
                                    cash_flet.sharp_spred-=0.1
                                    if cash_flet.sharp_spred < sharp_spred_sld.min: 
                                            cash_flet.sharp_spred=sharp_spred_sld.min
                                    sharp_spred_sld.value=cash_flet.sharp_spred
                                    sharp_spred_sld.update()
                                    sharp_spred_txt.value=round(cash_flet.sharp_spred,2)
                                    sharp_spred_txt.update()

                                if cash_flet.selected == "sharp_amplif_sld":
                                    cash_flet.sharp_amplif-=1
                                    if cash_flet.sharp_amplif < sharp_amplif_sld.min: 
                                            cash_flet.sharp_amplif=sharp_amplif_sld.min
                                    sharp_amplif_sld.value=cash_flet.sharp_amplif
                                    sharp_amplif_sld.update()
                                    sharp_amplif_txt.value=round(cash_flet.sharp_amplif,2)
                                    sharp_amplif_txt.update()


                            #__+_x2___#__+_x2___#__+_x2___#__+_x2___#__+_x2___#__+_x2___#__+_x2___
                            if e.key == "D": #  >>>////
                                if cash_flet.selected == "wbr_scan_sld":
                                        cash_flet.wbr_scan+=0.05
                                        if cash_flet.wbr_scan>wbr_scan_sld.max:
                                            cash_flet.wbr_scan=wbr_scan_sld.max
                                        wbr_scan_sld.value=cash_flet.wbr_scan
                                        wbr_scan_sld.update()
                                        wbr_scan_txt.value=round(cash_flet.wbr_scan,2)
                                        wbr_scan_txt.update()

                                if cash_flet.selected == "wbb_scan_sld":
                                        cash_flet.wbb_scan+=0.05
                                        if cash_flet.wbb_scan>wbb_scan_sld.max:
                                            cash_flet.wbb_scan=wbb_scan_sld.max
                                        wbb_scan_sld.value=cash_flet.wbb_scan
                                        wbb_scan_sld.update()
                                        wbb_scan_txt.value=round(cash_flet.wbb_scan,2)
                                        wbb_scan_txt.update()                                    

                                if cash_flet.selected == "wbr_sld":
                                        cash_flet.wbr+=1
                                        if cash_flet.wbr>wbr_sld.max:
                                            cash_flet.wbr=wbr_sld.max
                                        wbr_sld.value=cash_flet.wbr
                                        wbr_sld.update()
                                        wbr_txt.value=round(cash_flet.wbr,2)
                                        wbr_txt.update()   

                                if cash_flet.selected == "wbb_sld":
                                        cash_flet.wbb+=1
                                        if cash_flet.wbb>wbb_sld.max:
                                            cash_flet.wbb=wbb_sld.max
                                        wbb_sld.value=cash_flet.wbb
                                        wbb_sld.update()
                                        wbb_txt.value=round(cash_flet.wbb,2)
                                        wbb_txt.update()   


                                #___________________________________________
                                    
                                if cash_flet.selected == "gamma_sld":
                                        cash_flet.gamma+=1
                                        if cash_flet.gamma>gamma_sld.max:
                                            cash_flet.gamma=gamma_sld.max
                                        gamma_sld.value=cash_flet.gamma
                                        gamma_sld.update()
                                        gamma_txt.value=round(cash_flet.gamma,2)
                                        gamma_txt.update()   

                                if cash_flet.selected == "print_exp_sld":
                                        cash_flet.print_exp+=1
                                        if cash_flet.print_exp>print_exp_sld.max:
                                            cash_flet.print_exp=print_exp_sld.max
                                        print_exp_sld.value=cash_flet.print_exp
                                        print_exp_sld.update()
                                        print_exp_txt.value=round(cash_flet.print_exp,2)
                                        print_exp_txt.update()   

                                if cash_flet.selected == "print_cont_sld":
                                        cash_flet.print_cont+=0.1
                                        if cash_flet.print_cont>print_cont_sld.max:
                                            cash_flet.print_cont=print_cont_sld.max
                                        print_cont_sld.value=cash_flet.print_cont
                                        print_cont_sld.update()
                                        print_cont_txt.value=round(cash_flet.print_cont,2)
                                        print_cont_txt.update()  

                                if cash_flet.selected == "sut_sld":
                                        cash_flet.sut+=0.5
                                        if cash_flet.sut>sut_sld.max:
                                            cash_flet.sut=sut_sld.max
                                        sut_sld.value=cash_flet.sut
                                        sut_sld.update()
                                        sut_txt.value=round(cash_flet.sut,2)
                                        sut_txt.update()  
                                    

                                #_________________________________________
                                        
                                if cash_flet.selected == "white_point_sld":
                                        cash_flet.white_point+=0.1
                                        if cash_flet.white_point>white_point_sld.max:
                                            cash_flet.white_point=white_point_sld.max
                                        white_point_sld.value=cash_flet.white_point
                                        white_point_sld.update()
                                        white_point_txt.value=round(cash_flet.white_point,2)
                                        white_point_txt.update()  

                                if cash_flet.selected == "light_compr_sld":
                                        cash_flet.light_compr+=1
                                        if cash_flet.light_compr>light_compr_sld.max:
                                            cash_flet.light_compr=light_compr_sld.max
                                        light_compr_sld.value=cash_flet.light_compr
                                        light_compr_sld.update()
                                        light_compr_txt.value=round(cash_flet.light_compr,2)
                                        light_compr_txt.update()

                                if cash_flet.selected == "zone_sld":
                                        cash_flet.zone+=0.1
                                        if cash_flet.zone>zone_sld.max:
                                            cash_flet.zone=zone_sld.max
                                        zone_sld.value=cash_flet.zone
                                        zone_sld.update()
                                        zone_txt.value=round(cash_flet.zone,2)
                                        zone_txt.update()

                                    #_____________________________________________
                                    
                                if cash_flet.selected == "amplify_grain_sld":
                                        cash_flet.amplify_grain+=0.1
                                        if cash_flet.amplify_grain>amplify_grain_sld.max:
                                            cash_flet.amplify_grain=amplify_grain_sld.max
                                        amplify_grain_sld.value=cash_flet.amplify_grain
                                        amplify_grain_sld.update()
                                        amplify_grain_txt.value=round(cash_flet.amplify_grain,2)
                                        amplify_grain_txt.update()

                                if cash_flet.selected == "amplify_mask_sld":
                                        cash_flet.amplify_mask+=1
                                        if cash_flet.amplify_mask>amplify_mask_sld.max:
                                            cash_flet.amplify_mask=amplify_mask_sld.max
                                        amplify_mask_sld.value=cash_flet.amplify_mask
                                        amplify_mask_sld.update()
                                        amplify_mask_txt.value=round(cash_flet.amplify_mask,2)
                                        amplify_mask_txt.update()

                                if cash_flet.selected == "blur_rad_sld":
                                        cash_flet.blur_rad+=1
                                        if cash_flet.blur_rad>blur_rad_sld.max:
                                            cash_flet.blur_rad=blur_rad_sld.max
                                        blur_rad_sld.value=cash_flet.blur_rad
                                        blur_rad_sld.update()
                                        blur_rad_txt.value=round(cash_flet.blur_rad,2)
                                        blur_rad_txt.update()

                                if cash_flet.selected == "blur_spred_sld":
                                        cash_flet.blur_spred+=1
                                        if cash_flet.blur_spred>blur_spred_sld.max:
                                            cash_flet.blur_spred=blur_spred_sld.max
                                        blur_spred_sld.value=cash_flet.blur_spred
                                        blur_spred_sld.update()
                                        blur_spred_txt.value=round(cash_flet.blur_spred,2)
                                        blur_spred_txt.update()

                                if cash_flet.selected == "halation_sld":
                                        cash_flet.halation+=1
                                        if cash_flet.halation>halation_sld.max:
                                            cash_flet.halation=halation_sld.max
                                        halation_sld.value=cash_flet.halation
                                        halation_sld.update()
                                        halation_txt.value=round(cash_flet.halation,2)
                                        halation_txt.update()

                                if cash_flet.selected == "bloom_rad_sld":
                                        cash_flet.bloom_rad+=10
                                        if cash_flet.bloom_rad>bloom_rad_sld.max:
                                            cash_flet.bloom_rad=bloom_rad_sld.max
                                        bloom_rad_sld.value=cash_flet.bloom_rad
                                        bloom_rad_sld.update()
                                        bloom_rad_txt.value=round(cash_flet.bloom_rad,2)
                                        bloom_rad_txt.update()


                                if cash_flet.selected == "bloom_spred_sld":
                                    cash_flet.bloom_spred+=1
                                    if cash_flet.bloom_spred > bloom_spred_sld.max: 
                                            cash_flet.bloom_spred=bloom_spred_sld.max
                                    bloom_spred_sld.value=cash_flet.bloom_spred
                                    bloom_spred_sld.update()
                                    bloom_spred_txt.value=round(cash_flet.bloom_spred,2)
                                    bloom_spred_txt.update()

                                if cash_flet.selected == "bloom_Halation_sld":
                                    cash_flet.bloom_Halation+=1
                                    if cash_flet.bloom_Halation > bloom_Halation_sld.max: 
                                            cash_flet.bloom_Halation=bloom_Halation_sld.max
                                    bloom_Halation_sld.value=cash_flet.bloom_Halation
                                    bloom_Halation_sld.update()
                                    bloom_Halation_txt.value=round(cash_flet.bloom_Halation,2)
                                    bloom_Halation_txt.update()

                                if cash_flet.selected == "sharp_rad_sld":
                                    cash_flet.sharp_rad+=10
                                    if cash_flet.sharp_rad > sharp_rad_sld.max: 
                                            cash_flet.sharp_rad=sharp_rad_sld.max
                                    sharp_rad_sld.value=cash_flet.sharp_rad
                                    sharp_rad_sld.update()
                                    sharp_rad_txt.value=round(cash_flet.sharp_rad,2)
                                    sharp_rad_txt.update()

                                if cash_flet.selected == "sharp_spred_sld":
                                    cash_flet.sharp_spred+=1
                                    if cash_flet.sharp_spred > sharp_spred_sld.max: 
                                            cash_flet.sharp_spred=sharp_spred_sld.max
                                    sharp_spred_sld.value=cash_flet.sharp_spred
                                    sharp_spred_sld.update()
                                    sharp_spred_txt.value=round(cash_flet.sharp_spred,2)
                                    sharp_spred_txt.update()

                                if cash_flet.selected == "sharp_amplif_sld":
                                    cash_flet.sharp_amplif+=10
                                    if cash_flet.sharp_amplif > sharp_amplif_sld.max: 
                                            cash_flet.sharp_amplif=sharp_amplif_sld.max
                                    sharp_amplif_sld.value=cash_flet.sharp_amplif
                                    sharp_amplif_sld.update()
                                    sharp_amplif_txt.value=round(cash_flet.sharp_amplif,2)
                                    sharp_amplif_txt.update()


                            #__-_x2__#__-_x2__#__-_x2__#__-_x2__#__-_x2__#__-_x2__#__-_x2__
                            if e.key == "A": #  <<<<<mmmm
                                if cash_flet.selected == "wbr_scan_sld":
                                        cash_flet.wbr_scan-=0.05
                                        if cash_flet.wbr_scan<wbr_scan_sld.min:
                                            cash_flet.wbr_scan=wbr_scan_sld.min
                                        wbr_scan_sld.value=cash_flet.wbr_scan
                                        wbr_scan_sld.update()
                                        wbr_scan_txt.value=round(cash_flet.wbr_scan,2)
                                        wbr_scan_txt.update()

                                if cash_flet.selected == "wbb_scan_sld":
                                        cash_flet.wbb_scan-=0.05
                                        if cash_flet.wbb_scan<wbb_scan_sld.min:
                                            cash_flet.wbb_scan=wbb_scan_sld.min
                                        wbb_scan_sld.value=cash_flet.wbb_scan
                                        wbb_scan_sld.update()
                                        wbb_scan_txt.value=round(cash_flet.wbb_scan,2)
                                        wbb_scan_txt.update()                                    

                                if cash_flet.selected == "wbr_sld":
                                        cash_flet.wbr-=1
                                        if cash_flet.wbr<wbr_sld.min:
                                            cash_flet.wbr=wbr_sld.min
                                        wbr_sld.value=cash_flet.wbr
                                        wbr_sld.update()
                                        wbr_txt.value=round(cash_flet.wbr,2)
                                        wbr_txt.update()   

                                if cash_flet.selected == "wbb_sld":
                                        cash_flet.wbb-=1
                                        if cash_flet.wbb<wbb_sld.min:
                                            cash_flet.wbb=wbb_sld.min
                                        wbb_sld.value=cash_flet.wbb
                                        wbb_sld.update()
                                        wbb_txt.value=round(cash_flet.wbb,2)
                                        wbb_txt.update()   


                                #___________________________________________
                                    
                                if cash_flet.selected == "gamma_sld":
                                        cash_flet.gamma-=1
                                        if cash_flet.gamma<gamma_sld.min:
                                            cash_flet.gamma=gamma_sld.min
                                        gamma_sld.value=cash_flet.gamma
                                        gamma_sld.update()
                                        gamma_txt.value=round(cash_flet.gamma,2)
                                        gamma_txt.update()   

                                if cash_flet.selected == "print_exp_sld":
                                        cash_flet.print_exp-=1
                                        if cash_flet.print_exp<print_exp_sld.min:
                                            cash_flet.print_exp=print_exp_sld.min
                                        print_exp_sld.value=cash_flet.print_exp
                                        print_exp_sld.update()
                                        print_exp_txt.value=round(cash_flet.print_exp,2)
                                        print_exp_txt.update()   

                                if cash_flet.selected == "print_cont_sld":
                                        cash_flet.print_cont-=0.1
                                        if cash_flet.print_cont<print_cont_sld.min:
                                            cash_flet.print_cont=print_cont_sld.min
                                        print_cont_sld.value=cash_flet.print_cont
                                        print_cont_sld.update()
                                        print_cont_txt.value=round(cash_flet.print_cont,2)
                                        print_cont_txt.update()  

                                if cash_flet.selected == "sut_sld":
                                        cash_flet.sut-=0.5
                                        if cash_flet.sut<sut_sld.min:
                                            cash_flet.sut=sut_sld.min
                                        sut_sld.value=cash_flet.sut
                                        sut_sld.update()
                                        sut_txt.value=round(cash_flet.sut,2)
                                        sut_txt.update()  
                                    

                                #_________________________________________
                                        
                                if cash_flet.selected == "white_point_sld":
                                        cash_flet.white_point-=0.1
                                        if cash_flet.white_point<white_point_sld.min:
                                            cash_flet.white_point=white_point_sld.min
                                        white_point_sld.value=cash_flet.white_point
                                        white_point_sld.update()
                                        white_point_txt.value=round(cash_flet.white_point,2)
                                        white_point_txt.update()  

                                if cash_flet.selected == "light_compr_sld":
                                        cash_flet.light_compr-=1
                                        if cash_flet.light_compr<light_compr_sld.min:
                                            cash_flet.light_compr=light_compr_sld.min
                                        light_compr_sld.value=cash_flet.light_compr
                                        light_compr_sld.update()
                                        light_compr_txt.value=round(cash_flet.light_compr,2)
                                        light_compr_txt.update()

                                if cash_flet.selected == "zone_sld":
                                        cash_flet.zone-=0.1
                                        if cash_flet.zone<zone_sld.min:
                                            cash_flet.zone=zone_sld.min
                                        zone_sld.value=cash_flet.zone
                                        zone_sld.update()
                                        zone_txt.value=round(cash_flet.zone,2)
                                        zone_txt.update()

                                    #_____________________________________________
                                    
                                if cash_flet.selected == "amplify_grain_sld":
                                        cash_flet.amplify_grain-=0.1
                                        if cash_flet.amplify_grain<amplify_grain_sld.min:
                                            cash_flet.amplify_grain=amplify_grain_sld.min
                                        amplify_grain_sld.value=cash_flet.amplify_grain
                                        amplify_grain_sld.update()
                                        amplify_grain_txt.value=round(cash_flet.amplify_grain,2)
                                        amplify_grain_txt.update()

                                if cash_flet.selected == "amplify_mask_sld":
                                        cash_flet.amplify_mask-=1
                                        if cash_flet.amplify_mask<amplify_mask_sld.min:
                                            cash_flet.amplify_mask=amplify_mask_sld.min
                                        amplify_mask_sld.value=cash_flet.amplify_mask
                                        amplify_mask_sld.update()
                                        amplify_mask_txt.value=round(cash_flet.amplify_mask,2)
                                        amplify_mask_txt.update()

                                if cash_flet.selected == "blur_rad_sld":
                                        cash_flet.blur_rad-=1
                                        if cash_flet.blur_rad<blur_rad_sld.min:
                                            cash_flet.blur_rad=blur_rad_sld.min
                                        blur_rad_sld.value=cash_flet.blur_rad
                                        blur_rad_sld.update()
                                        blur_rad_txt.value=round(cash_flet.blur_rad,2)
                                        blur_rad_txt.update()

                                if cash_flet.selected == "blur_spred_sld":
                                        cash_flet.blur_spred-=1
                                        if cash_flet.blur_spred<blur_spred_sld.min:
                                            cash_flet.blur_spred=blur_spred_sld.min
                                        blur_spred_sld.value=cash_flet.blur_spred
                                        blur_spred_sld.update()
                                        blur_spred_txt.value=round(cash_flet.blur_spred,2)
                                        blur_spred_txt.update()

                                if cash_flet.selected == "halation_sld":
                                        cash_flet.halation-=1
                                        if cash_flet.halation<halation_sld.min:
                                            cash_flet.halation=halation_sld.min
                                        halation_sld.value=cash_flet.halation
                                        halation_sld.update()
                                        halation_txt.value=round(cash_flet.halation,2)
                                        halation_txt.update()

                                if cash_flet.selected == "bloom_rad_sld":
                                        cash_flet.bloom_rad-=10
                                        if cash_flet.bloom_rad<bloom_rad_sld.min:
                                            cash_flet.bloom_rad=bloom_rad_sld.min
                                        bloom_rad_sld.value=cash_flet.bloom_rad
                                        bloom_rad_sld.update()
                                        bloom_rad_txt.value=round(cash_flet.bloom_rad,2)
                                        bloom_rad_txt.update()


                                if cash_flet.selected == "bloom_spred_sld":
                                    cash_flet.bloom_spred-=1
                                    if cash_flet.bloom_spred < bloom_spred_sld.min: 
                                            cash_flet.bloom_spred=bloom_spred_sld.min
                                    bloom_spred_sld.value=cash_flet.bloom_spred
                                    bloom_spred_sld.update()
                                    bloom_spred_txt.value=round(cash_flet.bloom_spred,2)
                                    bloom_spred_txt.update()

                                if cash_flet.selected == "bloom_Halation_sld":
                                    cash_flet.bloom_Halation-=1
                                    if cash_flet.bloom_Halation < bloom_Halation_sld.min: 
                                            cash_flet.bloom_Halation=bloom_Halation_sld.min
                                    bloom_Halation_sld.value=cash_flet.bloom_Halation
                                    bloom_Halation_sld.update()
                                    bloom_Halation_txt.value=round(cash_flet.bloom_Halation,2)
                                    bloom_Halation_txt.update()

                                if cash_flet.selected == "sharp_rad_sld":
                                    cash_flet.sharp_rad-=10
                                    if cash_flet.sharp_rad < sharp_rad_sld.min: 
                                            cash_flet.sharp_rad=sharp_rad_sld.min
                                    sharp_rad_sld.value=cash_flet.sharp_rad
                                    sharp_rad_sld.update()
                                    sharp_rad_txt.value=round(cash_flet.sharp_rad,2)
                                    sharp_rad_txt.update()

                                if cash_flet.selected == "sharp_spred_sld":
                                    cash_flet.sharp_spred-=1
                                    if cash_flet.sharp_spred < sharp_spred_sld.min: 
                                            cash_flet.sharp_spred=sharp_spred_sld.min
                                    sharp_spred_sld.value=cash_flet.sharp_spred
                                    sharp_spred_sld.update()
                                    sharp_spred_txt.value=round(cash_flet.sharp_spred,2)
                                    sharp_spred_txt.update()

                                if cash_flet.selected == "sharp_amplif_sld":
                                    cash_flet.sharp_amplif-=10
                                    if cash_flet.sharp_amplif < sharp_amplif_sld.min: 
                                            cash_flet.sharp_amplif=sharp_amplif_sld.min
                                    sharp_amplif_sld.value=cash_flet.sharp_amplif
                                    sharp_amplif_sld.update()
                                    sharp_amplif_txt.value=round(cash_flet.sharp_amplif,2)
                                    sharp_amplif_txt.update()
                            if tabss.selected_index == 2:
                                    #and tabss.tabs[2].content.controls[0].controls[0].selected_index == 0
                                if e.key == "Z":
                                        tabss.tabs[2].content.controls[0].controls[0].selected_index = 0
                                        tabss.tabs[2].update()
                                
                                if e.key == "X":
                                        tabss.tabs[2].content.controls[0].controls[0].selected_index = 1
                                        tabss.tabs[2].update()

                                if e.key == "C":
                                        tabss.tabs[2].content.controls[0].controls[0].selected_index = 2
                                        tabss.tabs[2].update()
                                if cash_flet.blok_spase!=True:
                                    if e.key == " ":
                                        go_go(e)
                            
                            if e.key == "1":
                                    tabss.selected_index = 0
                                    tabss.update()
                            if e.key == "2":
                                    tabss.selected_index = 1
                                    tabss.update()
                            if e.key == "3":
                                    tabss.selected_index = 2
                                    tabss.update()
                            if e.key == "4":
                                    tabss.selected_index = 3
                                    tabss.update()


        page.on_keyboard_event = on_keyboard
        
        page.add(tabss)
        tabss.update()
        page.update()

    ft.app(target=main)






"""                                                ft.Column(controls=[
                                                            ft.Row(controls=[ft.Container(ft.Image(src="/Users/aleksejromadin/Desktop/defolt.png"))],expand=5),
                                                            ft.GridView(
                                                                            expand=1,
                                                                            runs_count=0,
                                                                            max_extent=150,
                                                                            child_aspect_ratio=1.0,
                                                                            spacing=0,
                                                                            run_spacing=0,
                                                                            controls=([(ft.Image(src="/Users/aleksejromadin/Desktop/Assambley/imgs/"+imgs[i][1]+".jpeg",fit=ft.ImageFit.SCALE_DOWN)) for i in imgs]),horizontal=True
                                                                         )
                                                            
                                                                ],expand=True,wrap=False)"""