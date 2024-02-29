import numpy as np
from PIL import Image
from matplotlib.backend_bases import MouseButton
from colour import io
import matplotlib.pyplot as plt
import matplotlib
import rawpy
import cash_flet
#from flet_app import main
import time

def add_corners(img):
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



def pre_grain(image,gr_sample):
    image=image[200:int(image.shape[0]-200),200:int(image.shape[1]-200),:]
    if np.shape(image)[0]>np.shape(image)[1]:
        gr_sample=gr_sample.transpose(method=Image.Transpose.ROTATE_90)

    Grain_crop=gr_sample.resize((int(np.shape(image)[1]),int(np.shape(image)[0])),resample=Image.LANCZOS, )
    grain=np.array(Grain_crop,dtype=np.uint8)
    grain=io.convert_bit_depth(grain,"float64")
    grain_0=(grain-np.average(grain))
    return grain_0


def preper_lut(
            wrong,
            right,

            ):
    degr=9
    lut_size=64
    wrong-=np.min(wrong)
    wrong*=(1/np.max(wrong))
    right-=np.min(right)
    right*=(1/np.max(right))
    def ntrls(sourse_tif):
            ntrl_r=np.zeros((1,6,3))


            for i in range(1,20):
                z=(i*4)-1
                y=sourse_tif[z,:,:]
                y=y.reshape((1,6,3))
                ntrl_r=np.vstack((ntrl_r,y))
            ntrl_r=ntrl_r[1:20,:,:]
            ntrl_r=np.sort(ntrl_r)


            return ntrl_r


    wrong_0=np.array(wrong)
    right_0=np.array(right)
    wrong=ntrls(wrong)
    right=ntrls(right)
    wrong=np.nan_to_num(wrong)
    right=np.nan_to_num(right)


    #a2=abs(avavrge(wrong,right))
    lut_curve_in=np.stack((np.linspace(0,1,lut_size),np.linspace(0,1,lut_size),np.linspace(0,1,lut_size)),-1)
    lut_curve_out=np.array(lut_curve_in)
    
    def rgb_curve(lut,wrong,right,degr):
            def avavrge (wro,righ):
                    delt=np.fabs((righ-wro))
                    delt=np.nan_to_num(delt)
                    delt=np.average(delt)
                    return delt
            count=0
            wrong2=np.array(wrong)
            x=0.1
            lut2=np.array(lut)
            for i in range(20):
                    count+=1
                    if count==4:
                            a3=avavrge(wrong,right)
                            count=0
                            x=x/3
                    for a in range(lut_size):
                            for d in range(3):
                                    poli=np.polynomial.polynomial.Polynomial.fit(lut2[:,d],lut[:,d],deg=degr)
                                    wrong[:,:,d]=poli(wrong2[:,:,d])
                                    a2=avavrge(wrong[:,:,:],right[:,:,:])
                                    lut[a,d]+=x
                                    poli=np.polynomial.polynomial.Polynomial.fit(lut2[:,d],lut[:,d],deg=degr)
                                    wrong[:,:,d]=poli(wrong2[:,:,d])
                                    a2r=avavrge(wrong[:,:,:],right[:,:,:])
                                    if a2r>=a2:
                                            lut[a,d]-=(x)
                                            lut[a,d]-=(x)

                                            poli=np.polynomial.polynomial.Polynomial.fit(lut2[:,d],lut[:,d],deg=degr)
                                            wrong[:,:,d]=poli(wrong2[:,:,d])
                                            a2r=avavrge(wrong[:,:,:],right[:,:,:])
                                            if a2r>=a2:
                                                    lut[a,d]+=(x)



            return lut_curve_out


    lut_curve_out=rgb_curve(lut_curve_out,wrong,right,degr)

    wrong=np.array(wrong_0)

    right=np.array(right_0)
    poli=np.polynomial.polynomial.Polynomial.fit(lut_curve_in[:,1],lut_curve_out[:,1],deg=degr)

    for d in range(3):
            wrong[:,:,d]=poli(wrong[:,:,d])

    #wromg2 = Image.fromarray(io.convert_bit_depth(wrong,"uint8")).show()

    wrong=wrong.reshape((456,3))
    right=right.reshape((456,3))



    
    def colcolo(wrong,right):
            wrong_0=np.array(wrong)
            right_0=np.array(right)
            suttt=1
            coeff=0.1
            
            a=1000
            a1=0
            a2=0
            a3=0


            g_r=0
            g_g=suttt
            g_b=0

            r_r=suttt
            r_g=0
            r_b=0

            b_r=0
            b_g=0
            b_b=suttt



            wrong=wrong[264:288,:]
            right=right[264:288,:]

            wrong2=np.array(wrong)
            wrong3=np.array(wrong)

            #pill=Image.fromarray(io.convert_bit_depth(np.reshape(right,(4,6,3)),"uint8"))
            #pill.show()

            def bw(samp):
                    blwa=1-((samp[:,0]+samp[:,1]+samp[:,2])/3)
                    #blwa=np.stack((blwa,blwa,blwa),-1)
                    return blwa

            #wrong2 = Image.fromarray(io.convert_bit_depth(wrong.reshape(6,4,3),"uint8")).show()
            #wrong2 = Image.fromarray(io.convert_bit_depth(right.reshape(6,4,3),"uint8")).show()
            def average (wro,righ):
                    delt=np.sum(np.fabs(righ-wro))
                    return delt
            count=0
            for i in range(1):
                    a=abs(average(wrong,right))
                    count+=1
                    if count==100:
                            count=0
                            coeff*=0.95
                    wrong=np.array(wrong2)
                    wrong[:,0]=wrong3[:,0]*r_r+wrong3[:,1]*r_g+wrong3[:,2]*r_b
                    a1=abs(average(wrong[:,0],right[:,0]))
                    r_r+=coeff
                    wrong[:,0]=wrong3[:,0]*(r_r)+wrong3[:,1]*r_g+wrong3[:,2]*r_b
                    a1r=abs(average(wrong[:,0],right[:,0]))
                    if a1r>a1:
                            r_r-=(2*coeff)
                            wrong[:,0]=wrong3[:,0]*(r_r)+wrong3[:,1]*r_g+wrong3[:,2]*r_b
                            a1r=abs(average(wrong[:,0],right[:,0]))
                            if a1r>a1:
                                    r_r+=coeff

                    wrong[:,0]=wrong3[:,0]*r_r+wrong3[:,1]*r_g+wrong3[:,2]*r_b
                    a1=abs(average(wrong[:,0],right[:,0]))
                    r_g+=coeff
                    wrong[:,0]=wrong3[:,0]*r_r+wrong3[:,1]*(r_g)+wrong3[:,2]*r_b
                    a1g=abs(average(wrong[:,0],right[:,0]))
                    if a1g>a1:
                            r_g-=(2*coeff)
                            wrong[:,0]=wrong3[:,0]*r_r+wrong3[:,1]*(r_g)+wrong3[:,2]*r_b
                            a1g=abs(average(wrong[:,0],right[:,0]))
                            if a1g>a1:
                                    r_g+=coeff

                    wrong[:,0]=wrong3[:,0]*r_r+wrong3[:,1]*r_g+wrong3[:,2]*r_b
                    a1=abs(average(wrong[:,0],right[:,0]))
                    r_b+=coeff
                    wrong[:,0]=wrong3[:,0]*r_r+wrong3[:,1]*r_g+wrong3[:,2]*(r_b)
                    a1b=abs(average(wrong[:,0],right[:,0]))
                    if a1b>a1:
                            r_b-=(2*coeff)
                            wrong[:,0]=wrong3[:,0]*r_r+wrong3[:,1]*r_g+wrong3[:,2]*(r_b)
                            a1b=abs(average(wrong[:,0],right[:,0]))
                            if a1b>a1:
                                    r_b+=coeff




                    ################################
                    #
                    #
                    #################################

                    wrong[:,1]=wrong3[:,0]*g_r+wrong3[:,1]*g_g+wrong3[:,2]*g_b
                    a2=abs(average(wrong[:,1],right[:,1]))
                    g_r+=coeff
                    wrong[:,1]=wrong3[:,0]*(g_r)+wrong3[:,1]*g_g+wrong3[:,2]*g_b
                    a2r=abs(average(wrong[:,1],right[:,1]))
                    if a2r>a2:
                            g_r-=(2*coeff)
                            wrong[:,1]=wrong3[:,0]*(g_r)+wrong3[:,1]*g_g+wrong3[:,2]*g_b
                            a2r=abs(average(wrong[:,1],right[:,1]))
                            if a2r>a2:
                                    g_r+=coeff

                    wrong[:,1]=wrong3[:,0]*g_r+wrong3[:,1]*g_g+wrong3[:,2]*g_b
                    a2=abs(average(wrong[:,1],right[:,1]))
                    g_g+=coeff
                    wrong[:,1]=wrong3[:,0]*g_r+wrong3[:,1]*(g_g)+wrong3[:,2]*g_b
                    a2g=abs(average(wrong[:,1],right[:,1]))
                    if a2g>a2:
                            g_g-=(2*coeff)
                            wrong[:,1]=wrong3[:,0]*g_r+wrong3[:,1]*(g_g)+wrong3[:,2]*g_b
                            a2g=abs(average(wrong[:,1],right[:,1]))
                            if a2g>a2:
                                    g_g+=coeff

                    wrong[:,1]=wrong3[:,0]*g_r+wrong3[:,1]*g_g+wrong3[:,2]*g_b
                    a2=abs(average(wrong[:,1],right[:,1]))
                    g_b+=coeff
                    wrong[:,1]=wrong3[:,0]*g_r+wrong3[:,1]*g_g+wrong3[:,2]*(g_b)
                    a2b=abs(average(wrong[:,1],right[:,1]))
                    if a2b>a2:
                            g_b-=(2*coeff)
                            wrong[:,1]=wrong3[:,0]*g_r+wrong3[:,1]*g_g+wrong3[:,2]*(g_b)
                            a2b=abs(average(wrong[:,1],right[:,1]))
                            if a2b>a2:
                                    g_b+=coeff



                    ################################
                    #
                    #
                    #################################

                    wrong[:,2]=wrong3[:,0]*b_r+wrong3[:,1]*b_g+wrong3[:,2]*b_b
                    a3=abs(average(wrong[:,2],right[:,2]))
                    b_r+=coeff
                    wrong[:,2]=wrong3[:,0]*(b_r)+wrong3[:,1]*b_g+wrong3[:,2]*b_b
                    a3r=abs(average(wrong[:,2],right[:,2]))
                    if a3r>a3:
                            b_r-=(2*coeff)
                            wrong[:,2]=wrong3[:,0]*(b_r)+wrong3[:,1]*b_g+wrong3[:,2]*b_b
                            a3r=abs(average(wrong[:,2],right[:,2]))
                            if a3r>a3:
                                    b_r+=coeff

                    wrong[:,2]=wrong3[:,0]*b_r+wrong3[:,1]*b_g+wrong3[:,2]*b_b
                    a3=abs(average(wrong[:,2],right[:,2]))
                    b_g+=(coeff)
                    wrong[:,2]=wrong3[:,0]*b_r+wrong3[:,1]*(b_g)+wrong3[:,2]*b_b
                    a3g=abs(average(wrong[:,2],right[:,2]))
                    if a3g>a3:
                            b_g-=(2*coeff)
                            wrong[:,2]=wrong3[:,0]*b_r+wrong3[:,1]*(b_g)+wrong3[:,2]*b_b
                            a3g=abs(average(wrong[:,2],right[:,2]))
                            if a3g>a3:
                                b_g+=(coeff)

                    wrong[:,2]=wrong3[:,0]*b_r+wrong3[:,1]*b_g+wrong3[:,2]*b_b
                    a3=abs(average(wrong[:,2],right[:,2]))
                    b_b+=coeff
                    wrong[:,2]=wrong3[:,0]*b_r+wrong3[:,1]*b_g+wrong3[:,2]*(b_b)
                    a3b=abs(average(wrong[:,2],right[:,2]))
                    if a3b>a3:
                            b_b-=(2*coeff)
                            wrong[:,2]=wrong3[:,0]*b_r+wrong3[:,1]*b_g+wrong3[:,2]*(b_b)
                            a3b=abs(average(wrong[:,2],right[:,2]))
                            if a3b>a3:
                                    b_b+=coeff



                    

                    #print(a," its a")
            
            right=right_0
            wrong=wrong_0
            """wrong[:,0]=wrong_0[:,0]*r_r+wrong_0[:,1]*r_g+wrong_0[:,2]*r_b
            wrong[:,1]=wrong_0[:,0]*g_r+wrong_0[:,1]*g_g+wrong_0[:,2]*g_b
            wrong[:,2]=wrong_0[:,0]*b_r+wrong_0[:,1]*b_g+wrong_0[:,2]*b_b"""

            #wrong2 = Image.fromarray(io.convert_bit_depth(wrong.reshape(6,64,3),"uint8")).show()
            coefs = {"r_r": r_r, "r_g": r_g, "r_b": r_b,
                     "g_r": g_r, "g_g": g_g, "g_b": g_b,
                     "b_r": b_r, "b_g": b_g, "b_b": b_b,
                     "lut_curve_out": lut_curve_out,"lut_curve_in": lut_curve_in,
                     "degr": degr
                    }

            return wrong,coefs    
    
    tempp=colcolo(wrong,right)

    return tempp


def plot_zone(
            width_shad,
            steepness_shad,

            width_light,
            steepness_light,):
        def curve(img,steepness,width,apex):
                img=(2/(1+np.power(10**steepness,(-width-apex+img))))*(2/(1+np.power(10**steepness,(-width+apex-img))))
                return img
        matplotlib.use("svg")

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
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 2)
        ax.plot(plot_shad, shad, linewidth=3.0,color='black')
        ax.plot(plot_light,light,linewidth=3.0,color='white')
        ax.plot(plot_light,mid,linewidth=3.0,color='gray')
        ax.set_facecolor("silver")
        ax.set_axis_off()
        plt.tight_layout()



                
        return plt












def read_raw(all):
        imgs = None
        try:
  

                with rawpy.imread(all) as raw:
                        rgb = raw.postprocess(output_color=rawpy.ColorSpace(1), half_size=True,
                                        use_camera_wb=True, highlight_mode=rawpy.HighlightMode(0),demosaic_algorithm=rawpy.DemosaicAlgorithm(0),exp_preserve_highlights=(1.0), exp_shift=(4.0),
                                        output_bps=8,  no_auto_scale=False, no_auto_bright=False,auto_bright_thr=0.000000001,fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(0),
                                        gamma=(2.222, 4.5))
                        
                        rgb=np.dstack((rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]))
                        coef=1920/np.max(rgb.shape)
                        rgb=Image.fromarray(rgb)
                        rgb=rgb.resize((int(rgb.width*coef),int(rgb.height*coef)),resample=Image.Resampling.NEAREST)
                        name = str(all)
                        name = name[-12:-4]
                        rgb.save("imgs/"+name+".jpeg", format="JPEG", quality=75)
                        imgs=[all,name,False,False,{},False]
                        cash_flet.count_bar+=1
        
                        """prog_bar.value = ((cash_flet.count_bar/cash_flet.length_bar))

                        prog_bar.update()"""
                        

        except: # Replace Exception with something more specific.
                        cash_flet.count_bar+=1 


        #main.prog_bar_update()
        return imgs                
        

