import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from scipy import signal
from colour import gamma_function











def reduse_sharpness (q_in_blur,q_out_blur):

     while True:
        get_par = q_in_blur.get()
        input = get_par[0]
        

        Blur_rad = get_par[1]
        Halation = get_par[2]
        Blur_spred = get_par[3]
        shape = get_par[4]
        bloom_rad = get_par[5]
        bloom_Halation = get_par[6]
        bloom_spred = get_par[7]
        sharp_rad = get_par[8]
        sharp_spread = get_par[9]
        sharp_amplify = get_par[10]
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
        blur_coef=float(np.max(shape)/1500)
        def sharp(img,Sharp_rad, Sharp_spread,Sharp_amplify):
            
            img = gamma_function(img,0.4) #gamma 2.5
            img = np.nan_to_num(img)
            blured = signal.oaconvolve(img,blur(int(3*Sharp_rad*blur_coef),(2**Sharp_spread)*(blur_coef**2)),'same',None)
            sharp=(img-blured)*(Sharp_amplify/100)
            img+=sharp
            
            img[img>=1]=1
            img[img<=0]=0
            img = gamma_function(img,2.5)
            
            
            return img

        for i in input:
            
            if "r" in i:
                input[i] = signal.oaconvolve(input[i],blur(int(3*Blur_rad*Halation*blur_coef),(2**Blur_spred)*(blur_coef**2)),'same',None)
                input[i] = signal.oaconvolve(input[i] ,blur(int(3*bloom_rad*bloom_Halation*blur_coef),(2**bloom_spred)*(blur_coef**2)),'same',None)
                input[i] = sharp(input[i],sharp_rad,sharp_spread,sharp_amplify)

            if "g" in i:
                input[i]= signal.oaconvolve(input[i],blur(int(3*Blur_rad*blur_coef),(2**Blur_spred)*(blur_coef**2)),'same',None)
                input[i]= signal.oaconvolve(input[i],blur(int(3*bloom_rad*blur_coef),(2**bloom_spred)*(blur_coef**2)),'same',None)
                input[i] = sharp(input[i],sharp_rad,sharp_spread,sharp_amplify)
            if "b" in i:

                input[i]= signal.oaconvolve(input[i],blur(int(np.around((1/Halation)*3*Blur_rad*blur_coef)),(2**Blur_spred)*(blur_coef**2)),'same',None)
                input[i]= signal.oaconvolve(input[i],blur(int(np.around((1/bloom_Halation)*3*bloom_rad*blur_coef)),(2**bloom_spred)*(blur_coef**2)),'same',None) 
                input[i] = sharp(input[i],sharp_rad,sharp_spread,sharp_amplify)       

                
       
        
        

        
        
        

        q_out_blur.put(input)
        q_in_blur.task_done()





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