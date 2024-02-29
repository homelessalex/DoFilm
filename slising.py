
import numpy as np


from colour import LUT3D



#img = io.convert_bit_depth(io.read_image("/Users/aleksejromadin/Desktop/temper/untitled folder/SDIM3197.jpeg",'uint8'),'float64')

#tic = time.time()

def slise_for_blur(img):
    shape = img.shape
    r = img[...,0]
    #r1 = img[0:shape[0]:2,0:shape[1]:2,0]
    #r2 = img[0:shape[0]:2,1:shape[1]:2,0]
    #r3 = img[1:shape[0]:2,0:shape[1]:2,0]
    #r4 = img[1:shape[0]:2,1:shape[1]:2,0]
    g = img[...,1]
    #g1 = img[0:shape[0]:2,0:shape[1]:2,1]
    #g2 = img[0:shape[0]:2,1:shape[1]:2,1]
    #g3 = img[1:shape[0]:2,0:shape[1]:2,1]
    #g4 = img[1:shape[0]:2,1:shape[1]:2,1]
    b = img[...,2]
    #b1 = img[0:shape[0]:2,0:shape[1]:2,2]
    #b2 = img[0:shape[0]:2,1:shape[1]:2,2]
    #b3 = img[1:shape[0]:2,0:shape[1]:2,2]
    #b4 = img[1:shape[0]:2,1:shape[1]:2,2]

    slise_names = {}
    slise_names["r"]=r
    slise_names["g"]=g
    slise_names["b"]=b

    """slise_names["r1"]=r1
    slise_names["r2"]=r2
    slise_names["r3"]=r3
    slise_names["r4"]=r4

    slise_names["g1"]=g1
    slise_names["g2"]=g2
    slise_names["g3"]=g3
    slise_names["g4"]=g4

    slise_names["b1"]=b1
    slise_names["b2"]=b2
    slise_names["b3"]=b3
    slise_names["b4"]=b4"""

    slise_names_keys ={}

    for i in slise_names:
        slise_names_keys[i] = i

    for i in slise_names:
        slise_names[i] = {i:slise_names[i]}
    slised_for_blur = [slise_names,shape]
    return slised_for_blur

#slised_for_blur = slise_for_blur(img)
#slise_names = slised_for_blur[0]
#shape = slised_for_blur[1]


def assembling_after_blur(slise_names,shape):
    img_t = np.zeros((slise_names["r"].shape[0],slise_names["r"].shape[1],3))
    img_t[...,0] = slise_names["r"]
    img_t[...,1] = slise_names["g"]
    img_t[...,2] = slise_names["b"]    




    """img_t[0:shape[0]:2,0:shape[1]:2,0] = slise_names["r1"]
    img_t[0:shape[0]:2,1:shape[1]:2,0] = slise_names["r2"]
    img_t[1:shape[0]:2,0:shape[1]:2,0] = slise_names["r3"]
    img_t[1:shape[0]:2,1:shape[1]:2,0] = slise_names["r4"]

    img_t[0:shape[0]:2,0:shape[1]:2,1] = slise_names["g1"]
    img_t[0:shape[0]:2,1:shape[1]:2,1] = slise_names["g2"]
    img_t[1:shape[0]:2,0:shape[1]:2,1] = slise_names["g3"]
    img_t[1:shape[0]:2,1:shape[1]:2,1] = slise_names["g4"]

    img_t[0:shape[0]:2,0:shape[1]:2,2] = slise_names["b1"]
    img_t[0:shape[0]:2,1:shape[1]:2,2] = slise_names["b2"]
    img_t[1:shape[0]:2,0:shape[1]:2,2] = slise_names["b3"]
    img_t[1:shape[0]:2,1:shape[1]:2,2] = slise_names["b4"]"""
    return img_t

#img = assembling_after_blur(slise_names, shape)


def slise_for_zoom(img):
    shape = img.shape

    r1 = img[:int(shape[0]/2),:int(shape[1]/2),0]
    r2 = img[:int(shape[0]/2),int(shape[1]/2):,0]
    r3 = img[int(shape[0]/2):,:int(shape[1]/2),0]
    r4 = img[int(shape[0]/2):,int(shape[1]/2):,0]
    shape_chunk = r1.shape
    r11 = r1[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    r12 = r1[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    r13 = r1[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    r14 = r1[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    r21 = r2[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    r22 = r2[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    r23 = r2[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    r24 = r2[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    r31 = r3[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    r32 = r3[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    r33 = r3[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    r34 = r3[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    r41 = r4[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    r42 = r4[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    r43 = r4[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    r44 = r4[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]




    g1 = img[:int(shape[0]/2),:int(shape[1]/2),1]
    g2 = img[:int(shape[0]/2),int(shape[1]/2):,1]
    g3 = img[int(shape[0]/2):,:int(shape[1]/2),1]
    g4 = img[int(shape[0]/2):,int(shape[1]/2):,1]

    g11 = g1[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    g12 = g1[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    g13 = g1[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    g14 = g1[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    g21 = g2[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    g22 = g2[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    g23 = g2[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    g24 = g2[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    g31 = g3[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    g32 = g3[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    g33 = g3[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    g34 = g3[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    g41 = g4[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    g42 = g4[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    g43 = g4[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    g44 = g4[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    b1 = img[:int(shape[0]/2),:int(shape[1]/2),2]
    b2 = img[:int(shape[0]/2),int(shape[1]/2):,2]
    b3 = img[int(shape[0]/2):,:int(shape[1]/2),2]
    b4 = img[int(shape[0]/2):,int(shape[1]/2):,2]
    shape_chunk = b1.shape
    b11 = b1[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    b12 = b1[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    b13 = b1[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    b14 = b1[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    b21 = b2[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    b22 = b2[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    b23 = b2[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    b24 = b2[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    b31 = b3[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    b32 = b3[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    b33 = b3[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    b34 = b3[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    b41 = b4[:int(shape_chunk[0]/2),:int(shape_chunk[1]/2)]
    b42 = b4[:int(shape_chunk[0]/2),int(shape_chunk[1]/2):]
    b43 = b4[int(shape_chunk[0]/2):,:int(shape_chunk[1]/2)]
    b44 = b4[int(shape_chunk[0]/2):,int(shape_chunk[1]/2):]

    slise_names = {"r11": r11,
                   "r12": r12,
                   "r13": r13,
                   "r14": r14,
                   
                   "r21": r21,
                   "r22": r22,
                   "r23": r23,
                   "r24": r24,
                   
                   "r31": r31,
                   "r32": r32,
                   "r33": r33,
                   "r34": r34,
                   
                   "r41": r41,
                   "r42": r42,
                   "r43": r43,
                   "r44": r44,
                   
                   
                   
                   "g11": g11,
                   "g12": g12,
                   "g13": g13,
                   "g14": g14,
                   
                   "g21": g21,
                   "g22": g22,
                   "g23": g23,
                   "g24": g24,
                   
                   "g31": g31,
                   "g32": g32,
                   "g33": g33,
                   "g34": g34,
                   
                   "g41": g41,
                   "g42": g42,
                   "g43": g43,
                   "g44": g44,
                   
                   
                   
                   "b11": b11,
                   "b12": b12,
                   "b13": b13,
                   "b14": b14,
                   
                   "b21": b21,
                   "b22": b22,
                   "b23": b23,
                   "b24": b24,
                   
                   "b31": b31,
                   "b32": b32,
                   "b33": b33,
                   "b34": b34,
                   
                   "b41": b41,
                   "b42": b42,
                   "b43": b43,
                   "b44": b44,}
    
    slise_names_keys = {}

    for i in slise_names:
        slise_names_keys[i] = i
    for i in slise_names:
        slise_names[i] = {i:slise_names[i]}
    separeted_for_zoom = [slise_names,slise_names_keys,shape,shape_chunk]
    return separeted_for_zoom



#separetes_for_zoom = slise_for_zoom(img)

#slise_names = separetes_for_zoom[0]
#slise_names_keys = separetes_for_zoom[1]
#shape = separetes_for_zoom[2]


def assembling_after_zoom(slise_names):

    img =np.dstack([ 
                    np.vstack([
                                np.hstack([slise_names["r11"],slise_names["r12"],slise_names["r21"],slise_names["r22"]]),
                                np.hstack([slise_names["r13"],slise_names["r14"],slise_names["r23"],slise_names["r24"]]),
                                np.hstack([slise_names["r31"],slise_names["r32"],slise_names["r41"],slise_names["r42"]]),
                                np.hstack([slise_names["r33"],slise_names["r34"],slise_names["r43"],slise_names["r44"]])
                                ]),
                    np.vstack([
                                np.hstack([slise_names["g11"],slise_names["g12"],slise_names["g21"],slise_names["g22"]]),
                                np.hstack([slise_names["g13"],slise_names["g14"],slise_names["g23"],slise_names["g24"]]),
                                np.hstack([slise_names["g31"],slise_names["g32"],slise_names["g41"],slise_names["g42"]]),
                                np.hstack([slise_names["g33"],slise_names["g34"],slise_names["g43"],slise_names["g44"]])
                                ]),
                    np.vstack([
                                np.hstack([slise_names["b11"],slise_names["b12"],slise_names["b21"],slise_names["b22"]]),
                                np.hstack([slise_names["b13"],slise_names["b14"],slise_names["b23"],slise_names["b24"]]),
                                np.hstack([slise_names["b31"],slise_names["b32"],slise_names["b41"],slise_names["b42"]]),
                                np.hstack([slise_names["b33"],slise_names["b34"],slise_names["b43"],slise_names["b44"]])
                                ])
                        ]) 
    return img

#img = assembling_after_zoom (slise_names)



def lut_slise():

    hald = LUT3D.linear_table(64)
    hald = np.array(hald, dtype=np.float32)
    lut_slises = {}
    lut_slises_keys = {}
    key_gener = ["lut"+str(i) for i in range(64)]
    lut_cnt = 0
    
    for i in key_gener:

        lut_slises[i] = {i:hald[lut_cnt,...]}
        lut_cnt+=1
    lut_cnt = 0
    
    for i in lut_slises:
        lut_slises_keys[i] = list(lut_slises.keys())[lut_cnt]
        lut_cnt+=1

    return lut_slises,lut_slises_keys

#slised_lut = lut_slise()
#lut_slises = slised_lut[0]
#lut_slises_keys = slised_lut[1]

def lut_assambley(lut_slises,lut_slises_keys):
    lut = np.zeros((64,64,64,3))
    lut_cnt = 0
    for i in lut_slises_keys:
        lut[lut_cnt,...] = lut_slises[i]
        lut_cnt+= 1
    return lut

#lut = lut_assambley(lut_slises,lut_slises_keys)

def slice_for_filmEm(img):
    threads = 8
    start,stop =0,0
    shape = img.shape
    coef=int(img.shape[0]/threads)
    imgs = ["img"+str(i) for i in range(threads)]
    slices_keys = {}
    for i in imgs:
            slices_keys[i] = i
    
    slices = {}

    for i in slices_keys:
        stop+=coef 
        slices[i]={i:img[start:stop,:,:]}
        start+=coef   
    slices["add"] = {"add":img[stop:shape[0],...]}
    return slices,slices_keys,shape,threads

#sliced_for_filmEm = slice_for_filmEm(img)
#slices_filmEm = sliced_for_filmEm[0]
#slices_filmEm_keys = sliced_for_filmEm[1]
#shape = sliced_for_filmEm[2]

def assembling_filmEm(slices_filmEm,slices_filmEm_keys,shape,threads):

    coef=int(shape[0]/threads)
    start,stop = 0,0
    img = np.zeros(shape,dtype=np.float32)

    for i in slices_filmEm_keys:
        stop+=coef 
        img[start:stop,:,:]=slices_filmEm[i]
        start+=coef   
    img[stop:shape[0],...] = slices_filmEm["add"]

    return img





#toc = time.time()
#print(toc-tic, "sec")
#print(img.shape)
#img = assembling_filmEm(slices_filmEm,slices_filmEm_keys,shape)

#pill = Image.fromarray(io.convert_bit_depth(img,"uint8")).show()

#print(lut.shape)
#print(shape, " :before ", img.shape)