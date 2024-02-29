import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import pickle
import copy
from colour import read_LUT
from PIL import Image
from multiprocessing import Queue,JoinableQueue
import os
close=False
q_in_zoom = JoinableQueue()
q_out_zoom = JoinableQueue()
blur_count = 0

q_in_blur = JoinableQueue()
q_out_blur = JoinableQueue()
zoom_count = 0

q_in_lut = JoinableQueue()
q_out_lut = JoinableQueue()
lut_count = 0

q_in_film = JoinableQueue()
q_out_film = JoinableQueue()
film_count = 0
clear_shape = None

donttoch = False
rerunconter = 0


cameras1 = pickle.load(open(os.path.abspath("Cum.pkl"),'rb'))
Films1 = pickle.load(open(os.path.abspath("Films.pkl"),'rb'))
inface1=pickle.load(open(os.path.abspath("inface.pkl"),'rb'))
Grain_curve=read_LUT(os.path.abspath('Grain_curve.spi1d'))
#Gr_sample=Image.open('grain_portra400.tif')
Gr_sample=Image.open(os.path.abspath('Grain_orwo500.tif'))
cameras=copy.deepcopy(cameras1)
Films=copy.deepcopy(Films1)
inface=copy.deepcopy(inface1)
Paper = np.load(os.path.abspath('Paper.npy')) 

WB_t=None
uploaded_file_temp=None
img_raw=None


img_crop=None
rotate_t=None
rotate_thin_t=None
crop_sl_t=None
y_shift_t=None
x_shift_t=None
acpect_t=None


img_zoom=None
resolution_t=None


img_corners=None
corners_t=True


prep_grain=None




img_reduse=None
blur_rad_t=None
halation_t=None
blur_spred_t=None

img_bloom=None
bloom_rad_t=None
bloom_Halation_t=None
bloom_spred_t=None


lut_mix=None
r_hue_t=None
r_sut_t=None
r_g_t=None
r_b_t=None
g_hue_t=None
g_sut_t=None
g_r_t=None
g_b_t=None
b_hue_t=None
b_sut_t=None
b_r_t=None
b_g_t=None


lut_film=None
accuracy_t=None
Film_t=None
camera_t=None



lut_prolab=None
sut_t=None
END_A_PLUS_t=None
END_A_MINUS_t=None
END_B_PLUS_t=None
END_B_MINUS_t=None
a_min_sut_t=None
a_plus_sut_t=None
b_min_sut_t=None
b_plus_sut_t=None
a_m_hue_t=None
a_p_hue_t=None
b_m_hue_t=None
b_p_hue_t=None
sut_compress_t=None



lut_wheel=None
steepness_shad_t=None
width_shad_t=None
steepness_light_t=None
width_light_t=None
shad_a_t=None
shad_b_t=None
mid_a_t=None
mid_b_t=None
light_a_t=None
light_b_t=None
neutral_mask_t=None
amply_wheel_t=None

lut_to_rgb=None
is_to_rgb=True

img_log=None
gamma_t=None

img_sharpen=None
sharp_rad_t=None
sharp_spred_t=None
sharp_amplif_t=None


count=0



r_scan = None
b_scan = None


names_zoom_process = None
names_blur_process = None
names_lut_process = None
names_film_process = None


gamma = None
print_exp = None
Wb_b = None
Wb_r = None
Wb_b2 = None
Wb_r2 = None
light_compr = None
zone = None
print_cont = None
amplify_grain = None
amplify_mask = None
on_grain = False
on_grain_t = None


img_film = None



name_t=None
conf_button=None
is_render=None
manage=False
save_conf=False
delete_conf=False