
import numpy as np
import pickle
import copy
from colour import read_LUT
from PIL import Image
import Not_multi_funk
import base64
from colour import io
from io import BytesIO
import os




film_count = 0
clear_shape = None
pil_img = None
is_half = False
cameras1 = pickle.load(open(os.path.abspath("Cum.pkl"),'rb'))
Films1 = pickle.load(open(os.path.abspath("Films.pkl"),'rb'))
inface1=pickle.load(open(os.path.abspath("inface.pkl"),'rb'))
Grain_curve=read_LUT(os.path.abspath('Grain_curve.spi1d'))



arr = io.convert_bit_depth(io.read_image(os.path.abspath("bg.jpeg"),method="Imageio"),"uint8")
pil_img = Image.fromarray(arr)
buff = BytesIO()
pil_img.save(buff, format="JPEG")
image_string = base64.b64encode(buff.getvalue()).decode("utf-8") 
grab_style_image = None
grab_style_params = None
wbr_scan = 1.0
wbb_scan = 1.0



gamma = 0.0
print_exp = 0.0
print_cont = 0.7
sut = 1.0
white_point = 1.0
light_compr = 0.0
zone = 0.00

sut_compr = 0.85
end_a_plus = 1.0
end_a_minus = 1.0
end_b_plus = 1.0
end_b_minus = 1.0
sut_a_plus = 1.0
sut_a_minus = 1.0
sut_b_plus = 1.0
sut_b_minus = 1.0

r_sut = 0.0
g_sut = 0.0
b_sut = 0.0
r_hue = 0.0
g_hue = 0.0
b_hue = 0.0
r_g = 0.0
r_b = 0.0
g_r = 0.0
g_b = 0.0
b_r = 0.0
b_g = 0.0
on_grain = False

wbr = 0.0
wbb = 0.0

amplify_grain = 0.2
amplify_mask = 7.0
blur_rad = 1.0
blur_spred = 9.0
halation = 1.7
bloom_rad = 140
bloom_spred = 22.0
bloom_Halation = 1.0
sharp_rad = 40.0
sharp_spred = 15.0
sharp_amplif = 0.0


width_shad = 0.25
steepness_shad = 6.0
width_light = 0.25
steepness_light = 6.0
neutral_mask = -1.0
amplyfy_wheel = 10.0
shad_a = 0.0
shad_b = 0.0
mid_a = 0.0
mid_b = 0.0
light_a = 0.0
light_b = 0.0

all = Not_multi_funk.plot_zone(width_shad,steepness_shad,width_light,steepness_light)

uploaded_file = None

pillow_for_grid_string = {}
camera = "Sigma FP"
def camera_c(e):
        global camera
        camera = e.control.value



film = "Kodak Ultramax 400 printed"
def film_c(e):
        global film
        film = e.control.value



Wb = "In camera WB"
def WB_c(e):
     global Wb
     Wb = e.control.value



resolution = 1600
save_resolution = 3000



imgs={}
count_bar = 0
length_bar = 1






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


aspect = "org"
rotate = 0.0

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





selected = None
blok_spase = False