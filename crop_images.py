from PIL import Image
from os import makedirs
from shutil import rmtree

#disable decompression bomb protection 
Image.MAX_IMAGE_PIXELS = None

#remove old data, and recreate the structure
rmtree("data")
makedirs("data/true")
makedirs("data/false")

#load and parse coords
with open("all.txt") as f:
    coords = [x.strip().split(';') for x in f.readlines()]

curr_img = 0
for x in coords:
    if int(x[0]) > curr_img:
        curr_img = int(x[0])
        print("Loading m%02d.tif..." % (curr_img,))
        img = Image.open("m%02d.tif" % (curr_img,))

    img_new = img.crop((int(x[1]), int(x[2]), int(x[1]) + 500, int(x[2]) + 500))
    
    if int(x[3]) == 0: #is this a neuron?
        img_new.save("data/false/m%02d_%d_%d.tif" % (curr_img, int(x[1]), int(x[2])))
    else:
        img_new.save("data/true/m%02d_%d_%d.tif" % (curr_img, int(x[1]), int(x[2])))