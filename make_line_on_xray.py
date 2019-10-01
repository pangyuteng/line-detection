import traceback
import numpy as np

import SimpleITK as sitk
from PIL import Image
from skimage.transform import rescale
import yaml

from make_line import make_one_sample
import matplotlib.pyplot as plt

def imread(fpath):
    reader= sitk.ImageFileReader()
    reader.SetFileName(fpath)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)    
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()    
    return arr,spacing,origin,direction

with open('/media/external/Downloads/data/luna16.yml','r') as f:
    luna16 = yaml.load(f.read())

def get_xray(isplot=False,fignum=0):
    
    filepath = np.random.choice(luna16)
    img,spacing,origin,direction = imread(filepath)
    xray = np.sum(img,axis=1)
    xray = (255.0 * (xray - xray.min())/(xray.max()-xray.min()))

    new_width = 256
    new_height = 256
    xray = rescale(xray,
                (spacing[2],spacing[0]),
                 anti_aliasing=True)
    xray = xray.astype(np.uint8)
    xray = Image.fromarray(xray,mode="L")
    xray = xray.resize((new_width, new_height), Image.ANTIALIAS)
    xray = np.array(xray)
    xray, container = make_one_sample(isplot=isplot,fignum=fignum,nx=256,ny=256,baseimage=xray)
    return xray, container

    #im = Image.fromarray(xray,mode="L")
    #im = im.resize((new_width, new_height), Image.ANTIALIAS)
    #im.save('testXray.png',"PNG")
