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

try:    
    with open('/media/external/Downloads/data/luna16.yml','r') as f:
        luna16 = yaml.load(f.read())
except:
    traceback.print_exc()
    luna16=[]
    
def get_xray(isplot=False,fignum=0,pngpath=None):
    
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
    
    if pngpath is not None:
        im = Image.fromarray(xray,mode="L")
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im.save(pngpath,"PNG")
        
    return xray, container


def get_grid(szs=(256,256,8*8),sms=(8,8)):
    szx,szy,szz=szs
    smx,smy=sms
    grid = {}#np.zeros((szx,szy)).astype(np.uint)

    # assign grid id.
    anchor_dict={}
    for x in range(szx):
        for y in range(szy):
            px = x//(szx/smx)
            py = y//(szy/smy)
            code = py + smx*px
            px = int(px)
            py = int(py)
            code = int(code)
            if code not in anchor_dict.keys():
                anchor_dict[code]=(x,y)
            grid[(x,y)]=(code,px,py)

    #plt.imshow(grid)
    #print(np.unique(grid))
    return grid, anchor_dict

def get_code(px,py,smx):
    code = py + smx*px
    return code

def make_xray_data(N=5,szs=(256,256,8*8),sms=(8,8)):
    
    szx,szy,szz = szs
    smx,smy = sms

    grid, anchor_dict = get_grid(szs=szs,sms=sms)
    X0 = np.zeros((N,szx,szy)).astype(np.uint8)
    Y0 = np.zeros((N,smx,smy,5)).astype(np.float)
    Y1 = np.zeros((N,szx,szy,smx*smy)).astype(np.uint8)

    c=0
    for n in range(N*100):
        try:
            # make random line
            terrain, container = get_xray()
            # rescale input image
            terrain = terrain.astype(np.float)
            terrain = (255*(terrain-np.min(terrain))/(np.max(terrain)-np.min(terrain))).astype(np.uint8)
            
            # for each line
            for row in container:
                # get mask and end points
                mask=row['mask'].astype(np.uint8)
                endpoints = np.array(row['endpoints']).astype(np.int)
                x0,y0=endpoints[0,:]
                x1,y1=endpoints[1,:]
                # ensure end points in line
                #assert(mask[x0,y0]==1 and mask[x1,y1]==1)

                istube=1
                # find min max point
                mask_indices = np.argwhere(mask>0)

                minx=np.min(mask_indices[:,0])
                miny=np.min(mask_indices[:,1])

                maxx=np.max(mask_indices[:,0])
                maxy=np.max(mask_indices[:,1])
                
                midx = (minx+maxx)/2.
                midy = (miny+maxy)/2.
                
                aindx=(midx).astype(np.int)
                aindy=(midy).astype(np.int)
                
                ind,indx,indy = grid[(aindx,aindy)]
                # get anchor in grid
                anchorx,anchory=anchor_dict[ind]
                
                # relative mid
                midx = midx-anchorx
                midy = midy-anchory
                
                # width
                widthx = maxx-minx
                widthy = maxy-miny
                
                # scale
                midx /= szx
                midy /= szy
                widthx /= szx
                widthy /= szy
                
                yolo = np.array([midx,midy,widthx,widthy,istube])
                #print(yolo)
                
                Y0[c,indx,indy,:]=yolo
                Y1[c,:,:,ind]=mask
            
            X0[c,:,:]=terrain
        
        except:
            #traceback.print_exc()
            
            if c >= N:
                break
            X0[c,...]=0
            Y0[c,...]=0
            Y1[c,...]=0
        
        if c>=N-1:
            break
        
        c+=1

    X0 = np.expand_dims(X0,axis=-1)
    return X0,Y0,Y1

