import traceback
import numpy as np
from scipy.ndimage import binary_dilation
from scipy import ndimage as ndi

from extract_centerline import extract_centerline
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt

def smooth_line(x,y,num=None,**kwargs):
    if num is None:
        num = len(x)
    w = np.arange(0,len(x),1)
    sx = UnivariateSpline(w,x,**kwargs)
    sy = UnivariateSpline(w,y,**kwargs)
    wnew = np.linspace(0,len(x),num)
    return sx(wnew),sy(wnew)

def make_line(nx=256,ny=256,minlen=50):
    sz = (nx,ny)
    canvas = np.zeros(sz)

    peak_num = np.random.randint(0,10,1)[0]
    for n in range(peak_num):
        x = np.random.choice(np.linspace(0, nx-1, nx),1).astype(np.int)[0]
        y = np.random.choice(np.linspace(0, ny-1, ny),1).astype(np.int)[0]
        canvas[x,y]=1

    distance = ndi.distance_transform_edt(np.logical_not(canvas))
    terrain = ndi.morphology.distance_transform_edt(distance)
    terrain = -1*terrain + -1*np.min(-1*terrain)


    prct = np.random.randint(25,75,1)[0]
    th = np.percentile(terrain.ravel(),prct)
    mask = terrain<th
    mask = np.expand_dims(mask,axis=-1)
    
    count0=0
    while True:
        pt = []
        for n in range(2):
            count1=0
            breakok = False
            while True:
                x = np.random.choice(np.linspace(0, nx-1, nx),1).astype(np.int)[0]
                y = np.random.choice(np.linspace(0, ny-1, ny),1).astype(np.int)[0]
                z = 0
                # determine first point
                if len(pt)==0 and mask[x,y,z] == 1:
                    breakok = True
                # determine second point
                if len(pt)==1 and mask[x,y,z] == 1 and \
                    np.abs(x-pt[0][0]) > minlen and np.abs(y-pt[0][1]) > minlen:
                    breakok = True
                if breakok:
                    break
                count1+=1
                if count1>100:
                    raise LookupError('ok1')
            pt.append((x,y,z))
        start_point = pt[0]
        end_point = pt[1]
        line = extract_centerline(mask,start_point,end_point)
        linexorg,lineyorg,linez=line
        smothing_factor=np.random.rand()*1000
        try:
            linex,liney = smooth_line(linexorg,lineyorg,num=100*len(linexorg),s=smothing_factor)
        except:
            pass
        if len(linexorg) > 5:
            break
            
        count0+=1
        if count0>200:
            raise LookupError('ok0')
            
    mask = mask.squeeze()
    return mask, terrain, linexorg,lineyorg, linex, liney


# create dataset with multiple lines

def make_one_sample(isplot=False,fignum=0,nx=256,ny=256,baseimage=None,minlen=50):
    if baseimage is None:
        maskorg, orgimage, linexorg,lineyorg, linex, liney = make_line(nx=nx,ny=ny,minlen=minlen)
        myimage = np.copy(orgimage)
    else:
        orgimage = np.copy(baseimage)
        myimage = np.copy(baseimage)
        
    mymax = np.max(myimage)
    linmax = np.random.randint(1,5,1)[0]
    container = []
    for x in range(linmax):
        prct = np.random.randint(75,100,1)[0]
        mymax = np.percentile(orgimage.ravel(),prct)
        _, _, linexorg,lineyorg, linex, liney = make_line(nx=nx,ny=ny,minlen=minlen)
        mask = np.zeros(myimage.shape)
        for x,y in zip(linex,liney):
            try:
                myimage[x.astype(int),y.astype(int)]=mymax
                mask[x.astype(int),y.astype(int)]=1
            except:
                pass
        endpoints = [[linex[0], liney[0]],[linex[-1], liney[-1]]]
        container.append({
            'mask':mask,
            'endpoints':endpoints,
        })
    if isplot:
        fig = plt.figure(fignum,figsize=(20,20))
        plt.subplot(121)
        plt.imshow(orgimage,cmap='gray',origin='lower')
        plt.subplot(122)
        for item in container:
            plt.scatter(item['endpoints'][0][1],item['endpoints'][0][0],300,'green',marker='+')
            plt.scatter(item['endpoints'][1][1],item['endpoints'][1][0],300,'green',marker='+',)
        plt.imshow(myimage,cmap='gray',origin='lower')
        plt.xlim(0,nx)
        plt.ylim(0,ny)
        
    return myimage, container


