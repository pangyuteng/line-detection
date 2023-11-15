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


def get_grid(szs=(64,64,64),sms=(8,8)):
    szx,szy,szz=szs
    smx,smy=sms
    grid = {}#np.zeros((szx,szy)).astype(np.uint)

    # assign grid id.
    anchor_dict={}
    for x in range(szx):
        for y in range(szy):
            px = x//smx
            py = y//smy
            code = py + smx*px
            if code not in anchor_dict.keys():
                anchor_dict[code]=(x,y)
            grid[(x,y)]=(code,px,py)

    #plt.imshow(grid)
    #print(np.unique(grid))
    return grid, anchor_dict

def get_code(px,py,smx):
    code = py + smx*px
    return code

def make_data(N=5,szs=(64,64,64),sms=(8,8)):
    
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
            terrain, container = make_one_sample(nx=64,ny=64,minlen=10)
            # rescale input image
            terrain = (255*(terrain-np.max(terrain))/(np.max(terrain)-np.min(terrain))).astype(np.uint8)
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
