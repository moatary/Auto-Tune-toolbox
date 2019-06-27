

def show_images_labels(item_images,labels='image',points2ploton1=[], points2ploton2=[],subplot=[],numer=False): #(TODO)
    import matplotlib.pyplot as plt
    if numer==True:
        if isinstance(labels,list):
            labels=['%02i:) '%(num+1) + labels[num] for num in range(len(item_images))]
        else:
            labels = ['%02i) ' % (num + 1) + labels for num in range(len(item_images))]
    if type(labels)!=type([1,2]): #if is list
        labels=['%s#%02i'%(labels,i) for i in range(len(item_images))]
    nvals=len(item_images)
    if subplot!=[]:
        subploty,subplotx= tuple(subplot)
    else:
        subploty = np.floor(np.sqrt(nvals / ((1+5**0.5)/2) ))
        subplotx = np.ceil(nvals / subploty)
    iterr=0
    plt.figure()
    for it in item_images:
        iterr+=1
        plt.subplot(subploty,subplotx,iterr)
        plt.imshow(it)
        plt.gca().set_xlabel(labels[iterr-1])

        if len(points2ploton1)!=0:
            l1=plt.plot(points2ploton1[iterr-1][:,1],points2ploton1[iterr-1][:,0],'ro', linewidth=0.125, markersize=0.125)
            #plt.setp(l1,linewidth=2)
        if len(points2ploton2)!=0:
            l2=plt.plot(points2ploton2[iterr-1][:,1],points2ploton2[iterr-1][:,0],'bo', linewidth=0.125, markersize=0.125)
    plt.show()

def comparetunes(img, transformers, inputs=[[]], tunes=[], start=[1], stop=[10], step=0, count=5, scale=1, shift=0, label=':',subplot=[]):
    ## transformer= lambda inp,tune: sk.filters.gaussian(inp, sigma=tune, mode='reflect')
    import numpy as np
    import skimage as sk
    import scipy.ndimage as ndi
    from image_preprocessing_tools_segmentation import createSobelKernel
    enum = -1
    if len(img)==0:
        print('')#return
    if transformers is None:
        print('')#return
    if len(tunes)==0:
        if not isinstance(start,list):
            if step!=0:
                tunes=np.arange(start,stop+0.5,step)
            elif count!=0:
                tunes=np.arange(start,stop+(stop-start)/(-1+count),(stop-start)/(-1+count))
            else:
                print('22')#return
        else:
            if (step!=[]) and (step!=0):
                tunes=[np.arange(star,sto+0.5,ste) for star,sto,ste in zip(start,stop,step)]
            elif count!=[]:
                tunes=[np.arange(star,sto+(sto-star)/(-1+coun),(sto-star)/(-1+coun)) for star,sto,coun in zip(start,stop,count)]
            else:
                print('23')#return
    else:
        if not isinstance(tunes[0], list):
            if not isinstance(scale,list):
                tunes=tunes*scale+shift
            else:
                tunes=np.asarray([tunes*scal+shif for scal,shif in zip(scale,shift)])
    tunefunind = []
    tuneinpind = [] 
    if not isinstance(tunes,list):
        tunes=[tunes]  # np.asarray([tunes.tolist()])
    # check if there is more than one transformers
    if isinstance(transformers,list):
        # generate transformer using only func names and their needed inputs
        if len(tunes)==1:
            if len(transformers)>1:
                tunes=np.asarray(tunes*len(transformers))
        if len(inputs)==0:
            print('')#return
        cur, funct=[lambda imgg,tunes: imgg], []
        parameterstate=-1
        isthereanytune=False
        part1in, part2in=[], []
        for numer,(func,inpt) in enumerate(zip(transformers,inputs)):
            for enum,each in enumerate(inpt):
                if each is 'tune':
                    tunefunind.append(numer) ### THIS IS FOR LATER
                    tuneinpind.append(enum) ### THIS IS FOR LATER
                    break
                enum=-1
            if enum!=-1:
                isthereanytune=True
                if enum==0:
                    part1in.append([])
                else:
                    part1in.append(inpt[:enum-1])
                if enum==len(inpt)-1:
                    part2in.append([])
                else:
                    part2in.append(inpt[enum+1:])
                parameterstate+=1
            else:
                part1in.append(inpt)
                part2in.append([])
            # in exec, phases must be done:
            variables={'transformers':transformers,'part1in':part1in,'part2in':part2in,'tunes':tunes,'cur':cur,'funct':funct}
            exec('funct.append( lambda inp,param: transformers[%i](inp,*part1in[%i],*param,*part2in[%i])  )'%(numer,numer,numer),variables)
            funct=variables['funct']
            if isthereanytune:
                variables = {'transformers': transformers, 'part1in': part1in, 'part2in': part2in, 'tunes': tunes,'cur': cur, 'funct': funct}
                exec('cur.append( lambda imgg,tunes: funct[%i](cur[%i](imgg,tunes),[tunes[%i]] )   )' % (numer,numer,parameterstate), variables)
                cur = variables['cur']
                isthereanytune=False
            else:
                variables = {'transformers': transformers, 'part1in': part1in, 'part2in': part2in, 'tunes': tunes,'cur': cur, 'funct': funct}
                exec('cur.append( lambda imgg,tunes: funct[%i](cur[%i](imgg,tunes),[] )  )' % (numer,numer), variables)
                cur = variables['cur']
        # then use cur as transformer:
        transformer = lambda imgg, tunes: cur[len(cur) - 3](imgg, tunes)
    else:
        transformer=transformers
        tunefunind = [0]
        tuneinpind = [0]
    # generate set of tuningparameters using mesh function:
    numparams=len(tunefunind)
    perparamtunecount=[len(tun) for tun in tunes]
    def base(x,n): # n is vec of tunes_length of each param
        u=[]
        current=x
        for i in range(len(n)):
            m=np.prod(n[i+1:])
            u.append(current//m)
            current = current % m
        return u
    mshfunc= lambda b: np.asarray([base(x,b) for x in range(np.prod(b))]).astype(np.int32).tolist()
    tunes_indices=mshfunc(perparamtunecount)
    # process of generating images of each tune case:
    imgs=[]
    if isinstance(transformers,list):
        funind2funnameMap=[transformers[ii].__name__ for ii in tunefunind]
    else:
        funind2funnameMap=[transformers.__name__]
    labels=[]

    for tune in tunes_indices:
        tunevaluesthiscase= [tunes[whichtuneindex][ind] for whichtuneindex,ind in enumerate(tune) ]
        try:
            imgtmp=transformer(img, tunevaluesthiscase)
        except:
            imgtmp=[]
        imgs.append(imgtmp)
        # generating labels:
        labels.append(' | '.join([funind2funnameMap[numerr]+label+str(ind) for numerr,ind in enumerate(tunevaluesthiscase)]))
    # process of showing images
    show_images_labels(imgs, labels=labels,subplot=subplot)



