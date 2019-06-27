import skimage as sk
import skimage.color as clr
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion,dilation,opening,closing
from skimage.morphology import disk
import scipy
import scipy.ndimage as ndi
import sklearn


def show_bunchof_images(item_images):
    import matplotlib.pyplot as plt
    nvals=len(item_images)
    subplotx = np.floor(np.sqrt(nvals / 5 * 8)) + 1
    subploty = np.floor(nvals / subplotx)+1
    iterr=0
    for it in item_images:
        iterr+=1
        plt.subplot(subploty,subplotx,iterr)
        plt.imshow(it)
    plt.show()

def segmentImage(imagepath='/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/abas afshar/1.tif',image=None,plot=0):
    if image is None:
        image=sk.io.imread(imagepath)
    image = image / np.max([np.max(image), 1]) * 255.0
    isnotint=np.all((image<1) & (image>0))
    if isnotint:
        image=np.floor(255*image)
    img1=clr.rgb2gray(image)
    ###> remove background
    img2 = sk.filters.gaussian(img1, sigma=1, mode='reflect')  ##(TODO): tune sigma
    img2=img1*((img2<220).astype(np.int)) ##(TODO): tune thresh
    ###> morphology process: (dilate or open ##todo)
    dsk=disk(2) ##(TODO): tune
    img4=closing(img2,selem=dsk)
    ###> edge detection:
    #### case1:
    gradx_kernel, grady_kernel= createSobelKernel(0) ##(TODO): tune
    gradx=ndi.convolve(img4,gradx_kernel)
    grady=ndi.convolve(img4,grady_kernel)
    img5=np.hypot(gradx,grady) # sobel gradient intensity
    img5=img5/np.max([1.0,np.max(img5)])*255.0
    img6=img5>150
    ### normally threshold the edge-detected:
    img6=(img6>0)#.astype(np.int8)
    ###> now list all cropped chromosomes' image and corresponding mask
    labels=sk.measure.label(img6)
    regions=sk.measure.regionprops(labels)
    item_masks=[region.filled_image for region in regions]
    item_crops=[img1[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] for region in regions]
    item_images=[crop_image(tup[0],tup[1]) for tup in zip(item_crops,item_masks)]
    ###> prune unwanted ones:
    wantedsizes=np.asarray([np.sum(np.sum(x)) for x in item_masks])>300
    wantedsizes=wantedsizes & (np.asarray([np.sum(np.sum(x))/np.prod(x.shape) for x in item_masks])>0.3)
    item_masks=[item_masks[itm] for itm in np.where(wantedsizes)[0]]
    item_crops=[item_crops[itm] for itm in np.where(wantedsizes)[0]]
    item_images=[item_images[itm] for itm in np.where(wantedsizes)[0]]
    if plot==1:
        show_bunchof_images(item_images)
    return item_images,item_masks



def otsu_thresh(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth
        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        value = Wb * Wf * (mub - muf) ** 2
        if value > final_value:
            final_thresh = t
            final_value = value
        final_img = gray.copy()
        print(final_thresh)
        final_img[gray > final_thresh] = 255
        final_img[gray < final_thresh] = 0
        return final_thresh




def plot_comparison(original, filtered, filter_name='filtered'):
    import matplotlib.pyplot as plt
    # ensure in integer form:
    #original=original/np.max([1.0,np.max(original)])*255.0
    #filtered=filtered/np.max([1.0,np.max(filtered)])*255.0
    original = original / np.max([np.max(original), 1]) * 255.0
    filtered = filtered / np.max([np.max(filtered), 1]) * 255.0
    original = original.astype(np.int)
    filtered = filtered.astype(np.int)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')


def createSobelKernel(n):
    side=n*2+3
    Kx=np.zeros([side,side])
    Ky=np.copy(Kx)
    halfside=side//2
    for i in range(side):
        k= halfside+i if i<=halfside else side+halfside-i-1
        for j in range(side):
            if j<halfside:
                Kx[i,j]=Ky[j,i]=j-k
            elif j>halfside:
                Kx[i,j]=Ky[j,i]=k-(side-j-1)
            else:
                Kx[i,j]=Ky[j,i]=0
    return Kx,Ky


def crop_image(image, crop_mask):  #### only for 2d monochrom
    ''' Crop the non_zeros pixels of an image  to a new image
    '''
    from skimage.util import crop
    pxlst = np.where(crop_mask.ravel())[0]
    dims = crop_mask.shape
    imgwidthy = dims[1]  # dimension in y, but in plot being x
    imgwidthx = dims[0]  # dimension in x, but in plot being y
    pixely = pxlst % imgwidthy
    pixelx = pxlst // imgwidthy
    minpixelx = np.min(pixelx)
    minpixely = np.min(pixely)
    maxpixelx = np.max(pixelx)
    maxpixely = np.max(pixely)
    crops = crop_mask * image  # (TODO): Computational Burden issue
    img_crop = crop(crops, ((minpixelx, imgwidthx - maxpixelx - 1),
                            (minpixely, imgwidthy - maxpixely - 1)))
    return img_crop

def crop_colored_image(imag, crop_mask):  #### only for 2d monochrom
    ''' Crop the non_zeros pixels of an image  to a new image
    '''
    import numpy as np
    image = np.copy(imag)
    rgbIn1stDimension=image.shape[0]==3
    if rgbIn1stDimension:
        image=image.transpose([1,2,0])

    from skimage.util import crop
    pxlst = np.where(crop_mask.ravel())[0]
    dims = crop_mask.shape
    imgwidthy = dims[1]  # dimension in y, but in plot being x
    imgwidthx = dims[0]  # dimension in x, but in plot being y
    # x and y are flipped???
    # matrix notation!!!
    pixely = pxlst % imgwidthy
    pixelx = pxlst // imgwidthy
    minpixelx = np.min(pixelx)
    minpixely = np.min(pixely)
    maxpixelx = np.max(pixelx)
    maxpixely = np.max(pixely)
    crops1 = crop_mask * image[0, :, :] + (255-255*crop_mask)  # (TODO): Computational Burden issue
    crops2 = crop_mask * image[1, :, :] + (255 - 255 * crop_mask)  # (TODO): Computational Burden issue
    crops3 = crop_mask * image[2, :, :] + (255 - 255 * crop_mask)  # (TODO): Computational Burden issue
    img_crop1 = crop(crops1, ((minpixelx, imgwidthx - maxpixelx - 1),
                            (minpixely, imgwidthy - maxpixely - 1)))
    img_crop2 = crop(crops2, ((minpixelx, imgwidthx - maxpixelx - 1),
                            (minpixely, imgwidthy - maxpixely - 1)))
    img_crop3 = crop(crops3, ((minpixelx, imgwidthx - maxpixelx - 1),
                            (minpixely, imgwidthy - maxpixely - 1)))
    img_crop = np.asarray(img_crop1,img_crop2,img_crop3)
    if not rgbIn1stDimension:
        img_crop= np.transpose(img_crop,[1,2,0])
    return img_crop

def crop_image2(image, crop_mask):  #### only for 2d monochrom
    ''' Crop the non_zeros pixels of an image  to a new image
    '''
    from skimage.util import crop
    pxlst = np.where(crop_mask.ravel())[0]
    dims = crop_mask.shape
    imgwidthy = dims[1]  # dimension in y, but in plot being x
    imgwidthx = dims[0]  # dimension in x, but in plot being y
    # x and y are flipped???
    # matrix notation!!!
    pixely = pxlst % imgwidthy
    pixelx = pxlst // imgwidthy
    minpixelx = np.min(pixelx)
    minpixely = np.min(pixely)
    maxpixelx = np.max(pixelx)
    maxpixely = np.max(pixely)
    crops = crop_mask * image + (255-255*crop_mask)  # (TODO): Computational Burden issue
    img_crop = crop(crops, ((minpixelx, imgwidthx - maxpixelx - 1),
                            (minpixely, imgwidthy - maxpixely - 1)))
    return img_crop



