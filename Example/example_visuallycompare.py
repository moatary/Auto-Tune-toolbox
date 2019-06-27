from visually_compare_all_tuning_cases_imageprocess import comparetunes

#### Load predefinitions and image
import skimage as sk
import skimage.color as clr
import numpy as np
from matplotlib import pyplot
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
import scipy
import scipy.ndimage as ndi
from matplotlib.pyplot import imshow
from matplotlib.pyplot import subplot

arrr = sk.io.imread('cat.jpg')

img=sk.color.rgb2gray(arrr)

function1_in_pipeline=  sk.filters.gaussian # note: 1st input is always image, from 2nd input to end, one must be assigned as 'tune' input.
#function2_in_pipeline= lambda inn,x:opening(inn,disk(x))# lambda x,y:np.hypot( ndi.convolve(x,createSobelKernel(1)[0]),ndi.convolve(x,createSobelKernel(1)[1])) # note: 1st input is always image, from 2nd input to end, one must be assigned as 'tune' input.

Input_Image_to_tune= img
pipeline= [function1_in_pipeline]#, function2_in_pipeline]
inputs_to_eachfunc_assign_inputs2gettuned =[['tune'],['tune'] ] # each function may only have one tuning variable or zero, assigned by mentioning 'tune'. Other values are only assigned as its initial value.
each_tuning_param_values=[[8,9,10,12,20,30,35,40,50,70,75,80,90,100]]#, [3,4,5,6,7,8,9]] # [8,10,12,20,30,35,40] for gaussian filter tuning , and [1,2,3,4] for sobel function.

comparetunes(Input_Image_to_tune, pipeline, inputs_to_eachfunc_assign_inputs2gettuned, each_tuning_param_values)
