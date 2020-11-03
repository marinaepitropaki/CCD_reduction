import glob
import ccdproc as ccdp
import os
from pathlib import Path
from astropy.nddata import CCDData
from astropy.visualization import hist
from astropy.stats import mad_std
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from ccdproc import ImageFileCollection
from astropy import units as u
import astroalign as aa

from astropy import visualization as aviz
from astropy.nddata.utils import block_reduce, Cutout2D
from matplotlib import pyplot as plt
import argparse
import convenience_functions
import image_sim


def find_nearest_dark_exposure(image, dark_exposure_times, tolerance=200, warning_tolerance=60):
    """
    Find the nearest exposure time of a dark frame to the exposure time of the image,
    raising an error if the difference in exposure time is more than tolerance.
    
    Parameters
    ----------
    
    image : astropy.nddata.CCDData
        Image for which a matching dark is needed.
    
    dark_exposure_times : list
        Exposure times for which there are darks.
    
    tolerance : float or ``None``, optional
        Maximum difference, in seconds, between the image and the closest dark. Set
        to ``None`` to skip the tolerance test.
    
    Returns
    -------
    
    float
        Closest dark exposure time to the image.
    """

    dark_exposures = np.array(list(dark_exposure_times))
    idx = np.argmin(np.abs(dark_exposures - image.header['exptime']))
    closest_dark_exposure = dark_exposures[idx]

    if (tolerance is not None and 
        np.abs(image.header['exptime'] - closest_dark_exposure) > warning_tolerance):
        pritn('WARNING:HIGH TOLERANCE')
#rewrite

    if (tolerance is not None and 
        np.abs(image.header['exptime'] - closest_dark_exposure) > tolerance):
        
        raise RuntimeError('Closest dark exposure time is {} for flat of exposure '
                           'time {}.'.format(closest_dark_exposure, image.header['exptime']))
        
    
    return closest_dark_exposure

def load_data(args):

    master_images = ccdp.ImageFileCollection(args.data_path)
    raw_gaias = ccdp.ImageFileCollection(args.raw_images)
    raw_gaias_im = raw_gaias.files_filtered(imagetyp='light', include_path=True)
    raw_gaias_im=sorted(raw_gaias_im)
   
    if args.red_path:
        reduced_path=Path(args.red_path)
    else:
        reduced_path=Path(args.raw_images, 'reduced')
    reduced_path.mkdir(exist_ok= True)
    reduced_images=ccdp.ImageFileCollection(reduced_path)

    if args.al_path:
        aligned_path=Path(args.al_path)
    else:
        aligned_path= Path(args.raw_images, 'aligned')
    aligned_path.mkdir(exist_ok=True)
    aligned_images=ccdp.ImageFileCollection(aligned_path)

    if args.st_path:
        stacked_path=Path(args.st_path)
    else:
        stacked_path=Path(args.raw_images, 'stacked')
    stacked_path.mkdir(exist_ok=True)
    stacked_images=ccdp.ImageFileCollection(stacked_path)

    return master_images, raw_gaias, raw_gaias_im,reduced_path,reduced_images,aligned_path,aligned_images, stacked_path,stacked_images

# Reduce images (remove bias, darks, flats)

def reducing(raw_gaias, master_images,reduced_path, reduced_images, args):
    if (args.cal_bias and args.cal_dark and args.cal_flats)=='':
        print('NOTHING TO REDUCE')
        all_reds = []
        light_ccds = []
        print(raw_gaias.summary)
        for light, file_name in raw_gaias.ccds(imagetyp='light', return_fname=True, ccd_kwargs={'unit': 'adu'}):
            light_ccds.append(light)
            all_reds.append(light)
            light.write(reduced_path / file_name, overwrite=True)

    else:   
        if args.cal_dark:
            combined_darks = {ccd.header['EXPOSURE']: ccd for ccd in master_images.ccds(imagetyp='dark', combined=True)}
            combined_dark = CCDData.read(master_images.files_filtered(imagetyp='dark', 
                                                                        combined=True, 
                                                                        include_path=True)[0])
        
        if args.cal_flats:
            combined_flats = {ccd.header['filter']: ccd for ccd in master_images.ccds(imagetyp='flat', combined=True)}

        if args.cal_bias:
            combined_bias = [ccd for ccd in master_images.ccds(imagetyp='bias', combined=True)][0]


        all_reds = []
        light_ccds = []
        print(raw_gaias.summary)
        for light, file_name in raw_gaias.ccds(imagetyp='light', return_fname=True, ccd_kwargs={'unit': 'adu'}):
            light_ccds.append(light)
            
            if args.cal_bias:
                light = ccdp.subtract_bias(light, combined_bias)
            if args.cal_dark:
                closest_dark = find_nearest_dark_exposure(light, combined_darks.keys())
                light = ccdp.subtract_dark(light, combined_darks[closest_dark], exposure_time='EXPOSURE', exposure_unit= u.s )
            if args.cal_flats:    
                good_flat = combined_flats[light.header['filter']]
                light = ccdp.flat_correct(light, good_flat)
            all_reds.append(light)
            light.write(reduced_path / file_name, overwrite=True)

    print('DATA REDUCTION: DONE')
    reduced_images.refresh()
    return reduced_path, reduced_images


#Align Images

def aligning (reduced_images, aligned_path, aligned_images, args):

    sloan_filters = set(reduced_images.summary['filter'])
    print('FILTERS USED:', sloan_filters)

    for fltr in sorted(sloan_filters) :    
        sloan_fltr= reduced_images.ccds(imagetyp='light',filter=fltr, 
                                            return_fname=True, ccd_kwargs={'unit': 'adu'})
        sloan_fltr_list = list(sloan_fltr)
        ref_im=sloan_fltr_list[0][0]
        

        for light, file_name in sloan_fltr_list :
            
            aligned_light=light.copy()
            aligned_image, footprint= aa.register(light.data.byteswap().newbyteorder(), 
                                                    ref_im.data.byteswap().newbyteorder())
                
            aligned_light.data=aligned_image
            aligned_light.meta['aligned'] = True
            aligned_file_name = f'aligned_light_file_{fltr}.fit'
            aligned_light.write(aligned_path / file_name , overwrite=True)    
    aligned_images.refresh()

    print('IMAGE ALIGNEMENT: DONE')
    return aligned_path, aligned_images


#Stacking Images

def stacking(aligned_images, stacked_path, stacked_images, args):


    stack_filters = set(aligned_images.summary['filter'])
    
    for filtr in sorted(stack_filters):
        stacked_image_list = aligned_images.files_filtered(imagetyp='light', filter=filtr,
                                                        include_path=True)
        stacked_image = ccdp.combine(stacked_image_list,
                                    method='median',
                                    sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                    sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                    mem_limit=350e6
                                    )
        stacked_image.meta['stacked'] = True
        stacked_file_name = f'stacked_image_{filtr}.fit'
        stacked_image.write(stacked_path / stacked_file_name, overwrite=True)
    stacked_images.refresh()
    print('IMAGE STACKING:DONE')
    return stacked_images, stack_filters


def show_gaia(raw_gaias, stacked_images, stack_filters, show=True):
    print('READY TO SHOW THE IMAGES')
    for filtr in sorted(stack_filters):
        raw_to_show=raw_gaias.files_filtered(imagetyp='light', filter=filtr, include_path=True)

        stacked_to_show= stacked_images.files_filtered(imagetyp='light', filter=filtr, include_path=True)
        #plot uncalibrated image and calibrated image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        convenience_functions.show_image(CCDData.read(raw_to_show[0], unit='adu').data, cmap='gray', ax=ax1, fig=fig)
        ax1.set_title('Before reduction image with {}'.format(filtr))
        convenience_functions.show_image(CCDData.read(stacked_to_show[0], unit='adu').data, cmap='gray', ax=ax2, fig=fig)
        ax2.set_title('After reduction image with {}'.format(filtr))
        plt.show()



if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='Image after reduction')
    parser.add_argument('-o','--data_path', type=str, help='path of the combined bdf(master) files')
    parser.add_argument('-r','--raw_images', type=str, help='path of the raw images')
    parser.add_argument('-m','--red_path', type=str,nargs='?', default='', help='path where the reduced files should be saved' )
    parser.add_argument('-a','--al_path', type=str,nargs='?', default='', help='path where the aligned files should be saved' )
    parser.add_argument('-s','--st_path', type=str,nargs='?', default='', help='path where the stacked files should be saved' )
    parser.add_argument('-b','--cal_bias',type=str, nargs='?', default='', help='if true, there is bias to be calculated')
    parser.add_argument('-d','--cal_dark',type=str, nargs='?', default='', help='if true, there is dark to be calculated')
    parser.add_argument('-f','--cal_flats', type=str, nargs='?',default='',help='if true, there is flat to be calculated')
    
    args = parser.parse_args()
    master_images, raw_gaias, raw_gaias_im,reduced_path,reduced_images,aligned_path,aligned_images, stacked_path,stacked_images= load_data(args)
    reduced_path, reduced_images = reducing(raw_gaias, master_images,reduced_path, reduced_images, args)
    aligned_path, aligned_images = aligning (reduced_images, aligned_path, aligned_images, args)
    stacked_images, stack_filters = stacking(aligned_images, stacked_path, stacked_images, args)
    show_gaia(raw_gaias, stacked_images, stack_filters)


