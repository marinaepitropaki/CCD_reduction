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

def complete_image(args):

    master_path= Path(args.raw_files, 'masters')
    master_path.mkdir(exist_ok=True)
    master_images=ccdp.ImageFileCollection(master_path)

    raw_gaias = ccdp.ImageFileCollection(args.raw_images)
    raw_gaias_im = raw_gaias.files_filtered(imagetyp='light', include_path=True)

    reduced_path= Path(args.raw_images, 'reduced')
    reduced_path.mkdir(exist_ok= True)
    reduced_images=ccdp.ImageFileCollection(reduced_path)

    aligned_path= Path(args.raw_images, 'aligned')
    aligned_path.mkdir(exist_ok=True)
    aligned_images=ccdp.ImageFileCollection(aligned_path)

    stacked_path=Path(args.raw_images, 'stacked')
    stacked_path.mkdir(exist_ok=True)
    stacked_images=ccdp.ImageFileCollection(stacked_path)



    # Reduce images (remove bias, darks, flats)
    combined_darks = {ccd.header['EXPOSURE']: ccd for ccd in master_images.ccds(imagetyp='dark', combined=True)}
    combined_flats = {ccd.header['filter']: ccd for ccd in master_images.ccds(imagetyp='flat', combined=True)}

    # There is only one bias, so no need to set up a dictionary.
    combined_bias = [ccd for ccd in master_images.ccds(imagetyp='bias', combined=True)][0]

    combined_dark = CCDData.read(master_images.files_filtered(imagetyp='dark', 
                                                                combined=True, 
                                                                include_path=True)[0])
    combined_flat = CCDData.read(master_images.files_filtered(imagetyp='flat', 
                                                                combined=True, 
                                                                include_path=True)[0])


    combo_calibs = master_images.summary[master_images.summary['combined'].filled(False).astype('bool')]
    combo_calibs['date-obs', 'file', 'imagetyp', 'filter', 'exposure']

    raw_gaias.summary

    all_reds = []
    light_ccds = []
    for light, file_name in raw_gaias.ccds(imagetyp='light', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        light_ccds.append(light)

        light = ccdp.subtract_bias(light, combined_bias)

        light = ccdp.subtract_dark(light, combined_dark, exposure_time='EXPOSURE', exposure_unit= u.s )
        
        good_flat = combined_flats[light.header['filter']]
        light = ccdp.flat_correct(light, good_flat)
        all_reds.append(light)
        light.write(reduced_path / file_name, overwrite=True)


    reduced_images.refresh()
    reduced_images.summary

    #Align Images

    sloan_filters = set(reduced_images.summary['filter'])
    print(sloan_filters)

    for fltr in sorted(sloan_filters):
        
        sloan_fltr= reduced_images.ccds(imagetyp='light',filter=fltr, 
                                        return_fname=True, ccd_kwargs={'unit': 'adu'})
        sloan_fltr_list = list(sloan_fltr)
        ref_im=sloan_fltr_list[0][0]
        print(ref_im)
        for light, file_name in sloan_fltr_list :
            print(type(light))
            aligned_light=light.copy()
            aligned_image, footprint= aa.register(light.data.byteswap().newbyteorder(), 
                                                ref_im.data.byteswap().newbyteorder())
            
            aligned_light.data=aligned_image
            aligned_light.meta['aligned'] = True
            aligned_file_name = f'aligned_light_file_{fltr}.fit'
            aligned_light.write(aligned_path / file_name , overwrite=True)


    aligned_images.refresh()

    #Stack Images

    stack_filters = set(aligned_images.summary['filter'])
    print(stack_filters)


    for filtr in sorted(stack_filters):
        stacked_images = aligned_images.files_filtered(imagetyp='light', filter=filtr,
                                                        include_path=True)

        stacked_image = ccdp.combine(stacked_images,
                                    method='median',
                                    sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                    sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                    mem_limit=350e6
                                    )

        stacked_image.meta['stacked'] = True

        stacked_file_name = f'stacked_image_{filtr}.fit'
        stacked_image.write(stacked_path / stacked_file_name, overwrite=True)
        return raw_gaias_im, stacked_images


def show_gaia(raw_gaias_im, stacked_images, show=True):

    #plot uncalibrated image and calibrated image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    convenience_functions.show_image(CCDData.read(raw_gaias_im[0], unit='adu').data, cmap='gray', ax=ax1, fig=fig)
    ax1.set_title('Before reduction')
    convenience_functions.show_image(CCDData.read(stacked_images[0], unit='adu').data, cmap='gray', ax=ax2, fig=fig)
    ax2.set_title('After reduction')
    plt.show()

if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='Image after reduction')
    parser.add_argument('raw_files', type=str, default='/home/marinalinux/Downloads/data/bdf/', help='path of the raw files')
    parser.add_argument('raw_images', type=str, default='/home/marinalinux/Downloads/data/gaia19dke', help='path of the raw images')
    args = parser.parse_args()
    raw_gaias_im, stacked_images= complete_image(args)
    show_gaia(raw_gaias_im, stacked_images)


