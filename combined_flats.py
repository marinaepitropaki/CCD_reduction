import ccdproc as ccdp
from pathlib import Path
import numpy as np
from astropy.stats import mad_std
import matplotlib.pyplot as plt
from astropy import visualization as aviz
from astropy.nddata.utils import block_reduce, Cutout2D
from matplotlib import pyplot as plt
from astropy.nddata import CCDData
from astropy import units as u
import os
import argparse
import convenience_functions


def inv_median(a):
     return 1 / np.median(a)

def load_data(args):

    bdf_files = ccdp.ImageFileCollection(args.data_path)
    raw_flats = bdf_files.files_filtered(imagetyp='flat', include_path=True)

    if args.calib_path:
        calibrated_path=Path(args.calib_path)
    else:
        calibrated_path = Path(args.data_path, 'calibrated')

    calibrated_path.mkdir(exist_ok=True)
    calibrated_images = ccdp.ImageFileCollection(calibrated_path)

    if args.output_path:
        output_path=Path(args.output_path)
    else:
        output_path= Path(args.data_path, 'masters')

    output_path.mkdir(exist_ok=True)
    master_images=ccdp.ImageFileCollection(output_path)

    calibrated_images.refresh()
    
    return bdf_files, raw_flats, calibrated_path,calibrated_images, output_path, master_images


def flat_creation(bdf_files, calibrated_path,calibrated_images, output_path, master_images, args):
   
    set(bdf_files.summary['exptime'][bdf_files.summary['imagetyp'] == 'FLAT'])

    if args.cal_bias:
        combined_bias = list(master_images.ccds(combined=True, imagetyp='bias'))[0]
    
    if args.cal_dark:
        combined_dark = CCDData.read(master_images.files_filtered(imagetyp='dark', 
                                                                    combined=True, 
                                                                    include_path=True)[0])
   

    print('list of the flat files')
    for a_flat, f_name in bdf_files.ccds(imagetyp='flat', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        print(f_name)

        if args.cal_bias:
            print('BIAS')
            a_flat = ccdp.subtract_bias(a_flat, combined_bias)
            print('DONE BIAS')
        if args.cal_dark:
            print('DARK')
            a_flat = ccdp.subtract_dark(a_flat, combined_dark, exposure_time='EXPOSURE', exposure_unit=u.s, scale=True)
            print('done dark')
        a_flat.write(calibrated_path / f_name, overwrite=True)
    calibrated_images.refresh()


    flats = calibrated_images.summary['imagetyp'] == 'FLAT'
    
    flat_filters = set(calibrated_images.summary['filter'][flats])
    print('flat filters:', flat_filters)

    for filtr in sorted(flat_filters):
        calibrated_flats = calibrated_images.files_filtered(imagetyp='flat', filter=filtr,
                                                        include_path=True)
        print(len(calibrated_flats))
        combined_flat = ccdp.combine(calibrated_flats,
                                    method='median', scale=inv_median,
                                    sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                    sigma_clip_func=np.ma.median, 
                                    signma_clip_dev_func=mad_std,
                                    mem_limit=350e6)

        combined_flat.meta['combined'] = True

        flat_file_name = f'combined_flat_{filtr}.fit'
        combined_flat.write(output_path / flat_file_name, overwrite=True)

    calibrated_images.refresh()
    master_images.refresh()

    
    return flat_filters


def show_flat(bdf_files, output_path, flat_filters, master_images, show=True):
    print (flat_filters)
  
   
    for filtr in sorted(flat_filters):
        
        flat_to_show =bdf_files.files_filtered(imagetyp='flat', filter=filtr,
                                                        include_path=True)
        
        combined_flat= master_images.files_filtered(imagetyp='flat', filter=filtr,
                                                        include_path=True)
        #plot single flat and combined flat
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        convenience_functions.show_image(CCDData.read(flat_to_show[0], unit='adu').data, cmap='gray', ax=ax1, fig=fig)
        ax1.set_title('Uncalibrated flat')
    
        convenience_functions.show_image(CCDData.read(combined_flat[0], unit= 'adu').data, cmap='gray', ax=ax2, fig=fig)
        
        ax2.set_title('{} flat images combined {}'.format(len(flat_to_show), filtr))
        plt.show()



if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='Directory of the bias, darks and flats')
    parser.add_argument('data_path', type=str, help='path of the bdf files')
    parser.add_argument('output_path', type=str,nargs='?', default='', help='path where the combined files should be saved' )
    parser.add_argument('calib_path', type=str,help='path where the calibrated files should be saved' )
    parser.add_argument('cal_bias',type=str, nargs='?', default='', help='if true, there is bias to be calculated')
    parser.add_argument('cal_dark',type=str, nargs='?', default='', help='if true, there is dark to be calculated')
    args = parser.parse_args()
    bdf_files, raw_flats, calibrated_path,calibrated_images, output_path, master_images= load_data(args)
    flat_filters=flat_creation(bdf_files, calibrated_path,calibrated_images, output_path, master_images, args)
    show_flat(bdf_files, output_path, flat_filters, master_images)