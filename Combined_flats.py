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



if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='Directory of the bias, darks and flats')
    parser.add_argument('raw_files', type=str, default='/home/marinalinux/Downloads/data/bdf/', help='path of the raw files')
    args = parser.parse_args()


raw_files = ccdp.ImageFileCollection('args.raw_files')

calibrated_path = Path('args.raw_files', 'calibrated')
calibrated_path.mkdir(exist_ok=True)
calibrated_images = ccdp.ImageFileCollection(calibrated_path)

master_path= Path('args.raw_files', 'masters')
master_path.mkdir(exist_ok=True)
master_images=ccdp.ImageFileCollection(master_path)



flat_image_type = 'FLAT'
flat_image_type
set(raw_files.summary['exptime'][raw_files.summary['imagetyp'] == 'FLAT'])
combined_bias = list(master_images.ccds(combined=True, imagetyp='bias'))[0]
combined_bias

combined_dark = CCDData.read(master_images.files_filtered(imagetyp='dark', 
                                                               combined=True, 
                                                               include_path=True)[0])
combined_dark


for a_flat, f_name in raw_files.ccds(imagetyp='flat', return_fname=True, ccd_kwargs={'unit': 'adu'}):
    print(f_name)
    a_flat = ccdp.subtract_bias(a_flat, combined_bias)
    a_flat = ccdp.subtract_dark(a_flat, combined_dark, exposure_time='EXPOSURE', exposure_unit=u.s, scale=True)
    a_flat.write(calibrated_path / f_name, overwrite=True)

calibrated_images.refresh()
flats = calibrated_images.summary['imagetyp'] == 'FLAT'
flats
flat_filters = set(calibrated_images.summary['filter'][flats])
print(flat_filters)

for filtr in sorted(flat_filters):
    calibrated_flats = calibrated_images.files_filtered(imagetyp='flat', filter=filtr,
                                                     include_path=True)

    combined_flat = ccdp.combine(calibrated_flats,
                                 method='median',
                                 sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                 sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                 mem_limit=350e6
                                )

    combined_flat.meta['combined'] = True

    flat_file_name = f'combined_flat_{filtr}.fit'
    combined_flat.write(master_path / flat_file_name, overwrite=True)

calibrated_images.refresh()