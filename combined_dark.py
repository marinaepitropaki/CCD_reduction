import ccdproc as ccdp
from pathlib import Path
import numpy as np
from astropy.stats import mad_std
import matplotlib.pyplot as plt
from astropy import visualization as aviz
from astropy.nddata.utils import block_reduce, Cutout2D
from matplotlib import pyplot as plt
from astropy.nddata import CCDData
import argparse


def dark_creation(args):
    raw_files = ccdp.ImageFileCollection(args.raw_files)

    calibrated_path = Path(args.raw_files, 'calibrated')
    calibrated_path.mkdir(exist_ok=True)
    calibrated_images = ccdp.ImageFileCollection(calibrated_path)

    master_path= Path(args.raw_files, 'masters')
    master_path.mkdir(exist_ok=True)
    master_images=ccdp.ImageFileCollection(master_path)

    calibrated_images.refresh()
    master_images.summary
    calibrated_images.refresh()
    calibrated_images.summary
    combined_bias = CCDData.read(master_images.files_filtered(imagetyp='bias', combined=True,include_path=True)[0])

    combined_bias

    if args.notabias == True:
        for ccd, f_name in raw_files.ccds(imagetyp='dark', return_fname=True, ccd_kwargs={'unit': 'adu'}):
            print(f_name)
            ccd.write(calibrated_path / f_name, overwrite=True)

        calibrated_images.refresh()
        calibrated_images.summary
    else:
        
        for ccd, f_name in raw_files.ccds(imagetyp='dark', return_fname=True, ccd_kwargs={'unit': 'adu'}):
            print(f_name)
            ccd = ccdp.subtract_bias(ccd, combined_bias)
            ccd.write(calibrated_path / f_name, overwrite=True)

        calibrated_images.refresh()
        calibrated_images.summary

    darks = calibrated_images.summary['imagetyp'] == 'DARK'
    dark_times = set(calibrated_images.summary['exptime'][darks])
    print(dark_times)

    for exp_time in sorted(dark_times):
        calibrated_darks = calibrated_images.files_filtered(imagetyp='dark', exptime=exp_time,
                                                        include_path=True)

        combined_dark = ccdp.combine(calibrated_darks,
                                    method='median',
                                    sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                    sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                    mem_limit=350e6
                                    )

        combined_dark.meta['combined'] = True

        dark_file_name = 'combined_dark_{:6.3f}.fit'.format(exp_time)
        combined_dark.write(master_path / dark_file_name, overwrite=True)

    return combined_dark




if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='Directory of the bias and the darks')
    parser.add_argument('raw_files', type=str, default='/home/marinalinux/Downloads/data/bdf/', help='path of the raw files')
    parser.add_argument('notabias',type=bool, default=False, help='set True if there is no bias to be substracted')
    args = parser.parse_args()
    combined_dark= dark_creation(args)
