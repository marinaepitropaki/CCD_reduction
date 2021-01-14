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
import convenience_functions

def load_data(args):

    bdf_files = ccdp.ImageFileCollection(args.data_path)
    raw_darks = bdf_files.files_filtered(imagetyp='Dark Frame', include_path=True)

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
    
    return bdf_files, raw_darks, calibrated_path,calibrated_images, output_path, master_images


def dark_creation(bdf_files, calibrated_path,calibrated_images, output_path,master_images, args):
    if args.cal_bias:
        combined_bias = CCDData.read(master_images.files_filtered(imagetyp='Bias Frame', combined=True,include_path=True)[0])

    print('list of the dark files')
    for ccd, f_name in bdf_files.ccds(imagetyp='Dark Frame', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        print(f_name)

        if args.cal_bias:
            ccd = ccdp.subtract_bias(ccd, combined_bias)

        ccd.write(calibrated_path / f_name, overwrite=True)

    calibrated_images.refresh()

    darks = calibrated_images.summary['imagetyp'] == 'Dark Frame'
    dark_times = set(calibrated_images.summary['exptime'][darks])
    print('the exposure time of the darks:', dark_times)

    for exp_time in sorted(dark_times):
        calibrated_darks = calibrated_images.files_filtered(imagetyp='Dark Frame', exptime=exp_time,
                                                        include_path=True)

        combined_dark = ccdp.combine(calibrated_darks,
                                    method='median',
                                    sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                    sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                    mem_limit=350e6
                                    )

        combined_dark.meta['combined'] = True

        dark_file_name = 'combined_dark_{:6.3f}.fit'.format(exp_time)
        combined_dark.write(output_path / dark_file_name, overwrite=True)

    return combined_dark

def show_dark(raw_darks, combined_dark, show=True):

    #plot single dark and combined dark
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    convenience_functions.show_image(CCDData.read(raw_darks[0], unit='adu').data, cmap='gray', ax=ax1, fig=fig)
    ax1.set_title('Uncalibrated dark')
    convenience_functions.show_image(combined_dark.data, cmap='gray', ax=ax2, fig=fig)
    ax2.set_title('{} dark images combined'.format(len(raw_darks)))
    plt.show()


if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='Directory of the bias and the darks')
    parser.add_argument('data_path', type=str, help='path of the bdf files')
    parser.add_argument('output_path', type=str,nargs='?', default='', help='path where the combined files should be saved' )
    parser.add_argument('calib_path', type=str,help='path where the calibrated files should be saved' )
    parser.add_argument('cal_bias',type=str, nargs='?', default='', help='if true, there is bias to be calculated')
    args = parser.parse_args()
    bdf_files, raw_darks, calibrated_path,calibrated_images, output_path, master_images=load_data(args)
    combined_dark = dark_creation(bdf_files, calibrated_path,calibrated_images, output_path, master_images, args)
    show_dark(raw_darks, combined_dark)
