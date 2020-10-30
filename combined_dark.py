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

    raw_files = ccdp.ImageFileCollection(args.data_path)
    raw_darks = raw_files.files_filtered(imagetyp='dark', include_path=True)

    if args.calib==True:
        calibrated_path=Path(args.calibs)
    else:
        calibrated_path = Path(args.data_path, 'calibrated')

    calibrated_path.mkdir(exist_ok=True)
    calibrated_images = ccdp.ImageFileCollection(calibrated_path)

    if args.master==True:
        master_path=Path(args.masters)
    else:
        master_path= Path(args.data_path, 'masters')

    master_path.mkdir(exist_ok=True)
    master_images=ccdp.ImageFileCollection(master_path)

    calibrated_images.refresh()
    
    return raw_files, raw_darks, calibrated_path,calibrated_images, master_path


def dark_creation(raw_files, calibrated_path,calibrated_images, master_path, args):
    if args.not_bias == False:
        combined_bias = CCDData.read(master_images.files_filtered(imagetyp='bias', combined=True,include_path=True)[0])

    for ccd, f_name in raw_files.ccds(imagetyp='dark', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        print(f_name)

        if args.not_bias == False:
            ccd = ccdp.subtract_bias(ccd, combined_bias)

        ccd.write(calibrated_path / f_name, overwrite=True)

    calibrated_images.refresh()

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

def show_bias(raw_darks, combined_dark, show=True):

    #plot single bias and combined bias
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    convenience_functions.show_image(CCDData.read(raw_darks[0], unit='adu').data, cmap='gray', ax=ax1, fig=fig)
    ax1.set_title('Uncalibrated dark')
    convenience_functions.show_image(combined_dark.data, cmap='gray', ax=ax2, fig=fig)
    ax2.set_title('{} dark images combined'.format(len(raw_darks)))
    plt.show()


if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='Directory of the bias and the darks')
    parser.add_argument('data_path', type=str, help='path of the raw files')
    parser.add_argument('masters', type=str,help='path where the combined files should be saved' )
    parser.add_argument('master', type=bool, help='if True, the combined will be saved in the path given by the user IF NOT, PRESS in arg.data_path/masters ')
    parser.add_argument('calibs', type=str,help='path where the calibrated files should be saved' )
    parser.add_argument('calib', type=bool, help='if True, the calibrated will be saved in the path given by the user IF NOT, PRESS in arg.data_path/calibrated ')
    parser.add_argument('not_bias',type=bool, default=False, help='set True if there is no bias to be substracted')
    args = parser.parse_args()
    raw_files, raw_darks, calibrated_path,calibrated_images, master_path=load_data(args)
    combined_dark= dark_creation(raw_files, calibrated_path,calibrated_images, master_path, args)
    show_bias(raw_darks, combined_dark)
