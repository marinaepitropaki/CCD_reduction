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




def bias_creation(args):
    raw_files = ccdp.ImageFileCollection(args.raw_files)

    master_path= Path(args.raw_files, 'masters')
    master_path.mkdir(exist_ok=True)
    master_images=ccdp.ImageFileCollection(master_path)


    raw_biases = raw_files.files_filtered(imagetyp='bias', include_path=True)

    combined_bias = ccdp.combine(raw_biases,
                                method='median',
                                sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                mem_limit=350e6, unit='adu')

    combined_bias.meta['combined'] = True

    combined_bias.write(master_path / 'combined_bias.fit', overwrite=True)
    return raw_biases, combined_bias

def show_bias(raw_biases, combined_bias, show=True):

    #plot single bias and combined bias
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    convenience_functions.show_image(CCDData.read(raw_biases[0], unit='adu').data, cmap='gray', ax=ax1, fig=fig)
    ax1.set_title('Single bias')
    convenience_functions.show_image(combined_bias.data, cmap='gray', ax=ax2, fig=fig)
    ax2.set_title('{} bias images combined'.format(len(raw_biases)))
    plt.show()





if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='Directory of the bias')
    parser.add_argument('raw_files', type=str, default='/home/marinalinux/Downloads/data/bdf/', help='path of the raw files')
    args = parser.parse_args()
    raw_biases, combined_bias= bias_creation(args)
    show_bias(raw_biases, combined_bias)


