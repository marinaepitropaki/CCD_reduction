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
    raw_biases = raw_files.files_filtered(imagetyp='bias', include_path=True)
    
    if args.master==True:
        master_path=Path(args.masters)
    else:
        master_path= Path(args.data_path, 'masters')

    master_path.mkdir(exist_ok=True)
    master_images=ccdp.ImageFileCollection(master_path)
    

    
    return raw_biases, master_path

def bias_creation(master_path, args):
    combined_bias = ccdp.combine(raw_biases,
                                method='median',
                                sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                mem_limit=350e6, unit='adu')

    combined_bias.meta['combined'] = True

    combined_bias.write(master_path / 'combined_bias.fit', overwrite=True)
    return combined_bias

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
    parser.add_argument('data_path', type=str, help='path of the raw files')
    parser.add_argument('masters', type=str,nargs='?', default='arg.data_path/masters', help='path where the combined files should be saved' )
    parser.add_argument('master', type=bool, help='if True, the combined will be saved in the path given by the user IF NOT, PRESS arg.data_path/masters ')
    args = parser.parse_args()
    raw_biases, master_path= load_data(args)
    combined_bias= bias_creation(master_path, args)
    show_bias(raw_biases, combined_bias)


