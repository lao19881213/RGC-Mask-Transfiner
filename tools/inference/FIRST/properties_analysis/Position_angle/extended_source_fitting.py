
FIRST_fits = '/home/data0/lbq/inference_data/FIRST_fits'
FIRST_mir = '/home/data0/lbq/inference_data/FIRST_mir'
FIRST_results = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRSY'

hdu = fits.open(os.path.join(input_dir, fits_file))[0]
med = np.nanmedian(hdu.data)
mad = np.nanmedian(np.abs(hdu.data - med))
local_sigma = mad / np.sqrt(2) * erfinv(2 * 0.75 - 1)
med = np.nanmedian(hdu.data)
clip_level = med + 3 * local_sigma
mir_file = mir_dir +"/" + os.path.splitext(imagefilename)[0] + ".mir"
logger.info("imfit in=%s 'region=boxes(%d,%d,%d,%d)' object=gaussian clip=%f" \
    % (mir_file, x1, int(hdu.data.shape[0]-y2), \
       x2, int(hdu.data.shape[0]-y1), clip_level))
miriad_cmd = "imfit in=%s 'region=boxes(%d,%d,%d,%d)' object=gaussian clip=%f" \
    % (mir_file, x1, int(hdu.data.shape[0]-y2), \
       x2, int(hdu.data.shape[0]-y1), clip_level)
