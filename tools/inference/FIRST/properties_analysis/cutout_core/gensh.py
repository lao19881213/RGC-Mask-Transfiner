import os
import shutil
import argparse

parser=argparse.ArgumentParser(description="Gen sh File")
parser.add_argument("-f", type=str, dest='fin',
        help="input CSV file", default='split_source')
parser.add_argument("-o", type=str, dest='out',
        help="output CSV file", default='sources')
parser.add_argument("-s", type=int, dest='wins',
        help="Number windows. Default 3", default=3)
parser.add_argument("-t", type=int, dest='files',
        help="Number files. Default 100", default=100)
parser.add_argument("-m", type=int, dest='mode',
        help="Work Mode. 0 split downloads, 1 copy img", default=0)
args=parser.parse_args()
Mode = args.mode

if Mode == 0:
    Win_Size = args.wins
    B_Size = args.files
    File_Name = 'run_download_w%d.sh'
    # Cmd = 'python3 ../fetch_cutouts.py fetch-batch -f split/%s.%d.csv -s WISE -r 2 -g MOSAIC\n'
    Cmd = 'python3 ../fetch_cutouts.py fetch-batch -f split/%s.%d.csv -s FIRST,WISE -r 2 -g MOSAIC\n'
    for i in range(Win_Size):
        with open(File_Name%(i+1), 'a') as f:
            for j in range(B_Size):
                f.write(Cmd%(args.fin, i*B_Size+j+1))
elif Mode == 1:
    Dir = './'
    # Out_dir = '/ibo9000/MWA/WISE/'
    # Out_dir = '/o9000/MWA/GLEAM/hetu_images/deep_learn/inference_sets/catalog/interesting_sources/WISE/'
    Stype = ['HT','O','S-shape']
    Vtype = ['FIRST','PanSTARRS','WISE']
    Out_dir = '/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/catalog/interesting_sources_new/%s/%s/' #%(Stype[0],Vtype[1])
    Count = 0
    for root, _, files in os.walk(Dir):
        for fits in files:
            if fits.endswith(".fits"):
                Count += 1
                new_file = fits.split("_")[2]+'.fits'
                In_File = os.path.join(root, fits)
                for i in range(len(Stype)):
                    for j in range(len(Vtype)):
                        # print(root)
                        # print(Stype[i], Vtype[i])
                        # print(Stype[i] in root, Vtype[j] in root)
                        if (Stype[i] in root) and (Vtype[j] in root):
                            Out_File = os.path.join(Out_dir%(Stype[i],Vtype[j]), new_file)                
                shutil.copy(In_File, Out_File)
            if Count%100 == 0:
                print(Count)
elif Mode == 21:
    Dir = './'
    Vtype = ['FIRST','WISE']
    Out_dir = '%s/%s/' 
    Count = 0
    for root, _, files in os.walk(Dir):
        for fits in files:
            if fits.endswith(".fits"):
                Count += 1
                new_file = fits.split("_")[2]+'.fits'
                In_File = os.path.join(root, fits)
                for j in range(len(Vtype)):
                    # print(root)
                    # print(Stype[i], Vtype[i])
                    # print(Stype[i] in root, Vtype[j] in root)
                    if (Vtype[j] in root):
                        Out_File = os.path.join(Out_dir%(args.out, Vtype[j]), new_file)                
                shutil.copy(In_File, Out_File)
            if Count%100 == 0:
                print(Count)
elif Mode == 2:
    Dir = './'
    Out_dir = args.out
    Count = 0
    for root, _, files in os.walk(Dir):
        for fits in files:
            if fits.endswith(".fits"):
                Count += 1
                new_file = fits.split("_")[2]+'.fits'
                # old_file = fits.split("_")[2]+'.fits'
                # new_ra = format(np.round(float(old_file[1:10]), decimals=1), '0>8.1f')
                # new_dec = old_file[10:]
                # new_file = 'J%s%s'%(new_ra, new_dec)
                In_File = os.path.join(root, fits)
                # print(old_file, new_ra, new_dec, new_file)
                # print(Stype[i], Vtype[i])
                # print(Stype[i] in root, Vtype[j] in root)
                Out_File = os.path.join(Out_dir, new_file)                
                shutil.copy(In_File, Out_File)
            if Count%100 == 0:
                print(Count)
elif Mode == 3:
    Dir = args.fin
    csv_file = 'missing.csv'
    Count = 0
    with open(csv_file, 'w') as f:
        f.write('File\n')
        for root, _, files in os.walk(Dir):
            for _, _, Out_File in os.walk(args.out):
                for fits in files:
                    if fits.endswith(".fits"):
                        Count += 1
                        if not (fits in Out_File):
                            f.write(fits+'\n')