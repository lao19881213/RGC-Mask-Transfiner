import numpy as np
import argparse


parser=argparse.ArgumentParser(description="Split CSV File")
parser.add_argument("fin", type=str, help="input CSV file", default='extended_sources.csv')
parser.add_argument("-o", type=str, dest='out', 
        help="output CSV file", default='sources')
parser.add_argument("-s", type=int, dest='lines', 
        help="Number lines in split files. Default 360", default=360)
parser.add_argument("-m", type=int, dest='mode', 
        help="Work Mode. 0 split lines, 1 interesting, 2 missing", default=2)
args=parser.parse_args()

lin = args.lines
Mode = args.mode

if Mode == 0:
    with open(args.fin, encoding='utf-8') as fn:
        data = np.loadtxt(fn, str, delimiter=',', skiprows=1)
        fn.close()
    file_num = (data.shape[0])//lin
    print(file_num)
    for i in range(file_num):
        file_name = 'split/%s.%d.csv'%(args.out, (i+1))
        if i == file_num -1 :
            s_data = data[i*lin:, :]
        else:
            s_data = data[i*lin: (i+1)*lin, :]

        with open(file_name, 'w') as f:
            f.write('RA, DEC\n')
            for j in range(s_data.shape[0]):
                f.write('%s,%s\n'%(s_data[j, 0], s_data[j, 1]))
            f.close()

elif Mode == 1:
    file_name = '%s.csv'%args.out
    with open(args.fin, encoding='utf-8') as fn:
        data = np.loadtxt(fn, str, delimiter=',', skiprows=1)
        fn.close()
    line_num = data.shape[0]
    print(data.shape)
    with open(file_name, 'w') as f:
        f.write('RA, DEC\n')
        for j in range(line_num):
            f.write('%s,%s\n'%(data[j, 12], data[j, 13]))
        f.close()

elif Mode == 2:
    file_name = '%s.csv'%args.out
    with open(args.fin, encoding='utf-8') as fn:
        data = np.loadtxt(fn, str, delimiter=',', skiprows=1)
        fn.close()
    line_num = data.shape[0]
    print(data.shape)
    with open(file_name, 'w') as f:
        f.write('RA, DEC\n')
        for j in range(line_num):
            f.write('%s,%s\n'%(data[j, 11], data[j, 12]))
        f.close()


