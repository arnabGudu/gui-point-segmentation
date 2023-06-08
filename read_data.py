import os
import math
import argparse
import pandas as pd


def read_data(data_dir, mapping_file):
    length, volume = [], []
    files = os.listdir(data_dir)
    mp = pd.read_csv(mapping_file)

    for file in files:
        if not file.endswith('.xlsx'):
            continue
        file_path = os.path.join(data_dir, file)
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            for index, row in df.iterrows():
                params = mp[(mp['filename'] == os.path.splitext(file)[0]) & (mp['sheet'] == ''.join(sheet_name.split(' ')))]

                if params.empty:
                    print('No params found for file {} sheet {}'.format(file, sheet_name))
                    continue

                k = params['k'].values[0]
                q = params['q'].values[0]
                g = params['g'].values[0]

                l = row[8]
                t = row[9]
                v = row[15]

                lmod = l / (0.855 * (q ** 1.105) * (g ** -1.16) * (k ** -1.89))
                vmod = v / (39.83 * (q ** 1.730) * (g ** -1.75) * (k ** -2.07))
                length.append([math.log(t), math.log(lmod), params.index[0]])
                volume.append([math.log(t), math.log(vmod), params.index[0]])
    return length, volume

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read data from excel file')
    parser.add_argument('data_dir', nargs='?', default='data', help='path to the dir containing excel files')
    parser.add_argument('-m', '--mapping_file', default='mapping.csv', help='path to the mapping file')
    parser.add_argument('-l', '--length_out', default='length.csv', help='path to the output length file')
    parser.add_argument('-v', '--volume_out', default='volume.csv', help='path to the output volume file')
    args = parser.parse_args()

    length, volume = read_data(args.data_dir, args.mapping_file)

    with open(args.length_out, 'w') as f:
        for item in length:
            f.write("%s\n" % ','.join(map(str, item)))
    
    with open(args.volume_out, 'w') as f:
        for item in volume:
            f.write("%s\n" % ','.join(map(str, item)))
