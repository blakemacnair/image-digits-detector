import numpy as np
import h5py
import pandas
from tqdm import tqdm

if __name__ == '__main__':
    filename = './data/train/digitStruct.mat'
    with h5py.File(filename, 'r') as h:
        d = h['digitStruct']
        mtype = d.attrs['MATLAB_class'].decode()  # Is 'struct' type

        boxes = d['bbox']
        names = d['name']

        assert len(boxes) == len(names)

        rows = []

        for i in tqdm(range(len(boxes))):
            name_ref = names[i][0]
            box_ref = boxes[i][0]

            name_raw = h[name_ref]
            box_raw = h[box_ref]

            name = ''.join([chr(x) for x in name_raw])

            row = {'name': name}

            box_vals = {}

            for key in box_raw:
                key_raw = box_raw[key]
                np_values = []

                if key_raw.shape == (1, 1):
                    # Single box in this frame, so don't try to iterate, just decode
                    value = key_raw[0][0]
                    np_values.append(value)
                else:
                    # Multiple boxes
                    for key_i in range(len(key_raw)):
                        key_ref = key_raw[key_i][0]
                        key_val = h[key_ref]
                        value = np.array(key_val, dtype=key_val.dtype)
                        np_values.append(value)

                np_values = np.array(np_values).flatten()

                box_vals[key] = np_values

            box_dataframe = pandas.DataFrame(box_vals)
            row['boxes'] = box_dataframe
            rows.append(row)

        print('no')
