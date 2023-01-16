
import numpy as np
import os.path as osp
import pickle

if __name__ == "__main__":
    with open('bucket_targeted_amount_0.8_saved_trajs.pkl', 'rb') as handle:
        data = pickle.load(handle)
    # unit: mm
    scale = 117 / data['bucket_front_length']
    data['bucket_front_length'] = 117
    print('scale:', scale)
    print(data.keys())
    scaled_keys = ['tank_height', 'tank_border_thickness', 'tank_length', 
                   'tank_width', 'loader_pos_trajs', 'loader_vel_trajs', 
                   'waterline_trajs', 'targeted_pos_trajs']
    for key in scaled_keys:
        data[key] = data[key] * scale

    # print(data['tank_length'])
    # print(data['loader_pos_trajs'])
    # print(data['targeted_pos_trajs'])

    with open('scaled_bucket_targeted_amount_0.8_saved_trajs.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

    