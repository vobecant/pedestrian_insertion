import json
import os
import ntpath


def process_folder(folder, save_dir):
    # train
    package_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(package_dir, folder, 'train')
    train_cities = os.listdir(train_dir)
    for c in train_cities:
        tmp_save_path = os.path.join(save_dir, 'train', c)
        city_path = os.path.join(train_dir, c)
        if not os.path.exists(tmp_save_path):
            os.makedirs(tmp_save_path)
        fnames = os.listdir(city_path)
        for fname in fnames:
            fpath = os.path.join(city_path, fname)
            process_json_file(fpath, tmp_save_path)

    # val
    val_dir = os.path.join(package_dir, folder, 'val')
    val_cities = os.listdir(val_dir)
    for c in val_cities:
        tmp_save_path = os.path.join(save_dir, 'val', c)
        city_path = os.path.join(val_dir, c)
        if not os.path.exists(tmp_save_path):
            os.makedirs(tmp_save_path)
        fnames = os.listdir(city_path)
        for fname in fnames:
            fpath = os.path.join(city_path, fname)
            process_json_file(fpath, tmp_save_path)


def process_json_file(json_file, save_dir):
    fname = os.path.join(save_dir, ntpath.basename(json_file)[:-23] + '_leftImg8bit.txt')
    if not os.path.isfile(json_file):
        print('Given json file not found: {}'.format(json_file))
        return
    with open(json_file, 'r') as fp:
        loaded_json = json.load(fp)
        json2ann(loaded_json, fname)


def json2ann(json_obj, fname):
    name2idx = {'pedestrian': 0, 'rider': 1, 'sitting person': 2, 'person other': 3}
    img_h, img_w = json_obj['imgHeight'], json_obj['imgWidth']
    with open(fname, 'w+') as f:
        valid_objs = [obj for obj in json_obj['objects'] if obj['label'] in name2idx.keys()]
        n_objs = len(valid_objs)
        for i,obj in enumerate(valid_objs):
            x, y, w, h = normalize_and_center_bb(obj['bboxVis'], im_h=img_h, im_w=img_w)
            label_idx = name2idx[obj['label']]
            txt = '{} {} {} {} {}'.format(label_idx, x, y, w, h)
            if i<(n_objs-1):
                txt += '\n'
            f.write(txt)


def normalize_and_center_bb(bb, im_h, im_w):
    # bounding box are (x, y, w, h), where (x, y) is its top-left corner and (w, h) its width and height
    x, y, w, h = bb

    # normalize to [0,1]
    x, w = x / im_w, w / im_w
    y, h = y / im_h, h / im_h

    # shift to center
    x_center = x + w / 2
    y_center = y + h / 2

    return x_center, y_center, w, h


if __name__ == '__main__':
    folder = '/opt/datasets/cityscapes/gtBboxCityPersons'
    save_dir = '/home/tonda/projects/pedestrian_insertion/data/gtBbox_cityPersons_trainval'
    process_folder(folder, save_dir)