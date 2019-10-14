import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from torchvision import transforms
import numpy as np
from shutil import copyfile
from PIL import Image
import pickle
from data.base_dataset import get_transform, get_params
import matplotlib.pyplot as plt


def insert_generated(original, generated, to_insert):
    merged = np.zeros_like(original)
    to_keep = np.ones_like(to_insert) - to_insert
    merged[np.where(to_keep)] = original[np.where(to_keep)]
    merged[np.where(to_insert)] = generated[np.where(to_insert)]
    merged = Image.fromarray(merged)
    return merged


def extend_bbs(original_txt_file, new_bbs, new_file):
    new_file_dir, _ = os.path.split(new_file)
    if not os.path.exists(new_file_dir):
        os.makedirs(new_file_dir)

    # copy the content of the original file
    copyfile(original_txt_file, new_file)

    # and append new bbs
    with open(new_file, 'a+') as f:
        n_bbs = len(new_bbs)
        for i, new_bb in enumerate(new_bbs):
            assert all([coord <= 1 for coord in new_bb])
            x, y, w, h = new_bb
            line = '0 {} {} {} {}'.format(x, y, w, h)
            if i < (n_bbs - 1):
                line += '\n'
            f.write(line)


def crop_and_save(image, instances, bbs, crop_size, counter, save_dir, inflate=1.1):
    '''
    :param image: image with the generated new pedestrians
    :param bbs: CENTERED bounding boxes in format [x_center, y_center, bb_width, bb_height] normalized to [0,1]
    :param crop_size: resulting size of the cropped image
    :param counter: integer used for naming the files
    :param save_dir: path to the directory where the cropped images should be saved
    '''
    # normalized bbs to image coordinates
    img_w, img_h = image.size
    bbs_img = []
    for bb in bbs:
        x_c, w = int(bb[0] * img_w), int(bb[2] * img_w)
        y_c, h = int(bb[1] * img_h), int(bb[3] * img_h)
        bb_img = [x_c, y_c, w, h]
        bbs_img.append(bb_img)

    # crop image and mask
    for x_c, y_c, w, h in bbs_img:

        inst_id = instances.squeeze()[y_c, x_c]

        bb_size = max([w, h])
        bb_size = int(max([bb_size, crop_size]) * inflate)  # must be at least crop_size big
        bb_half = bb_size // 2
        # left, upper, right, and lower pixel coordinate
        left = x_c - bb_half
        upper = y_c - bb_half
        right = left + bb_size
        lower = upper + bb_size

        # assert that the box is in the image
        left, upper, side = fit_to_image(left, upper, bb_size, img_w, img_h)
        right = left + bb_size
        lower = upper + bb_size

        resize = bb_size < crop_size

        # crop IMAGE
        cropped = image.crop((left, upper, right, lower))
        if resize: cropped = cropped.resize((crop_size, crop_size), Image.BICUBIC)
        counter += 1
        # save
        fname = os.path.join(save_dir, '{}.png'.format(counter))
        cropped.save(fname)

        # get instance MASK and crop it
        mask = (instances == inst_id).squeeze().numpy().astype(np.uint8) * 255
        assert mask.any()
        mask = Image.fromarray(mask)
        mask_cropped = mask.crop((left, upper, right, lower))
        if resize: mask_cropped = mask_cropped.resize((crop_size, crop_size), Image.NEAREST)
        # save
        fname = os.path.join(save_dir, '{}.pbm'.format(counter))
        mask_cropped.save(fname)

    return counter


def transform_input(inp, transform):
    out = transform(inp) * 255.0
    return out


def fit_to_image(x, y, side, im_w, im_h):
    '''
    params:
        (x,y): top-left corner of the crop
        side: crop size of a square box in px
        im_w, im_h: image width and height
    output:
        modified values (x,y) such that the whole box is inside an image
    '''

    # move the top-left corner so that it is valid first
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    # coordinates of the bottom right corned
    x2, y2 = x + side, y + side
    xsize = abs(x - x2)
    ysize = abs(y - y2)

    max_x, max_y = im_w - 1, im_h - 1

    if x2 > max_x:
        # delta: how much do we need to shift the top-left x-coordinate
        delta_x = x2 - max_x
        x = max([0, x - delta_x])
        x2 = min([max_x, x2])
        xsize = abs(x - x2)
    if y2 > max_y:
        delta_y = y2 - max_y
        y = max([0, y - delta_y])
        y2 = min([max_y, y2])
        ysize = abs(y - y2)

    if (x < 0) or (y < 0):
        raise Exception(
            "x and y cannot be negative! (x={:d}, x2={}, dx={}, y={:d}, y2={:d}, dy={}; "
            "image: {}x{})".format(x, x2, delta_x, y, y2, delta_y, im_h, im_w))

    orig_side = side
    side = min([xsize, ysize])
    if orig_side != side:
        pass
        # print('Original size {}, current size: {}.'.format(orig_side, side), file=sys.stderr)

    # return modified values of the top-left corner
    return x, y, side


class Loader:
    def __init__(self, opt, path, split, im_w=2048, im_h=1024, new_base=None):
        self.new_base = new_base
        with open(path, 'rb') as f:
            self.dataset = pickle.load(f)[split]
        self.im_w, self.im_h = im_w, im_h
        self.opt = opt
        self.pedestrian_id = 24

    def normalize(self):
        return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def insert_pedestrians(self, original_labels, pedestrian_pixels):
        modified_labels = np.array(original_labels)
        modified_labels[np.where(pedestrian_pixels[:, :, 0])] = self.pedestrian_id
        modified_labels = Image.fromarray(modified_labels)
        return modified_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.new_base is not None:
            for key in data.keys():
                if isinstance(data[key], str):
                    data[key] = data[key].replace('/opt/datasets/cityscapes', self.new_base)
        data['image'] = None
        data['image_orig'] = np.array(Image.open(data['img']))
        data['to_insert'] = np.array(Image.open(data['new_objs']).resize((self.im_w, self.im_h), Image.NEAREST))
        if len(data['to_insert'].shape) < 3:
            data['to_insert'] = np.repeat(data['to_insert'][:, :, np.newaxis], 3, axis=2)

        if 'new_seg' not in data.keys():
            splitted = data['seg'].split(os.sep)
            splitted[4] += '_extended'
            data['new_seg'] = os.sep.join(splitted)

        # load and resize labels
        labels = Image.open(data['new_seg']).resize((self.im_w, self.im_h), Image.NEAREST)
        labels_orig = Image.open(data['seg'])
        labels_orig = self.insert_pedestrians(labels_orig, data['to_insert'])

        # DO NOT USE MODIFIED SEGMENTATION MAP!
        labels = labels_orig

        # transform labels
        params = get_params(self.opt, labels.size)
        if self.opt.label_nc == 0:
            transform = get_transform(self.opt, params)
            data['label'] = transform(labels.convert('RGB'))
        else:
            transform = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            data['label'] = transform(labels) * 255.0
            # labels_orig = transform(labels_orig) * 255.0
        data['label'] = data['label'].unsqueeze(0)
        # labels_orig = labels_orig.unsqueeze(0)

        # labels_color = util.tensor2label(data['label'][0], self.opt.label_nc)
        # labels_orig_color = util.tensor2label(labels_orig[0], self.opt.label_nc)

        # load and resize instances
        inst = Image.open(data['new_instances']).resize((self.im_w, self.im_h), Image.NEAREST)

        # transform instances
        data['inst'] = transform(inst).unsqueeze(0)

        if self.opt.load_features:
            raise NotImplementedError('Not yet implemented!')
            feat_path = self.feat_paths[index]
            feat = Image.open(feat_path).convert('RGB')
            norm = self.normalize()
            feat_tensor = norm(transform(feat))

        return data


if __name__ == '__main__':

    insert2orig = False
    new_base = '/mnt/nas/data/cityscapes/cityscapes'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bbs_epoch = 160
    orig_path_bb = '/home/tonda/projects/pedestrian_insertion/data/gtBbox_cityPersons_trainval'
    proposed_bbs = './data/proposed_bbs_ep{}.pkl'.format(bbs_epoch)
    save_path_extended_bb = '/mnt/nas/data/CVPR2020/pix2pixHD_CS/cityscapes_bbs_extended'
    save_path_extended_img = '/mnt/nas/data/CVPR2020/pix2pixHD_CS/cityscapes_imgs_extended'
    save_path_cropped_img = '/mnt/nas/data/CVPR2020/pix2pixHD_CS/cityscapes_imgs_cropped'
    crop_size = 128
    splits = ['val', 'train', 'test']

    if not os.path.exists(save_path_cropped_img):
        os.makedirs(save_path_cropped_img)

    opt = TestOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    # create visualizer
    visualizer = Visualizer(opt)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)
        # model = model.to(device)

        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx

    for split in splits:

        dataset = Loader(opt, proposed_bbs, split, new_base=new_base)
        save_path_cropped_img_split = os.path.join(save_path_cropped_img, split)
        if not os.path.exists(save_path_cropped_img_split):
            os.makedirs(save_path_cropped_img_split)
        counter = 0

        for i, data in enumerate(dataset):

            if opt.data_type == 16:
                data['label'] = data['label'].half()
                data['inst'] = data['inst'].half()
            elif opt.data_type == 8:
                data['label'] = data['label'].uint8()
                data['inst'] = data['inst'].uint8()
            if opt.export_onnx:
                print("Exporting to ONNX: ", opt.export_onnx)
                assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
                torch.onnx.export(model, [data['label'], data['inst']],
                                  opt.export_onnx, verbose=True)
                exit(0)

            # data['label'] = data['label'].to(device)
            # data['inst'] = data['inst'].to(device)
            # if data['image'] is not None:
            #    data['image'] = data['image'].to(device)

            minibatch = 1
            if opt.engine:
                generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
            elif opt.onnx:
                generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
            else:
                generated = model.inference(data['label'], data['inst'], data['image'])

            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0]))])

            img_path = data['img']

            merged = None
            base_dir, img_name = os.path.split(img_path)
            split, city = base_dir.split(os.sep)[-2:]
            fdir = os.path.join(save_path_extended_img, split, city)
            if insert2orig:
                # merge the original image with the generated one
                original_image = data['image_orig']
                generated_image = visuals['synthesized_image']
                labels = visuals['input_label']
                inserted_instances = data['to_insert']
                assert (inserted_instances.shape == original_image.shape) and (
                        inserted_instances.shape == generated_image.shape), "Shapes do not match! Proposed solution: upsample the conditional input with NEAREST NEIGHBOR."
                merged = insert_generated(original_image, generated_image, inserted_instances)
                # and save it
                if not os.path.exists(fdir):
                    os.makedirs(fdir)
                fname = os.path.join(fdir, img_name)
                merged.save(fname)

            if merged is None:
                merged = Image.fromarray(visuals['synthesized_image'])

            # continue
            # append new bounding boxes
            new_bbs = data['bbs_centered']
            img_name_noext = img_name.split('.')[0]
            orig_bbs_file = os.path.join(orig_path_bb, split, city, '{}.txt'.format(img_name_noext))
            new_file = os.path.join(save_path_extended_bb, split, city, '{}_extended.txt'.format(img_name_noext))
            extend_bbs(orig_bbs_file, new_bbs, new_file)

            # crop generated images centered at the generated pedestrian
            new_obj_mask = data['to_insert']
            instances = data['inst']
            counter = crop_and_save(merged, instances, new_bbs, crop_size, counter,
                                    save_path_cropped_img_split)

            if (i + 1) % 100 == 0:
                print('Split: {}, {}/{} completed'.format(split, i + 1, len(dataset)))

        print('Split "{}" finished.\n\n'.format(split))

    # webpage.save()
