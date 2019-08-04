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
import numpy as np
from shutil import copyfile


def insert_generated(original, generated, to_insert):
    merged = np.zeros_like(original)
    to_keep = not to_insert
    merged[to_keep] = original
    merged[to_insert] = generated
    return merged


def extend_bbs(original_txt_file, new_bbs, new_file):
    # copy the content of the original file
    copyfile(original_txt_file, new_file)

    # and append new bbs
    with open(new_file, 'a+') as f:
        n_bbs = len(new_bbs)
        for i, new_bb in enumerate(new_bbs):
            assert all(new_bb <= 1)
            x, y, w, h = new_bb
            line = '0 {} {} {} {}'.format(x, y, w, h)
            if i < (n_bbs - 1):
                line += '\n'
            f.write(line)


if __name__ == '__main__':

    orig_path_bb = './data/cityscapes_bbs_orig'
    save_path_extended_bb = './data/cityscapes_bbs_extended'
    save_path_extended_img = './data/cityscapes_imgs_extended'

    opt = TestOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
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

        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
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
        minibatch = 1
        if opt.engine:
            generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
        elif opt.onnx:
            generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
        else:
            generated = model.inference(data['label'], data['inst'], data['image'])

        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                               ('synthesized_image', util.tensor2im(generated.data[0]))])
        img_path = data['path']
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

        # merge the original image with the generated one
        original_image = data['image']
        generated_image = visuals['synthesized_image']
        labels = visuals['input_label']
        inserted_instances = data['to_insert']
        assert (inserted_instances.shape == original_image.shape) and (
            inserted_instances.shape == generated_image.shape), "Shapes do not match!"
        merged = insert_generated(original_image, generated_image, inserted_instances)
        # and save it
        base_dir, img_name = os.path.split(img_path)
        img_name = img_name.split('.')[0]
        split, city = base_dir.split(os.sep)[-2:]
        fname = os.path.join(save_path_extended_img, split, city, '{}_extended.png'.format(img_name))
        np.save(fname, merged)

        # append new bounding boxes
        new_bbs = data['new_bb']
        orig_bbs_file = os.path.join(orig_path_bb, split, city, '{}.txt'.format(img_name))
        new_file = os.path.join(save_path_extended_bb, split, city, '{}_extended.txt'.format(img_name))
        extend_bbs(orig_bbs_file, new_bbs, new_file)

    webpage.save()
