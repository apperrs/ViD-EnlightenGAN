import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader, CreatePredictDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html, util

opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 2
opt.serial_batches = True
opt.no_flip = True

data_loader, data_loader2 = CreatePredictDataLoader(opt)
dataset = data_loader.load_data()
dataset2 = data_loader2.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

print(len(dataset))

try:
    for i, data in enumerate(dataset):
        model.set_predict_input(data)
        visuals = model.predict()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

    for i, data in enumerate(dataset2):
        model.set_predict_input(data)
        visuals = model.predict()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)
except:
    pass