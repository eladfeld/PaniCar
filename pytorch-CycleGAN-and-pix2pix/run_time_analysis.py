
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time



if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)  
    model = create_model(opt)      
    model.setup(opt)               



    start_t = time.time()
    for i, data in enumerate(dataset):

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        # img_path = model.get_image_paths()
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    end_t = time.time()
    print('total time:', end_t - start_t)
