from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time

retina_config_file = 'configs/retinanet/retinanet_x101_64x4d_fpn_mstrain_640-800_3x_coco.py'
retina_checkpoint_file = 'checkpoints/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth'

mrcnn_config_file = 'configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py'
mrcnn_checkpoint_file = 'checkpoints/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth'

yolo_config_file = 'configs/yolo/yolov3_d53_320_273e_coco.py'
yolo_checkpoint_file = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'

ssd_config_file = 'configs/ssd/ssd512_coco.py'
ssd_checkpoint_file = 'checkpoints/ssd512_coco_20210803_022849-0a47a1ca.pth'

frcnn_config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
frcnn_checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# berkley trained
yolo_berkley_config_file = '/sise/home/eladfeld/train_mmdet/configs/berkley/yolov3_d53_mstrain-608_273e_berkley.py'
yolo_berkley_checkpoint_file = '/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley/latest.pth'

yolo_pistol_config_file = '/sise/home/eladfeld/train_mmdet/configs/pistol/yolov3_d53_mstrain-608_273e_pistol.py'
yolo_pistol_checkpoint_file = '/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_pistol/latest.pth'


ssd_pistol_config_file = '/sise/home/eladfeld/train_mmdet/configs/pistol/ssdlite_mobilenetv2_scratch_600e_pistol.py'
ssd_pistol_checkpoint_file = '/sise/home/eladfeld/train_mmdet/work_dirs/ssdlite_mobilenetv2_scratch_600e_pistol/latest.pth'

faster_rcnn_pistol_config_file = "/sise/home/eladfeld/train_mmdet/work_dirs/faster_rcnn_r50_caffe_fpn_mstrain_3x_pistol/faster_rcnn_r50_caffe_fpn_mstrain_3x_pistol.py"
faster_rcnn_pistol_checkpoint_file = "/sise/home/eladfeld/train_mmdet/work_dirs/faster_rcnn_r50_caffe_fpn_mstrain_3x_pistol/latest.pth"

retinanet_pistol_config_file = "/sise/home/eladfeld/train_mmdet/configs/pistol/retinanet_r50_caffe_fpn_mstrain_3x_pistol.py"
reintnet_pistol_checkpoint_file = "/sise/home/eladfeld/train_mmdet/work_dirs/retinanet_r50_caffe_fpn_mstrain_3x_pistol/latest.pth"

yolo_x_pistol_config_file = "/sise/home/eladfeld/train_mmdet/configs/pistol/yolox_l_8x8_300e_pistol.py"
yolo_x_pistol_checkpoint_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolox_l_8x8_300e_pistol/latest.pth"

yolo_aug_config_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley_aug/yolov3_d53_mstrain-608_273e_berkley_aug.py"
yolo_aug_checkpoint_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley_aug/latest.pth"

ssd_aug_config_file = "/sise/home/eladfeld/train_mmdet/work_dirs/ssdlite_mobilenetv2_scratch_600e_berkley_aug/ssdlite_mobilenetv2_scratch_600e_berkley_aug.py"
ssd_aug_checkpoint_file = "/sise/home/eladfeld/train_mmdet/work_dirs/ssdlite_mobilenetv2_scratch_600e_berkley_aug/latest.pth"

yolo_cyclegan_config_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley_cyclegan/yolov3_d53_mstrain-608_273e_berkley_cyclegan.py"
yolo_cyclegan_checkpoint_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley_cyclegan/latest.pth"

yolo_cyclegan_small_config_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley_cyclegan_small/yolov3_d53_mstrain-608_273e_berkley_cyclegan_small.py"
yolo_cyclegan_small_checkpoint_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley_cyclegan_small/latest.pth"

yolo_cyclegan_big_config_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley_cyclegan_big/yolov3_d53_mstrain-608_273e_berkley_cyclegan_big.py"
yolo_cyclegan_big_checkpoint_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley_cyclegan_big/latest.pth"


yolo_cyclegan_super_config_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley_cyclegan_super/yolov3_d53_mstrain-608_273e_berkley_cyclegan_super.py"
yolo_cyclegan_super_checkpoint_file = "/sise/home/eladfeld/train_mmdet/work_dirs/yolov3_d53_mstrain-608_273e_berkley_cyclegan_super/latest.pth"

new_mrcnn_checkpoint = "/sise/home/eladfeld/eladfeld/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
new_mecnn_config = "/sise/home/eladfeld/eladfeld/configs/mask_rcnn/config.py"

new_frcnn_checkpoint = "/sise/home/eladfeld/eladfeld/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
new_frcnn_config = "/sise/home/eladfeld/eladfeld/configs/faster_rcnn/faster_rcnn_config.py"

new_yolo_checkpoint = "/sise/home/eladfeld/eladfeld/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
new_yolo_config = "/sise/home/eladfeld/eladfeld/configs/yolox/yolox_x_config.py"

new_retina_checkpoint = "/sise/home/eladfeld/eladfeld/checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"
new_retina_config = "/sise/home/eladfeld/eladfeld/configs/retinanet/retinanet_r101_fpn_1x_coco.py"


ssd_random_checkpoint = "/dt/shabtaia/dt-fujitsu-robustness/EpileptiCar/work_dirs/ssd_random/epoch_100.pth"
ssd_random_config = "/dt/shabtaia/dt-fujitsu-robustness/EpileptiCar/work_dirs/ssd_random/ssd_random.py"

ssd_cyclegan_checkpoint = "/dt/shabtaia/dt-fujitsu-robustness/EpileptiCar/work_dirs/ssd_cyclegan/epoch_100.pth"
ssd_cyclegan_config = "/dt/shabtaia/dt-fujitsu-robustness/EpileptiCar/work_dirs/ssd_cyclegan/ssd_cyclegan.py"

ssd_berkley_checkpoint = "/dt/shabtaia/dt-fujitsu-robustness/EpileptiCar/work_dirs/ssd_berkley/epoch_100.pth"
ssd_berkley_config = "/dt/shabtaia/dt-fujitsu-robustness/EpileptiCar/work_dirs/ssd_berkley/ssd_berkley.py"

def build_model(config_path, checkpoint_path):
    model = init_detector(config_path, checkpoint_path, device='cuda:0')
    return model


def print_results(model, result, classes, preds, is_tupple=False):
    tmp_scr = []
    if is_tupple:
        result = result[0]
    for clas in classes:
        try:
            index = model.CLASSES.index(clas)
            # rcnn result is a tuple, take only the first element

            if len(result[index]) > 0:
                # takes only the first row which is the maximum of all prediction for this class
                score_i = result[index][0][4]
                tmp_scr.append(score_i)
            else:
                tmp_scr.append(0)
                print('object not detected')
        except ValueError:
            print(f'object "{clas}" is not in the dataset, maybe a typo')
    preds.append(np.max(tmp_scr))


def predict(path, model, preds, outpath, is_tupple=False, classes=['car', 'bus', 'truck'], show_result=False):
    result = inference_detector(model, path)
    print_results(model, result, classes, preds, is_tupple=is_tupple)
    if show_result:
        model.show_result(path, result, out_file=outpath)


def main():
    ssd_model = build_model(ssd_config_file, ssd_checkpoint_file)
    yolo_model = build_model(yolo_config_file, yolo_checkpoint_file)
    frcnn_model = build_model(frcnn_config_file, frcnn_checkpoint_file)
    maskrcnn_model = build_model(mrcnn_config_file, mrcnn_checkpoint_file)
    retina_model = build_model(retina_config_file, retina_checkpoint_file)

    ssd_preds = []
    yolo_preds = []
    frcnn_preds = []
    maskrcnn_preds = []
    retina_preds = []

    for i in range(2):
        path = f'trans/frame{i}.jpeg'
        predict(path, ssd_model, ssd_preds, ('ssd_' + path))
        predict(path, yolo_model, yolo_preds, ('yolo_' + path))
        predict(path, frcnn_model, frcnn_preds, ('frcnn_' + path))
        predict(path, maskrcnn_model, maskrcnn_preds, ('mrcnn_' + path), is_tupple=True)
        predict(path, retina_model, retina_preds, ('retina_' + path))
        print(i)
    with open(f'trans_pred', 'wb') as f:
        pickle.dump((ssd_preds, yolo_preds, frcnn_preds, maskrcnn_preds, retina_preds), f)


def main2(name, num_of_folders):
    ssd_model = build_model(ssd_config_file, ssd_checkpoint_file)
    yolo_model = build_model(yolo_config_file, yolo_checkpoint_file)
    frcnn_model = build_model(frcnn_config_file, frcnn_checkpoint_file)
    maskrcnn_model = build_model(mrcnn_config_file, mrcnn_checkpoint_file)
    retina_model = build_model(retina_config_file, retina_checkpoint_file)

    DIR_ = f'/sise/home/eladfeld/eladfeld/{name}/'
    for i in range(num_of_folders):
        DIR = DIR_ + str(i)
        files_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        ssd_preds = []
        yolo_preds = []
        frcnn_preds = []
        maskrcnn_preds = []
        retina_preds = []

        print(f'video number: {i}')
        for j in range(files_count):
            path = f'{DIR}/frame{j}.jpg'
            predict(path, ssd_model, ssd_preds, ('ssd_' + path))
            predict(path, yolo_model, yolo_preds, ('yolo_' + path))
            predict(path, frcnn_model, frcnn_preds, ('frcnn_' + path))
            predict(path, maskrcnn_model, maskrcnn_preds, ('mrcnn_' + path), is_tupple=True)
            predict(path, retina_model, retina_preds, ('retina_' + path))
            if j % 10 == 0:
                print(j)
        with open(f'/sise/home/eladfeld/eladfeld/{name}/{i}_pred', 'wb') as f:
            pickle.dump((ssd_preds, yolo_preds, frcnn_preds, maskrcnn_preds, retina_preds), f)


def main3(name, num_of_folders):
    yolo_model = build_model(yolo_config_file, yolo_checkpoint_file)
    berkley_model = build_model(yolo_berkley_config_file, yolo_berkley_checkpoint_file)

    DIR_ = f'/sise/home/eladfeld/eladfeld/{name}/'
    for i in range(num_of_folders):
        DIR = DIR_ + str(i)
        files_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        yolo_preds = []
        berkley_preds = []

        print(f'video number: {i}')
        for j in range(files_count):
            path = f'{DIR}/frame{j}.jpg'
            predict(path, yolo_model, yolo_preds, ('yolo_' + path))
            predict(path, berkley_model, berkley_preds, ('berkley_' + path))

            if j % 10 == 0:
                print(j)
            with open(f'{i}_{name}_pred', 'wb') as f:
                pickle.dump((yolo_preds, berkley_preds), f)


def main4():
    # yolo3_pistol_model = build_model(yolo_pistol_config_file, yolo_pistol_checkpoint_file)
    ssd_pistol_model = build_model(ssd_pistol_config_file, ssd_pistol_checkpoint_file)
    # retina_pistol_model = build_model(retina_config_file, retina_checkpoint_file)
    faster_rcnn_pistol_model = build_model(faster_rcnn_pistol_config_file, faster_rcnn_pistol_checkpoint_file)
    yolo_x_pistol_model = build_model(yolo_x_pistol_config_file, yolo_x_pistol_checkpoint_file)

    DIR = "/sise/home/eladfeld/eladfeld/all_test/"

    yolo3_preds = []
    yolox_preds = []
    ssd_preds = []
    retina_preds = []
    faster_rcnn_preds = []

    for name in [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]:
        if name != 'test_coco.json':
            path = f'{DIR}/{name}'
            # predict(path, yolo3_pistol_model, yolo3_preds, ('yolo3_pistol' + path), classes=['pistol'])
            predict(path, yolo_x_pistol_model, yolox_preds, ('yoloX_pistol' + path), classes=['pistol'])
            predict(path, ssd_pistol_model, ssd_preds, ('ssd_pistol' + path), classes=['pistol'])
            # predict(path, retina_pistol_model, retina_preds, ('reinta_pistol_new' + path), classes=['pistol'])
            predict(path, faster_rcnn_pistol_model, faster_rcnn_preds, ('faster_rcnn_pistol' + path), classes=['pistol'])

        with open(f'pistol_pred', 'wb') as f:
            preds = {'yoloX':yolox_preds, 'ssd':ssd_preds, 'faster_rcnn':faster_rcnn_preds}
            pickle.dump((preds), f)


def main5():
    ssd_pistol_model = build_model(ssd_pistol_config_file, ssd_pistol_checkpoint_file)
    faster_rcnn_pistol_model = build_model(faster_rcnn_pistol_config_file, faster_rcnn_pistol_checkpoint_file)
    yolo_x_pistol_model = build_model(yolo_x_pistol_config_file, yolo_x_pistol_checkpoint_file)

    DIR = "/sise/home/eladfeld/eladfeld/all_test/"

    yolox_preds = []
    ssd_preds = []
    faster_rcnn_preds = []

    for i in range(len(os.listdir(DIR))):
        path = f'{DIR}/im_{i + 1}.jpg'
        predict(path, yolo_x_pistol_model, yolox_preds, ('yoloX_pistol' + path), classes=['pistol'])
        predict(path, ssd_pistol_model, ssd_preds, ('ssd_pistol' + path), classes=['pistol'])
        predict(path, faster_rcnn_pistol_model, faster_rcnn_preds, ('faster_rcnn_pistol' + path), classes=['pistol'])

    with open(f'pistol_pred', 'wb') as f:
        preds = {'yoloX':yolox_preds, 'ssd':ssd_preds, 'faster_rcnn':faster_rcnn_preds}
        pickle.dump((preds), f)

def main6(name, num_of_folders):
    yolo_model = build_model(yolo_config_file, yolo_checkpoint_file)
    berkley_model = build_model(yolo_berkley_config_file, yolo_berkley_checkpoint_file)
    berkley_aug_model = build_model(yolo_aug_config_file, yolo_checkpoint_file)

    DIR_ = f'/sise/home/eladfeld/eladfeld/{name}/'
    for i in range(num_of_folders):
        DIR = DIR_ + str(i)
        files_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        yolo_preds = []
        berkley_preds = []
        berkley_aug_preds = []

        print(f'video number: {i}')
        for j in range(files_count):
            path = f'{DIR}/frame{j}_fake.png'
            predict(path, yolo_model, yolo_preds, ('yolo_' + path))
            predict(path, berkley_model, berkley_preds, ('berkley_' + path))
            predict(path, berkley_aug_model, berkley_aug_preds, ('bekley_aug_' + path))

            if j % 10 == 0:
                print(j)
            with open(f'denoise_results/denoise_{i}_{name}_pred', 'wb') as f:
                pickle.dump((yolo_preds, berkley_preds, berkley_aug_preds), f)

def main7(name, num_of_folders):
    # cyclegan_model = build_model(yolo_cyclegan_config_file, yolo_cyclegan_checkpoint_file)
    # cyclegan_big_model = build_model(yolo_cyclegan_big_config_file, yolo_cyclegan_big_checkpoint_file)
    # cyclegan_small_model = build_model(yolo_cyclegan_small_config_file, yolo_cyclegan_small_checkpoint_file)
    cyclegan_super_model = build_model(yolo_cyclegan_super_config_file, yolo_cyclegan_super_checkpoint_file)

    DIR_ = f'/sise/home/eladfeld/eladfeld/{name}/'
    for i in range(num_of_folders):
        DIR = DIR_ + str(i)

        files_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        # cyclegan_big_preds = []
        # cyclegan_small_preds = []
        cyclegan_super_preds = []   
        print(f'video number: {i}')
        for j in range(files_count):
            path = f'{DIR}/frame{j}_fake.png'
            # predict(path, cyclegan_big_model, cyclegan_big_preds, ('yolo_' + path))
            # predict(path, cyclegan_small_model, cyclegan_small_preds, ('yolo_' + path))
            predict(path, cyclegan_super_model, cyclegan_super_preds, ('yolo_' + path))


            if j % 10 == 0:
                print(j)
            with open(f'super_denoising/super_denoise_aug_{i}_{name}_pred', 'wb') as f:
                # pickle.dump((cyclegan_big_preds, cyclegan_small_preds), f)
                pickle.dump((cyclegan_super_preds), f)


def main8(name, num_of_folders):
    yolo_model = build_model(yolo_config_file, yolo_checkpoint_file)
    berkley_model = build_model(yolo_berkley_config_file, yolo_berkley_checkpoint_file)
    berkley_aug_model = build_model(yolo_aug_config_file, yolo_checkpoint_file)

    DIR_ = f'/dt/shabtaia/dt-fujitsu-robustness/epilepticar/denoise_3/'
    for i in range(num_of_folders):
        DIR = DIR_ + str(i)
        files_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        yolo_preds = []
        berkley_preds = []
        berkley_aug_preds = []

        print(f'video number: {i}')
        for j in range(files_count):
            path = f'{DIR}/frame{j}_fake.png'
            predict(path, yolo_model, yolo_preds, ('yolo_' + path))
            predict(path, berkley_model, berkley_preds, ('berkley_' + path))
            predict(path, berkley_aug_model, berkley_aug_preds, ('bekley_aug_' + path))

            if j % 10 == 0:
                print(j)
        with open(f'denoise3_results_new/denoise_fake_3_{i}_{name}_pred', 'wb') as f:
            pickle.dump((yolo_preds, berkley_preds, berkley_aug_preds), f)

def main9(name, num_of_folders):
    model_name = '/dt/shabtaia/dt-fujitsu-robustness/epilepticar/'
    yolo_model_50 = build_model('/dt/shabtaia/dt-fujitsu-robustness/epilepticar/yolov3_d53_mstrain-608_273e_berkley_h_h.py', model_name + 'epoch_50.pth')
    yolo_model_100 = build_model('/dt/shabtaia/dt-fujitsu-robustness/epilepticar/yolov3_d53_mstrain-608_273e_berkley_h_h.py', model_name + 'epoch_100.pth')
    yolo_model_150 = build_model('/dt/shabtaia/dt-fujitsu-robustness/epilepticar/yolov3_d53_mstrain-608_273e_berkley_h_h.py', model_name + 'epoch_150.pth')
    yolo_model_200 = build_model('/dt/shabtaia/dt-fujitsu-robustness/epilepticar/yolov3_d53_mstrain-608_273e_berkley_h_h.py', model_name + 'epoch_200.pth')
    
    DIR_ = f'/sise/home/eladfeld/eladfeld/{name}/'
    for i in range(num_of_folders):
        DIR = DIR_ + str(i)
        files_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        yolo_preds_50 = []
        yolo_preds_100 = []
        yolo_preds_150 = []
        yolo_preds_200 = []
        
        print(f'video number: {i}')
        for j in range(files_count):
            path = f'{DIR}/frame{j}.jpg'
            predict(path, yolo_model_50, yolo_preds_50, ('yolo_' + path))
            predict(path, yolo_model_100, yolo_preds_100, ('yolo_' + path))
            predict(path, yolo_model_150, yolo_preds_150, ('yolo_' + path))
            predict(path, yolo_model_200, yolo_preds_200, ('yolo_' + path))
            if j % 10 == 0:
                print(j)
            with open(f'h_h_results/h_h{i}_{name}_pred', 'wb') as f:
                pickle.dump((yolo_preds_50, yolo_preds_100, yolo_preds_150, yolo_preds_200), f)



def main10(name, num_of_folders):
    # mrcnn_model = build_model(new_mecnn_config, new_mrcnn_checkpoint)
    # frcnn_model = build_model(new_frcnn_config, new_frcnn_checkpoint)
    # yolox_model = build_model(new_yolo_config, new_frcnn_checkpoint)
    retina_model = build_model(new_retina_config, new_retina_checkpoint)

    DIR_ = f'/sise/home/eladfeld/eladfeld/{name}/'
    for i in range(num_of_folders):
        DIR = DIR_ + str(i)
        files_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        mrcnn_preds = []


        print(f'video number: {i}')
        for j in range(files_count):
            path = f'{DIR}/frame{j}.jpg'
            #predict(path, frcnn_model, mrcnn_preds, ('yolo_' + path), is_tupple=True)
            predict(path, retina_model, mrcnn_preds, ('yolo_' + path))

            if j % 10 == 0:
                print(j)
            with open(f'retina/{i}_{name}_pred', 'wb') as f:
                pickle.dump((mrcnn_preds), f)



def main20():
    ssd_model = build_model(ssd_config_file, ssd_checkpoint_file)
    yolo_model = build_model(yolo_config_file, yolo_checkpoint_file)
    frcnn_model = build_model(frcnn_config_file, frcnn_checkpoint_file)
    maskrcnn_model = build_model(mrcnn_config_file, mrcnn_checkpoint_file)
    retina_model = build_model(retina_config_file, retina_checkpoint_file)

    ssd_preds = []
    yolo_preds = []
    frcnn_preds = []
    maskrcnn_preds = []
    retina_preds = []
    dir_preds = 'tesla_preds'
    for i in range(223):
        image_name = f'frame{i}.jpg'
        path = os.path.join('tesla_frames', image_name)
        ssd_result = inference_detector(ssd_model, path)
        ssd_preds.append(ssd_result)
        out = os.path.join(dir_preds, 'ssd')
        ssd_model.show_result(path, ssd_result, out_file= os.path.join(out, image_name))
        yolo_result = inference_detector(yolo_model, path)
        yolo_preds.append(ssd_result)
        out = os.path.join(dir_preds, 'yolo')
        yolo_model.show_result(path, yolo_result, out_file= os.path.join(out, image_name))
        frcnn_result = inference_detector(frcnn_model, path)
        frcnn_preds.append(frcnn_result)
        out = os.path.join(dir_preds, 'frcnn')
        frcnn_model.show_result(path, frcnn_result, out_file= os.path.join(out, image_name))
        mrcnn_result = inference_detector(maskrcnn_model, path)
        maskrcnn_preds.append(mrcnn_result)
        out = os.path.join(dir_preds, 'mrcnn')
        maskrcnn_model.show_result(path, mrcnn_result, out_file= os.path.join(out, image_name))
        retina_result= inference_detector(retina_model, path)
        retina_preds.append(retina_preds)
        out = os.path.join(dir_preds, 'retina')
        retina_model.show_result(path, retina_result, out_file= os.path.join(out, image_name))

        print(i)
    with open(f'tesla_result', 'wb') as f:
        pickle.dump((ssd_preds, yolo_preds, frcnn_preds, maskrcnn_preds, retina_preds), f)


def create_result_file():
    yolo_berkley = build_model(yolo_berkley_config_file, yolo_berkley_checkpoint_file)
    yolo_aug = build_model(yolo_aug_config_file, yolo_aug_checkpoint_file)
    yolo_cyclegan = build_model(yolo_cyclegan_config_file, yolo_cyclegan_checkpoint_file)
    yolo = build_model(yolo_config_file, yolo_checkpoint_file)

    # DIR_ = '/dt/shabtaia/dt-fujitsu-robustness/EpileptiCar/coco/val2017'
    DIR_ = f'/dt/shabtaia/dt-fujitsu-robustness/EpileptiCar/berkley/val/'
    # DIR_ = '/dt/shabtaia/dt-fujitsu-robustness/EpileptiCar/test_dataset/coco/'
    results = []
    i = 0
    for name in os.listdir(DIR_):
        path = os.path.join(DIR_, name)
        # result = inference_detector(yolo_berkley, path)
        # result = inference_detector(yolo_berkley, path)
        result = inference_detector(yolo_aug, path)
        # result = inference_detector(yolo_cyclegan, path)
        # yolo_berkley.show_result(path, result, out_file='results_file/berkley_img/berkley_' + name)
        results.append((name, result))
        i += 1
        # if i > 5000:
        #     break
        if i % 100 == 0:
            print(i)
    with open(f'results_file/new_berkley_val_yolo_aug_result', 'wb') as f:
        pickle.dump((results), f)   




def with_one():
    yolo = build_model(yolo_config_file, yolo_checkpoint_file)
    yolo_berkley = build_model(yolo_berkley_config_file, yolo_berkley_checkpoint_file)
    results = []

    # DIR_ = '/dt/shabtaia/dt-fujitsu-robustness/coco/val2017'
    DIR_ = '/dt/shabtaia/dt-fujitsu-robustness/berkley/val/'
    
    with open('with_one_berkley', 'rb') as f:
        with_one = pickle.load(f)
    for image in with_one:
        path = os.path.join(DIR_, image['file_name'])
        results_yolo = inference_detector(yolo, path)
        results_berkley = inference_detector(yolo_berkley, path)
        yolo.show_result(path, results_yolo, out_file='results_file/yolo_img/yolo_' + image['file_name'])

        results.append((image['file_name'], results_yolo, results_berkley))
    with open(f'results_file/compare_berkley', 'wb') as f:
        pickle.dump((results), f)


def time_eval():
    yolo = build_model(yolo_config_file, yolo_checkpoint_file)
    DIR_ = '/dt/shabtaia/dt-fujitsu-robustness/time_eval/'
    
    images = os.listdir(DIR_)

    start_t = time.time()
    for image in images:
        path = os.path.join(DIR_, image)
        results_yolo = inference_detector(yolo, path)

    end_t = time.time()

    with open('time_3090.txt', 'w') as f:
      f.write(str((end_t - start_t) / len(images)))
    print((end_t - start_t) / len(images))



def ssd_main8(name, num_of_folders):
    ssd_berkley = build_model(ssd_berkley_config, ssd_berkley_checkpoint)
    ssd_cyclegan = build_model(ssd_cyclegan_config, ssd_berkley_checkpoint)
    ssd_random = build_model(ssd_random_config, ssd_random_checkpoint)

    DIR_ = f'/dt/shabtaia/dt-fujitsu-robustness/epilepticar/denoise_3/'
    for i in range(num_of_folders):
        DIR = DIR_ + str(i)
        files_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        ssd_berkley_preds = []
        ssd_cyclegan_preds = []
        ssd_random_preds = []

        print(f'video number: {i}')
        for j in range(files_count):
            path = f'{DIR}/frame{j}_fake.png'
            predict(path, ssd_berkley, ssd_berkley_preds)
            predict(path, ssd_cyclegan, ssd_cyclegan_preds)
            predict(path, ssd_random, ssd_random_preds)

            if j % 10 == 0:
                print(j)
        with open(f'denoise3_results_new/denoise_fake_3_{i}_{name}_pred', 'wb') as f:
            pickle.dump((yolo_preds, berkley_preds, berkley_aug_preds), f)


def ssd_main6(name, num_of_folders):
    ssd_berkley = build_model(ssd_berkley_config, ssd_berkley_checkpoint)
    ssd_cyclegan = build_model(ssd_cyclegan_config, ssd_berkley_checkpoint)
    ssd_random = build_model(ssd_random_config, ssd_random_checkpoint)

    DIR_ = f'/dt/shabtaia/dt-fujitsu-robustness/EpileptiCar/videos_eval/'
    for i in range(num_of_folders):
        DIR = DIR_ + str(i)
        files_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        ssd_berkley_preds = []
        ssd_cyclegan_preds = []
        ssd_random_preds = []

        print(f'video number: {i}')
        for j in range(files_count):
            path = f'{DIR}/frame{j}.jpg'
            predict(path, ssd_berkley, ssd_berkley_preds)
            predict(path, ssd_cyclegan, ssd_cyclegan_preds)
            predict(path, ssd_random, ssd_random_preds)

            if j % 10 == 0:
                print(j)
            with open(f'ssd_new/{i}_{name}_pred', 'wb') as f:
                pickle.dump((ssd_berkley_preds, ssd_cyclegan_preds, ssd_random_preds), f)

if __name__ == "__main__":
    # for folder in os.listdir('21_9_23'):
    #     video_folder = os.path.join('21_9_23', folder)
    #     video_num = len(os.listdir(video_folder))
    #     main2(video_folder, video_num)
    ssd_main6('asd', 243)




