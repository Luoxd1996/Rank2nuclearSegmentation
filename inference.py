import os
import argparse
import torch
from unet_seg import UNet
from inference_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--test_root', type=str,
                    default='data/Tissue/Test2', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='Tissue/Train1/naive_unet_ssl', help='model_name')
parser.add_argument('--test_save_path', type=str,
                    default='data/Tissue/Test2', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "model/"+FLAGS.model+"/"
test_save_path = FLAGS.test_save_path
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

test_image_list = [] 
if "MICCAI2018" in FLAGS.test_root:
    cases_list = os.listdir(FLAGS.test_root + "/image")
    for case in cases_list:
        test_image_list.append([FLAGS.test_root + "/image/{}".format(
            case), FLAGS.test_root + "/label/{}".format(case)])

else:
    classes_list = os.listdir(FLAGS.test_root)
    for classes in classes_list:
        for i in range(1, 3):
            test_image_list.append([FLAGS.test_root + "/{}/Slide_0{}.png".format(
                classes, i), FLAGS.test_root + "/{}/GT_0{}.png".format(classes, i)])


def test_calculate_metric(epoch_num):
    net = UNet(3, 1, 64).cuda()
    save_mode_path = os.path.join(
        snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, test_image_list, patch_size=(
        400, 400), stride_hw=(100, 100), save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric(2000)