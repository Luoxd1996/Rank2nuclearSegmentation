import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from fcn import fcn8s, fcn16s, fcn32s
from segmentation_dataset import SegmentationData


parser = argparse.ArgumentParser()
parser.add_argument('--data_list', type=str,
                    default='MICCAI2018/Train', help='Name of Experiment')
parser.add_argument('--aug', type=bool, default=True,
                    help='Data augmentation')
parser.add_argument('--exp', type=str,
                    default='naive_fcn16s_ssl', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=3000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
# parser.add_argument('--pretrained', type=str,  default='model/ssl_all_encoder/iter_2000.pth',
#                     help='Pretrained model to use')
parser.add_argument('--pretrained', type=str,  default='',
                    help='Pretrained model to use')
args = parser.parse_args()

snapshot_path = "model/" + args.data_list + "/" + args.exp

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / \
            (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = fcn16s(pretrained=False, n_class=1)
    net = net.cuda()
    if len(args.pretrained) > 0:
        net.load_state_dict(torch.load(args.pretrained), strict=False)
        print("Loaded pretrained_model!")
    print("=======> Loading dataset!")
    db_train = SegmentationData(data_list=args.data_list, patch_size=[
                                480, 480], aug=args.aug)
    print("=======> Loaded dataset!")
    print("The dataset length is : {}".format(db_train.__len__()))

    logging.info("All datasets length is : {}".format(db_train.__len__()))

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    net.train()
    criterion = BCEDiceLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=5e-4)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            inputs, label = sampled_batch
            inputs, label = inputs.cuda(), label.cuda()
            outputs = net(inputs)
            outputs = nn.functional.sigmoid(outputs)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                image = inputs[0, :, :, :]
                writer.add_image('train/Image', image, iter_num)

                predict_seg = outputs[0, 0:1, :, :]
                writer.add_image('train/Predicted_label',
                                 predict_seg, iter_num)

                gt_img = label[0, 0:1, :, :]
                writer.add_image('train/Groundtruth_label', gt_img, iter_num)

            # change lr
            if iter_num % 1000 == 0 and lr_ > 0.000001:
                lr_ = base_lr * 0.1 ** (iter_num // 1000)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(
        snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
