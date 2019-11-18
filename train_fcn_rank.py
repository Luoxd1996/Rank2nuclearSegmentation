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

from vgg_rank import VGGNet, RankNet
from ranking_dataset import RankData


parser = argparse.ArgumentParser()
parser.add_argument('--data_list', type=str,
                    default="MICCAI2018/Train", help='Name of Experiment')
parser.add_argument('--aug', type=bool, default=False,
                    help='image augmentation')
parser.add_argument('--exp', type=str,
                    default='ssl_vgg', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.000001,
                    help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

snapshot_path = "model/" + args.exp + "/"

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


class RankingLoss(torch.nn.Module):
    def __init__(self, margin=0.0, mini_batch_size=5, image_sets=4):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.mini_batch_size = mini_batch_size
        self.image_sets = image_sets

    def _mini_batch_hinge_loss(self, prediction, label):
        mini_batch_loss = torch.zeros(1).cuda().float()
        for x, rank1 in enumerate(label):
            for rank2 in label[x+1:]:
                mini_batch_loss += torch.max(torch.zeros(1).cuda(), -(
                    rank1 - rank2).float() * (prediction[rank1] - prediction[rank2]))
        return mini_batch_loss

    def forward(self, inputs, label):
        loss = 0.0
        for rounds in range(self.image_sets):
            round_loss = 0.0
            round_loss += self._mini_batch_hinge_loss(
                inputs[rounds], label[rounds])
            loss += round_loss
        return loss / self.image_sets


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = VGGNet()
    net = net.cuda()
    ranknet = RankNet()
    ranknet = ranknet.cuda()

    print("=======> Loading dataset!")
    db_train = RankData(data_list=args.data_list, aug=args.aug, patch_size=[580, 580])
    print("=======> Loaded dataset!")
    print("The dataset length is : {}".format(db_train.__len__()))

    logging.info("All datasets length is : {}".format(db_train.__len__()))

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    net.train()
    ranknet.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=5e-4)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    criterion = RankingLoss(margin=0.0, mini_batch_size=5,
                            image_sets=batch_size).cuda()
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            batch_outputs = []
            batch_labels = []
            for ind in range(batch_size):
                mini_batch_inputs, mini_batch_labels = sampled_batch[0][ind], sampled_batch[1][ind]
                mini_batch_outputs = []

                for inp_index in range(mini_batch_inputs.size(0)):
                    inputs = mini_batch_inputs[inp_index, :, :, :].unsqueeze(
                        0).cuda()
                    outputs = net(inputs)['x5']
                    mini_batch_outputs.append(outputs)
                cat_feature = torch.cat(mini_batch_outputs, dim=0)
                rank_outputs = ranknet(cat_feature)
                batch_outputs.append(rank_outputs)
                mini_batch_labels = mini_batch_labels.cuda()
                batch_labels.append(mini_batch_labels)
            loss = criterion(batch_outputs, batch_labels)
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

                rg_img = outputs.mean(dim=1)
                writer.add_image('train/rankding_feature', rg_img, iter_num)

            # change lr
            # if iter_num % 3000 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 3000)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_
            if iter_num % 100 == 0:
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
