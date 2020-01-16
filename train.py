import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils import *
from options import get_args
from dataloader import nyudv2_dataloader
from models.loss import  cal_spatial_loss, cal_temporal_loss
from models.backbone_dict import backbone_dict
from models import modules
from models import net

cudnn.benchmark = True
args = get_args('train')

os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

# Create folder
makedir(args.checkpoint_dir)
makedir(args.logdir)

# creat summary logger
logger = SummaryWriter(args.logdir)

# dataset, dataloader
TrainImgLoader = nyudv2_dataloader.getTrainingData_NYUDV2(args.batch_size, args.trainlist_path, args.root_path)
# model, optimizer
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'

backbone = backbone_dict[args.backbone]()
Encoder = modules.E_resnet(backbone)

if args.backbone in ['resnet50']:
    model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048], refinenet=args.refinenet)
elif args.backbone in ['resnet18', 'resnet34']:
    model = net.model(Encoder, num_features=512, block_channel=[64, 128, 256, 512], refinenet=args.refinenet)

model = nn.DataParallel(model).cuda()

disc = net.C_C3D_1().cuda()

optimizer = build_optimizer(model = model,
			    learning_rate=args.lr,
			    optimizer_name=args.optimizer_name,
			    weight_decay = args.weight_decay,
			    epsilon=args.epsilon,
			    momentum=args.momentum
			    )

start_epoch = 0

if args.resume:
	all_saved_ckpts = [ckpt for ckpt in os.listdir(args.checkpoint_dir) if ckpt.endswith(".pth.tar")]
	print(all_saved_ckpts)
	all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x:int(x.split('_')[-1].split('.')[0]))
	loadckpt = os.path.join(args.checkpoint_dir, all_saved_ckpts[-1])
	start_epoch = int(all_saved_ckpts[-1].split('_')[-1].split('.')[0])
	print("loading the lastest model in checkpoint_dir: {}".format(loadckpt))
	state_dict = torch.load(loadckpt)
	model.load_state_dict(state_dict)
elif args.loadckpt is not None:
	print("loading model {}".format(args.loadckpt))
	start_epoch = args.loadckpt.split('_')[-1].split('.')[0]
	state_dict = torch.load(args.loadckpt)
	model.load_state_dict(state_dict)
else:
	print("start at epoch {}".format(start_epoch))

def train():
	for epoch in range(start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch, args.lr)
		batch_time = AverageMeter()
		losses = AverageMeter()
		model.train()
		end = time.time()
		for batch_idx, sample in enumerate(TrainImgLoader):

			image, depth = sample[0], sample[1]#(b,c,d,w,h)    

			depth = depth.cuda()
			image = image.cuda()
			image = torch.autograd.Variable(image)
			depth = torch.autograd.Variable(depth)			
			optimizer.zero_grad()
			global_step = len(TrainImgLoader) * epoch + batch_idx
			gt_depth = depth
			pred_depth = model(image)#(b, c, d, h, w)

			# Calculate the total loss 
			spatial_losses=[]
			for seq_idx in range(image.size(2)):
				spatial_loss = cal_spatial_loss(pred_depth[:,:,seq_idx,:,:], gt_depth[:,:,seq_idx,:,:])
				spatial_losses.append(spatial_loss)
			spatial_loss = sum(spatial_losses)

			pred_cls = disc(pred_depth)
			gt_cls = disc(gt_depth)
			temporal_loss = cal_temporal_loss(pred_cls, gt_cls)

			loss = spatial_loss + 0.1 * temporal_loss

			losses.update(loss.item(), image.size(0))
			loss.backward()
			optimizer.step()

			batch_time.update(time.time() - end)
			end = time.time()
			
			batchSize = depth.size(0)

			print(('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})'
			.format(epoch, batch_idx, len(TrainImgLoader), batch_time=batch_time, loss=losses)))

		if (epoch+1)%1 == 0:
			save_checkpoint(model.state_dict(), filename=args.checkpoint_dir + "ResNet18_checkpoints_small_" + str(epoch + 1) + ".pth.tar")

if __name__ == '__main__':
	train()
