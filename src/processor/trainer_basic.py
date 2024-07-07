from abc import abstractmethod
import torch
import numpy as np
from torch.optim import AdamW, lr_scheduler
from src.config.config_setup import build_model, get_dataloader
from monai.losses import DiceCELoss, DiceLoss
import torch.nn as nn
from src.utils.util import save_checkpoint
import time
import os
import torch.distributed as dist
from torch.cuda import amp
import torchio as tio


class Trainer_basic(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        a = time.time()
        self.train_data, self.val_data = get_dataloader(args, split='train'), get_dataloader(args, split='val')
        if self.args.use_sam3d_turbo:
            self.sam = build_model(args, checkpoint='../src/ckpt/sam_med3d_turbo.pth')
        else:
            self.sam = build_model(args)
        if self.args.ddp:
            self.sam = self.sam.module

        self.best_dice, self.best_epoch, self.start_epoch = 0, 0, 0
        self.pooling_layer = nn.AvgPool3d((self.args.boundary_kernel_size, self.args.boundary_kernel_size, 1), stride=1,
                                     padding=(int((self.args.boundary_kernel_size - 1) / 2),
                                              int((self.args.boundary_kernel_size - 1) / 2),
                                              0)).cuda()

        self.setup()
        print('dataloaders are created, models are loaded, and others are set, spent {} for rank {}'
              .format(round(time.time() - a, 2), self.args.rank))


    def run(self):
        self.scaler = amp.GradScaler()
        for epoch_num in range(self.start_epoch, self.args.max_epoch):
            self.sam.train()
            if self.args.ddp:
                # dist.barrier() # set a barrier until all processes are at same point
                self.train_data.sampler.set_epoch(epoch_num)

            self.train(epoch_num)
            if self.args.ddp and self.args.rank == 0:
                print('doing validation on rank=0')
                current_mean_dice = self.validate(epoch_num)
            else:
                current_mean_dice = self.validate(epoch_num)
            # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
            # if self.args.ddp:
                # dist.barrier()
            self.save_model(current_mean_dice, epoch_num)

    @abstractmethod
    def forward(self, model, image, label, iter_nums, train, return_each_iter):
        pass

    def train(self, epoch_num):
        loss_summary = []
        for idx, (image, label, data_train) in enumerate(self.train_data):
            image_path = data_train['image_path']
            print('processing {}'.format(image_path))
            self.optimizer.zero_grad()

            # increase speed based on gradient accumulation
            # my_context = self.sam.no_sync if self.args.rank != -1 and idx % self.args.accumulation_steps != 0 else nullcontext
            # with my_context():
            image, label = image.to(self.args.device), label.to(self.args.device)
            with amp.autocast():
                if self.args.initial_seg:
                    loss, _ = self.forward(self.sam, image, label, iter_nums=self.args.iter_nums, train=True, initial_seg=data_train['initial_seg'])
                else:
                    loss, _ = self.forward(self.sam, image, label, iter_nums=self.args.iter_nums, train=True)

            loss_summary.append(loss.detach().cpu().numpy())
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.sam.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            print('epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.train_data))
                  + ": loss:" + str(round(loss_summary[-1].flatten()[0], 4))
                  + ": rank:" + str(self.args.rank))
            self.logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.train_data))
                + ": loss:" + str(round(loss_summary[-1].flatten()[0], 4))
                + ": rank:" + str(self.args.rank))
        print('current lr: {}'.format(self.optimizer.param_groups[0]["lr"]))
        # If the first iteration creates NaN gradients (e.g. due to a high scaling factor and thus gradient overflow),
        # the optimizer.step() will be skipped and you might get this warning.
        self.update_lr(epoch_num, warm_up=self.args.warm_up)
        self.logger.info("- Train metrics: " + str(np.mean(loss_summary)))


    def validate(self, epoch_num):
        self.sam.eval()
        device = self.args.device
        with torch.no_grad():
            dice_list = []
            for idx, (image, label, data_val) in enumerate(self.val_data):
                image_path = data_val['image_path']
                mean_dice = 0
                image, label = image.to(device), label.to(device)
                #with amp.autocast():
                if self.args.initial_seg:
                    mean_dice_sub, masks = self.forward(self.sam, image, label, iter_nums=self.args.iter_nums,
                                                        train=False, initial_seg=data_val['initial_seg'])
                else:
                    mean_dice_sub, masks = self.forward(self.sam, image, label, iter_nums=self.args.iter_nums, train=False)

                mean_dice += mean_dice_sub
                dice_list.append(mean_dice)
                print(mean_dice_sub)
                self.logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.val_data)) +
                    ' subject: ' + str(image_path) + ' mean dice over clicks:' + str(mean_dice) +
                    ' stich left and right side (total size): ' + str(label.size(1)))
            self.logger.info("- Val metrics mean dice: " + str(np.mean(dice_list)))
        return dice_list


    def calculate_loss(self, mask, prev_masks, pred_dice, label, labels_input, iter_num, inter=False):
        mask_probs = torch.sigmoid(mask)

        seg_edge = abs(label - self.pooling_layer(label))
        mask_edge = abs(mask_probs - self.pooling_layer(mask_probs))
        edge_number = torch.count_nonzero(mask_edge) + 1

        pred_dice_score_loss = 0
        for batch_index in range(mask.size(0)):
            target_dice = 1 - self.loss_validation(mask[batch_index].unsqueeze(0), label[batch_index].unsqueeze(0))[0,0,0,0,0]

            target_dice = torch.tensor([target_dice])[0].to(self.args.device)
            pred_dice_score_loss += self.loss_boundary(pred_dice[batch_index], target_dice) * 1

        loss = self.loss_segmentation(mask, label) + self.loss_boundary(mask_edge, seg_edge) * 10
        loss = loss + pred_dice_score_loss
        return loss

    def get_dice_score(self, prev_masks, label, batch=False):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (label > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        if batch:
            return dice_list
        else:
            return (sum(dice_list) / len(dice_list)).item()

    def save_model(self, current_dice, epoch_num):
        is_best = False
        if np.mean(current_dice) > self.best_dice:
            self.best_dice = np.mean(current_dice)
            self.best_epoch = epoch_num
            is_best = True

        if not self.args.ddp or (self.args.ddp and self.args.rank == 0):
            save_checkpoint({"epoch": epoch_num + 1,
                             "best_val_loss": self.best_dice,
                             "model_state_dict": self.sam.state_dict(),
                             "optimizer": self.optimizer.state_dict(),
                             "lr_scheduler": self.lr_scheduler.state_dict(),
                             },
                            is_best=is_best,
                            checkpoint=self.args.save_dir)
        self.logger.info("- Val metrics best mean dice: {} at epoch {} " .format(self.best_dice, self.best_epoch))

    def setup(self):
        self.setup_loss()
        self.setup_optimizier()
        self.setup_scheduler()

        if self.args.resume:
            if self.args.ddp:
                dist.barrier()
            checkpoint = 'best.pth.tar' if self.args.resume_best else 'last.pth.tar'
            ckpt = torch.load(os.path.join(self.args.save_dir, checkpoint))

            self.start_epoch = ckpt["epoch"]
            self.best_epoch = self.start_epoch
            self.best_dice = ckpt["best_val_loss"]
            self.sam.load_state_dict(ckpt["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            #self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])lr_scheduler_regular
            self.lr_scheduler_regular.load_state_dict(ckpt['lr_scheduler'])


            self.logger.info(f"Resume training from epoch {self.start_epoch}!")
            del ckpt
            torch.cuda.empty_cache()

    def setup_loss(self):
        self.loss_boundary = nn.MSELoss()
        self.mse_none = nn.MSELoss(reduction='none')

        to_onehot_y = True if self.args.plain else False

        self.loss_segmentation = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean', to_onehot_y=to_onehot_y)
        self.loss_Dice = DiceLoss(sigmoid=True, to_onehot_y=to_onehot_y)
        self.loss_validation = DiceLoss(sigmoid=True, reduction='none', to_onehot_y=to_onehot_y)

        self.l1 = nn.L1Loss()
        self.inter_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def setup_optimizier(self):
        self.optimizer = AdamW([
            {'params': self.sam.image_encoder.parameters()},
            {'params': self.sam.prompt_encoder.parameters()},
            {'params': self.sam.mask_decoder.parameters()},
        ], lr=self.args.lr)

    def setup_scheduler(self):
        if self.args.lr_scheduler == 'linear':
            self.lr_scheduler_regular = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=500)
        else:
            self.lr_scheduler_regular = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        if self.args.warm_up:
            self.linear_warmup_scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)

    def update_lr(self, epoch, warmup_epoch=10, warm_up=False):
        if warm_up:
            if epoch < warmup_epoch:
                self.lr_scheduler = self.linear_warmup_scheduler
            else:
                self.lr_scheduler = self.lr_scheduler_regular
        else:
            self.lr_scheduler = self.lr_scheduler_regular
        self.lr_scheduler.step()










