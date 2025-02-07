import logging
import os.path
import torch
from utils.util import setup_logger
from config.config_args import *
import numpy as np
from torch.backends import cudnn
from src.config.config_setup import build_model, get_dataloader
import time, random
import torch.nn.functional as F
from src.utils.util import _bbox_mask
from src.utils import scribble, boundary_selection
import torchio as tio
import surface_distance
from surface_distance import metrics

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


class Tester(object):
    def __init__(self, args, logger, ckpt):
        self.args = args
        self.logger = logger

        self.val_data = get_dataloader(args, split='test')

        print('loading models and setting up')
        self.sam = build_model(args, checkpoint=ckpt)

        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder
        self.mask_decoder = self.sam.mask_decoder

        self.current_iter = 0
        # self._load_pretrain_model(ckpt)

    def _load_pretrain_model(self, ckpt):
        model_dict = torch.load(ckpt, map_location=self.args.device)
        state_dict = model_dict
        self.sam.load_state_dict(state_dict['model_state_dict'])

    def validate(self, epoch_num):
        self.image_encoder.eval()
        self.prompt_encoder.eval()
        self.mask_decoder.eval()

        if self.args.data == 'lits':
            loss = self.validater_sliding_window(epoch_num)
        else:
            loss = self.validater(epoch_num)
        return loss


    def validater_sliding_window(self, epoch_num):
        with torch.no_grad():
            dice_summary, nsd_summary = [], []
            for idx, (subject_dict, image_path, subject_dict_save) in enumerate(self.val_data):
                if subject_dict['label']['data'][0].sum() <= 0:
                    self.logger.info(image_path, 'label volume too small, and it has been skipped for validation')
                    continue
                mean_dice = 0
                subject = tio.Subject(image=tio.ScalarImage(tensor=subject_dict['image']['data'][0].float(), affine=subject_dict['image']['affine'][0]),
                                      label=tio.LabelMap(tensor=subject_dict['label']['data'][0].float(), affine=subject_dict['label']['affine'][0]))
                grid_sampler = tio.inference.GridSampler(subject, 128, 16)
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')


                for idx_patch, patches_batch in enumerate(patch_loader):
                    image, label = patches_batch['image'][tio.DATA].to(self.args.device), patches_batch['label'][tio.DATA].to(self.args.device)
                    print(torch.count_nonzero(label))
                    print('how many voxels')
                    locations = patches_batch[tio.LOCATION]

                    if torch.count_nonzero(label) == 0:
                        print('found empty patch')
                        masks = torch.zeros([1, 1, 128, 128, 128])
                    else:
                        _, masks = self._interaction(self.sam, image, label, iter_nums=self.args.iter_nums, train=False)

                    aggregator.add_batch(masks, locations)
                masks_iter_final = aggregator.get_output_tensor()
                mean_dice_sub = self.get_dice_score(torch.sigmoid(masks_iter_final), subject.label.data)

                mean_dice += mean_dice_sub
                dice_summary.append(mean_dice)

                ssd = surface_distance.compute_surface_distances(
                    (subject.label.data == 1)[0].cpu().numpy(),
                    (torch.sigmoid(masks_iter_final) > 0.5)[0].cpu().numpy(),
                    spacing_mm=(1,1,1)
                )
                nsd = metrics.compute_surface_dice_at_tolerance(ssd, 1)

                nsd_summary.append(nsd)
                print(mean_dice_sub)

                if self.args.save_predictions:
                    save_test_dir = os.path.join(self.args.save_test_dir, 'prism_prediction', self.args.data, self.args.save_name, str(self.args.iter_nums))
                    if not os.path.exists(save_test_dir):
                        os.makedirs(save_test_dir)
                    a = torch.sigmoid(masks_iter_final) > 0.5
                    a = a[0].float().cpu().numpy()
                    import SimpleITK as sitk
                    prediction = sitk.GetImageFromArray(a)
                    if self.args.data == 'lits':
                        base_name = image_path[0].split('/')[-2] + '_' +image_path[0].split('/')[-1]
                    if self.args.refine_test:
                        pred_name = base_name.replace('.nii.gz', '._pred.nii.gz')
                    else:
                        pred_name = base_name.replace('.nii.gz', '._pred_no_refine.nii.gz')
                    save_path = os.path.join(save_test_dir, pred_name)
                    sitk.WriteImage(prediction, save_path)

                    if self.args.iter_nums == 1:
                        if self.args.refine_test:
                            image_name = base_name.replace('.nii.gz', '._image.nii.gz')
                        else:
                            image_name = base_name.replace('.nii.gz', '._image_no_refine.nii.gz')
                        b = subject_dict_save['image']['data'][0][0].float().cpu().numpy()
                        image_save = sitk.GetImageFromArray(b)
                        sitk.WriteImage(image_save, os.path.join(save_test_dir, image_name))

                        if self.args.refine_test:
                            label_name = base_name.replace('.nii.gz', '._label.nii.gz')
                        else:
                            label_name = base_name.replace('.nii.gz', '._label_no_refine.nii.gz')
                        c = subject_dict_save['label']['data'][0][0].float().cpu().numpy()
                        label_save = sitk.GetImageFromArray(c)
                        sitk.WriteImage(label_save, os.path.join(save_test_dir, label_name))



                self.logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.val_data)) +
                    ' subject: ' + str(image_path) + ' mean nsd over clicks:' + str(nsd) + ' mean dice over clicks:' + str(mean_dice) +
                    ' stich left and right side (total size): ' + str(label.size(1)))
            self.logger.info("- Val metrics mean dice: " + str(np.mean(dice_summary)) + "- Val metrics nsd: " + str(np.mean(nsd_summary)))

            from scipy import stats
            data = dice_summary
            # Calculate mean
            mean = np.mean(data)
            # Calculate standard error of the mean (SEM)
            sem = stats.sem(data)
            # Determine the t-value for the 95% confidence interval
            # Degrees of freedom
            df = len(data) - 1
            # t-value for 95% CI
            t_value = stats.t.ppf(0.975, df)
            # Calculate the margin of error
            margin_of_error = sem * t_value
            # Calculate the 95% CI
            ci_lower = mean - margin_of_error
            ci_upper = mean + margin_of_error
            self.logger.info("- ci_lower dice: " + str(ci_lower) + " - ci_upper dice: " + str(ci_upper))

        return dice_summary

    def validater(self, epoch_num):
        device = self.args.device
        with torch.no_grad():
            loss_summary, nsd_summary = [], []
            # for idx, data in enumerate(val_data):
            # img, label = data['image'].to(device), data['label'].to(device)
            for idx, (image, label, data_output) in enumerate(self.val_data):

                self.image_path, self.subject_dict_save = data_output['image_path'], data_output['subject_save']
                image, label = image.to(device), label.to(device)

                if self.args.initial_seg:
                    masks, prompts = self.interaction(self.sam, image, label, initial_seg=data_output['initial_seg'])
                else:
                    masks, prompts = self.interaction(self.sam, image, label)

                # masks = self.interaction(self.sam, image, label)

                dice = self.get_dice_score(torch.sigmoid(masks), label)
                loss_summary.append(dice)

                ssd = surface_distance.compute_surface_distances(
                    (label == 1)[0][0].cpu().numpy(),
                    (torch.sigmoid(masks) > 0.5)[0][0].cpu().numpy(),
                    spacing_mm=(1, 1, 1)
                )
                nsd = metrics.compute_surface_dice_at_tolerance(ssd, 1)

                nsd_summary.append(nsd)


                self.logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.val_data)) +
                    ' subject: ' + str(self.image_path) + ' mean nsd over clicks:' + str(nsd) + ' mean dice over clicks:' + str(dice) +
                    ' stich left and right side (total size): ' + str(label.size(1)))

            self.logger.info("- Val metrics mean dice: " + str(np.mean(loss_summary)) + "- Val metrics nsd: " + str(np.mean(nsd_summary)))
            from scipy import stats
            data = loss_summary
            # Calculate mean
            mean = np.mean(data)
            # Calculate standard error of the mean (SEM)
            sem = stats.sem(data)
            # Determine the t-value for the 95% confidence interval
            # Degrees of freedom
            df = len(data) - 1
            # t-value for 95% CI
            t_value = stats.t.ppf(0.975, df)
            # Calculate the margin of error
            margin_of_error = sem * t_value
            # Calculate the 95% CI
            ci_lower = mean - margin_of_error
            ci_upper = mean + margin_of_error
            self.logger.info("- ci_lower dice: " + str(ci_lower) + "- ci_lower dice: " + str(ci_upper))

        return loss_summary


    def sampling_slice(self, fn_mask, fp_mask, step_size=1, joint=False):

        new_fn_mask = torch.zeros_like(fn_mask, dtype=torch.bool)
        new_fp_mask = torch.zeros_like(fp_mask, dtype=torch.bool)

        if joint:
            index_list = []
            union_mask = fn_mask | fp_mask
            if torch.count_nonzero(union_mask) > 0:
                for i in range(0, union_mask.shape[-1]):
                    current_slice = union_mask[:, :, i]
                    if torch.count_nonzero(current_slice) > 0:
                        index_list.append(i)
                if torch.count_nonzero(fn_mask) > 0:
                    for i in range(index_list[0], index_list[-1], step_size):
                        new_fn_mask[:, :, :, :, i] = fn_mask[:, :, :, :, i]
                else:
                    new_fn_mask = fn_mask
                if torch.count_nonzero(fp_mask) > 0:
                    for i in range(index_list[0], index_list[-1], step_size):
                        new_fp_mask[:, :, :, :, i] = fp_mask[:, :, :, :, i]
                else:
                    new_fp_mask = fp_mask

            else:
                new_fn_mask = fn_mask
                new_fp_mask = fp_mask
        else:
            if torch.count_nonzero(fn_mask) > 0:
                index_list = []
                for i in range(0, fn_mask.shape[-1]):
                    current_slice = fn_mask[:, :, i]
                    if torch.count_nonzero(current_slice) > 0:
                        index_list.append(i)
                for i in range(index_list[0], index_list[-1], step_size):
                    new_fn_mask[:, :, :, :, i] = fn_mask[:, :, :, :, i]
            else:
                new_fn_mask = fn_mask

            if torch.count_nonzero(fp_mask) > 0:
                index_list = []
                for i in range(0, fp_mask.shape[-1]):
                    current_slice = fp_mask[:, :, i]
                    if torch.count_nonzero(current_slice) > 0:
                        index_list.append(i)
                for i in range(index_list[0], index_list[-1], step_size):
                    new_fp_mask[:, :, :, :, i] = fp_mask[:, :, :, :, i]
            else:
                new_fp_mask = fp_mask

        return new_fn_mask, new_fp_mask

    def sampling_slice_for_propagate(self, gt, fn_mask, fp_mask, step_size=1):
        new_fn_mask = torch.zeros_like(fn_mask, dtype=torch.bool)
        new_fp_mask = torch.zeros_like(fp_mask, dtype=torch.bool)

        index_list = []
        for i in range(0, gt.shape[-1]):
            current_slice = gt[:, :, i]
            if torch.count_nonzero(current_slice) > 0:
                index_list.append(i)

        for i in range(2, len(index_list)-2, step_size):
            new_fn_mask[:, :, :, :, index_list[i]] = fn_mask[:, :, :, :, index_list[i]]
            new_fp_mask[:, :, :, :, index_list[i]] = fp_mask[:, :, :, :, index_list[i]]

        new_fn_mask[:, :, :, :, index_list[0]] = fn_mask[:, :, :, :, index_list[0]]
        new_fp_mask[:, :, :, :, index_list[0]] = fp_mask[:, :, :, :, index_list[0]]
        new_fn_mask[:, :, :, :, index_list[1]] = fn_mask[:, :, :, :, index_list[1]]
        new_fp_mask[:, :, :, :, index_list[1]] = fp_mask[:, :, :, :, index_list[1]]

        new_fn_mask[:, :, :, :, index_list[-1]] = fn_mask[:, :, :, :, index_list[-1]]
        new_fp_mask[:, :, :, :, index_list[-1]] = fp_mask[:, :, :, :, index_list[-1]]
        new_fn_mask[:, :, :, :, index_list[-2]] = fn_mask[:, :, :, :, index_list[-2]]
        new_fp_mask[:, :, :, :, index_list[-2]] = fp_mask[:, :, :, :, index_list[-2]]

        return new_fn_mask, new_fp_mask



    def get_next_click3D_torch_2(self, prev_seg, gt_semantic_seg):

        mask_threshold = 0.5

        batch_points = []
        batch_labels = []
        # dice_list = []

        pred_masks = (prev_seg > mask_threshold)
        true_masks = (gt_semantic_seg > 0)
        fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
        fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

        # propagate sampling
        # fn_masks, fp_masks = self.sampling_slice_for_propagate(gt_semantic_seg, fn_masks, fp_masks, step_size=2)
        # if self.current_iter < 7:
        #     fn_masks, fp_masks = self.sampling_slice_for_propagate(gt_semantic_seg, fn_masks, fp_masks, step_size=5)
        # fn_masks, fp_masks = self.sampling_slice(fn_masks, fp_masks, step_size=2, joint=True)

        print('fn: {}, fp: {}'.format(torch.count_nonzero(fn_masks) / torch.count_nonzero(true_masks),
                                      torch.count_nonzero(fp_masks) / torch.count_nonzero(true_masks)))
        to_point_mask = torch.logical_or(fn_masks, fp_masks)
        #to_point_mask = fn_masks
        for i in range(gt_semantic_seg.shape[0]):
            bp_list, bl_list = [], []
            points = torch.argwhere(to_point_mask[i])
            if self.args.num_clicks > len(points):
                click_size = len(points)
            else:
                click_size = self.args.num_clicks

            dynamic_size = random.randint(1, click_size) if self.args.dynamic else click_size

            point_index = np.random.choice(len(points), size=dynamic_size, replace=False)
            points_select = points[point_index]  # each row tensor([0, x, y, z]), size --> num_clicks x 4
            # point = points[np.random.randint(len(points))] # tensor([0, x, y, z])
            for click_index in range(dynamic_size):
                point = points_select[click_index]
                if fn_masks[i, 0, point[1], point[2], point[3]]:
                    is_positive = True
                else:
                    is_positive = False

                bp = point[1:].clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive), ]).reshape(1, 1)
                bp_list.append(bp)
                bl_list.append(bl)

            if self.args.use_scribble:
                #sample_method = random.choice(['line', 'center', 'default'])

                # sample_method = 'center'
                sample_method = 'center'
                scribble_types = {
                    'line': 'LineScribble',
                    'center': 'CenterlineScribble',
                    'default': 'ContourScribble'
                }

                def create_scribble_mask(scribble_type, data, orientation='axial'):

                    # non-random scribbles
                    #scribble_object = getattr(scribble, scribble_type)(warp=False, break_mask=False) # centerline
                    # scribble_object = getattr(scribble, scribble_type)(warp=False, break_mask=False, blur_mask=False) # contour

                    scribble_object = getattr(scribble, scribble_type)()

                    scribble_mask = scribble_object.batch_scribble(data)
                    if orientation == 'sagittal':
                        scribble_mask = scribble_mask.permute(1, 0, 2, 3)
                    elif orientation == 'coronal':
                        scribble_mask = scribble_mask.permute(1, 2, 0, 3)
                    else:
                        scribble_mask = scribble_mask.permute(1, 2, 3, 0)

                    return scribble_mask > 0

                # fg = gt_semantic_seg[i].permute(3, 0, 1, 2).float()
                # bg = (torch.ones_like(pred_masks[i, :]).float() - gt_semantic_seg[i].float()).permute(3, 0, 1, 2)


                if self.args.sample_fn_fp_masks:
                    CC_num = 5
                    import cc3d
                    fn_masks_np = fn_masks[i][0, :].cpu().numpy()
                    _, N = cc3d.connected_components(fn_masks_np, return_N=True)  # free
                    print('found CCs from FN:{}'.format(N))
                    if N > CC_num:
                        labels_out, _ = cc3d.largest_k(
                            fn_masks_np, k=CC_num,
                            connectivity=26, delta=0,
                            return_N=True,
                        )

                        fn_masks_np *= (labels_out > 0)
                        new_fn_masks = torch.from_numpy(fn_masks_np).unsqueeze(0).unsqueeze(0)
                        print('filtered out for FN:{}'.format(torch.sum(fn_masks[i].float() - new_fn_masks.float().to(self.args.device))))
                        #fn_masks[i].data = new_fn_masks
                        fn_masks[i] = new_fn_masks
                    if torch.count_nonzero(fp_masks[i]) > 0:
                        fp_masks_np = fp_masks[i][0, :].cpu().numpy()
                        _, N = cc3d.connected_components(fp_masks_np, return_N=True)  # free
                        print('found CCs from FP:{}'.format(N))
                        if N > CC_num:
                            labels_out, _ = cc3d.largest_k(
                                fp_masks_np, k=CC_num,
                                connectivity=26, delta=0,
                                return_N=True,
                            )

                            fp_masks_np *= (labels_out > 0)
                            new_fp_masks = torch.from_numpy(fp_masks_np).unsqueeze(0).unsqueeze(0)
                            print('filtered out :{}'.format(
                                torch.sum(fp_masks[i].float() - new_fp_masks.float().to(self.args.device))))
                            # fn_masks[i].data = new_fn_masks
                            fp_masks[i] = new_fp_masks


                fg, bg = fn_masks[i].permute(3, 0, 1, 2).float(), fp_masks[i].permute(3, 0, 1, 2).float()
                # axial
                # fg, bg = fn_masks[i].permute(3, 0, 1, 2).float(), fp_masks[i].permute(3, 0, 1, 2).float()

                # sagittal
                if self.args.scribble_sagittal:
                    orientation = 'sagittal'
                    fg, bg = fg.permute(1, 2, 3, 0).float(), bg.permute(1, 2, 3, 0).float()
                    fg, bg = fg.permute(1, 0, 2, 3).float(), bg.permute(1, 0, 2, 3).float()
                    print('sagittal')
                else:
                    orientation = 'axial'
                # elif self.args.scribble_coronal:
                # # coronal
                #     fg, bg = fn_masks[i].permute(2, 0, 1, 3).float(), fp_masks[i].permute(2, 0, 1, 3).float()
                #     print('coronal')
                # else:
                #     fg, bg = fn_masks[i].permute(3, 0, 1, 2).float(), fp_masks[i].permute(3, 0, 1, 2).float()
                #     print('axial')

                scribble_type = scribble_types.get(sample_method, scribble_types['default']) # if sample_method otherwise default
                scribble_mask_fg = create_scribble_mask(scribble_type, fg, orientation=orientation)

                #fg_coors = torch.argwhere(scribble_mask_fg)[:, 1:].unsqueeze(0)[:, 0: 100, :]  # for computation only
                fg_coors = torch.argwhere(scribble_mask_fg)[:, 1:].unsqueeze(0)
                if self.args.efficient_scribble:
                    fg_coors = fg_coors[:, 0: 10000, :]  # for computation only# for computation only
                fg_coors_label = torch.ones(1, fg_coors.size(1))
                bp_list.append(fg_coors)
                bl_list.append(fg_coors_label)


                #if sample_method == 'default':
                if torch.count_nonzero(fp_masks) > 0:
                    scribble_mask_bg = create_scribble_mask(scribble_type, bg, orientation=orientation)
                    bg_coors = torch.argwhere(scribble_mask_bg)[:, 1:].unsqueeze(0)
                    if self.args.efficient_scribble:
                        bg_coors = bg_coors[:, 0: 10000, :]
                    bg_coors_label = torch.zeros(1, bg_coors.size(1))
                    bp_list.append(bg_coors)
                    bl_list.append(bg_coors_label)

            batch_points.append(torch.cat(bp_list, dim=1))
            batch_labels.append(torch.cat(bl_list, dim=1))

            smallest_n = min(tensor.size(1) for tensor in batch_labels)
            batch_points = [tensor[:, :smallest_n] if tensor.size(1) > smallest_n else tensor for tensor in
                            batch_points]
            batch_labels = [tensor[:, :smallest_n] if tensor.size(1) > smallest_n else tensor for tensor in
                            batch_labels]

            # Check the shapes of the adjusted tensors
            for i, tensor in enumerate(batch_points):
                print(f"Tensor {i + 1} shape: {tensor.shape}")


        return batch_points, batch_labels

    def get_points(self, prev_masks, label):
        batch_points, batch_labels = self.get_next_click3D_torch_2(prev_masks, label)

        points_co = torch.cat(batch_points, dim=0).to(self.args.device)
        points_la = torch.cat(batch_labels, dim=0).to(self.args.device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_input = points_co
        labels_input = points_la

        bbox_coords = _bbox_mask(label[:, 0, :]).to(self.args.device) if self.args.use_box else None
        return points_input, labels_input, bbox_coords


    def batch_forward(self, sam_model, features, image_embedding, image, prev_masks, points=None, boxes=None):
        prev_masks = F.interpolate(prev_masks.float(), scale_factor=0.25)
        features = [features[i].to(self.args.device) for i in range(0, len(features))]

        # FIXME
        #  PRISM prompt encoder 31883327616

        # image_embedding = torch.rand(1, 384, 8, 8, 8)
        # mask = torch.rand(1, 1, 128, 128, 128)
        # mask = F.interpolate(mask.float(), scale_factor=0.25)
        # a = points[0].cpu()
        # b = points[1].cpu()
        # c = boxes.cpu()
        # self.sam.cpu()
        # from fvcore.nn import FlopCountAnalysis
        # flop_counter = FlopCountAnalysis(self.sam.prompt_encoder.cpu(), inputs=([a, b], c, mask, image_embedding))
        # print(flop_counter.total())
        # print(1)


        # sparse_embeddings --> (B, 2, embed_dim) 2 represents concat of coordination and its label
        # dense_embeddings --> (B, embed_dim, W, H, D), whd values are customized
        new_point_embedding, new_image_embedding = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=prev_masks,
            image_embeddings=image_embedding.to(self.args.device)
        )

        mask, pred_dice = sam_model.mask_decoder(
            prompt_embeddings=new_point_embedding,  # (B, 2, 256)
            image_embeddings=new_image_embedding,  # (B, 256, 64, 64)
            feature_list=features,
        )

        return mask, pred_dice

    def interaction(self, sam_model, image, label, initial_seg=None):

        if self.args.initial_seg:
            if self.args.use_penn:
                prev_masks = initial_seg.float().to(label.device)

                # bbox_coords = _bbox_mask(label[:, 0, :]).to(self.args.device) if self.args.use_box else None
                # prev_masks = torch.zeros_like(label)
                # prev_masks[:, :, bbox_coords[:, :, 0] : bbox_coords[:, :, 3],
                # bbox_coords[:, :, 1] : bbox_coords[:, :, 4], bbox_coords[:, :, 2] : bbox_coords[:, :, 5]] = 1

                # kernel_size = 3  # 3x3x3 cube
                # kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size)).to(label.device)
                # dilated_tensor = F.conv3d(prev_masks, kernel, padding=kernel_size // 2, stride=1)
                # prev_masks = (dilated_tensor > 0).float()



                dice_penn = self.get_dice_score(torch.sigmoid(prev_masks).cpu().numpy(), label.cpu().numpy())
                print('using segmentations from Penn, Dice: {}'.format(dice_penn))
                initial_mask = prev_masks
            else:
                # TODO: below is "produce initial segmentation without prompt"
                print('to do next')
                prev_masks = torch.zeros_like(label, dtype=torch.float).to(label.device)
                image_embedding, feature_list = self.sam.image_encoder(image)

                # import torchstat
                # torchstat.model_stats(self.sam.image_encoder, input_size=(1, 1, 128, 128, 128))

                prev_masks, _ = self.iteration_forward(sam_model, feature_list, image_embedding, prev_masks,
                                                       points=None, boxes=None)

                prev_masks = torch.sigmoid(prev_masks)
                prev_masks = (prev_masks > 0.5)[:, 0, :]
                prev_masks = prev_masks[:, None, :, :, :].float().to(label.device)
                dice = self.get_dice_score(torch.sigmoid(prev_masks[0, :]).cpu().numpy(), label[0, :].cpu().numpy())
                print(dice)
        else:
            prev_masks = torch.zeros_like(label, dtype=torch.float).to(label.device)

        prev_masks = prev_masks.float().to(label.device)

        if not self.args.use_penn:
            start_time = time.time()
            image_embedding, feature_list = self.sam.image_encoder(image)
            image_encoder_time = time.time() - start_time
            print('image_encoder takes: {}'.format(image_encoder_time))
        self.click_points = []
        self.click_labels = []


        for iter_num in range(self.args.iter_nums):
            self.current_iter = iter_num

            prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks
            points_input, labels_input, bbox_input = self.get_points(prev_masks_sigmoid, label)

            # prev_masks = (torch.sigmoid(prev_masks) > 0.5)
            # print(torch.unique(prev_masks))
            if self.args.use_penn:
                mask_best = prev_masks

                # mask_best = prev_masks if iter_num == 0 else torch.sigmoid(prev_masks)
                # if iter_num > 0:
                #     mask_best = mask_best > 0.5
                #     mask_best = mask_best.float()
            else:
                start_time = time.time()
                mask, pred_dice = self.batch_forward(sam_model, feature_list, image_embedding, image, prev_masks, points=[points_input, labels_input], boxes=bbox_input)
                batch_forward_time = time.time() - start_time
                print('batch forward takes: {}'.format(batch_forward_time))
                if self.args.multiple_outputs:
                    pred_best_dice, pred_dice_max_index = torch.max(pred_dice, dim=1)
                    mask_best = mask[:, pred_dice_max_index, :]
                else:
                    mask_best, pred_best_dice = mask, pred_dice

                # FIXME refine or not
            if self.args.refine and self.args.refine_test:
                start_time = time.time()
                if self.args.no_detach:
                    mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best,
                                                                          [self.click_points, self.click_labels],
                                                                          mask_best.detach())
                else:

                    #FIXME
                    # lightweight model flops 29285154816
                    # PRISM                   29209133056
                    # need to change some lines in Refine.forward function
                    #from fvcore.nn import FlopCountAnalysis
                    # image = torch.rand(1, 4, 128, 128, 128)
                    # mask = torch.rand(1, 1, 128, 128, 128)
                    # self.sam.cpu()
                    # flop_counter = FlopCountAnalysis(self.sam.mask_decoder.refine.cpu(), inputs=(image, mask))
                    # print(flop_counter.total())
                    # print(1)


                    mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best,
                                                                          [self.click_points, self.click_labels],
                                                                          mask_best.detach())


                refine_time = time.time() - start_time
                print('refine takes: {}'.format(refine_time))
                print('dice before refine {} and after {}'.format(
                    self.get_dice_score(torch.sigmoid(mask_best), label),
                    self.get_dice_score(torch.sigmoid(mask_refine), label))
                )
                mask_best = mask_refine

            prev_masks = mask_best

            dice = self.get_dice_score(torch.sigmoid(prev_masks).cpu().numpy(), label.cpu().numpy())
            print('---')
            if self.args.use_penn:
                print(f'Dice: {dice:.4f}, label: {labels_input}')
            else:
                print(f'Dice: {dice:.4f}, pred_dice: {pred_best_dice}, label: {labels_input}')

            prompts = {'points': self.click_points, 'labels': self.click_labels}

            print(time.time()-start_time)
            if self.args.save_predictions:
                save_test_dir = os.path.join(self.args.save_test_dir, 'prism_prediction', self.args.data,
                                             self.args.save_name, str(iter_num))
                if not os.path.exists(save_test_dir):
                    os.makedirs(save_test_dir)
                a = torch.sigmoid(prev_masks) > 0.5
                a = a.float().cpu().numpy()
                import cc3d
                a1, N = cc3d.largest_k(
                    a[0, 0, :], k=1,
                    connectivity=26, delta=0,
                    return_N=True,
                )
                a = np.expand_dims(a1, axis=0)
                a = np.expand_dims(a, axis=0)
                import SimpleITK as sitk
                prediction = sitk.GetImageFromArray(a)

                base_name = self.image_path[0].split('/')[-1]
                if self.args.refine_test:
                    pred_name = base_name.replace('.nii.gz', '_pred.nii.gz')
                else:
                    pred_name = base_name.replace('.nii.gz', '_pred_no_refine.nii.gz')
                save_path = os.path.join(save_test_dir, pred_name)
                sitk.WriteImage(prediction, save_path)

                b = np.zeros(a.shape)

                prompts_point, prompts_label = prompts['points'], prompts['labels']
                for iter_i in range(0, len(prompts_label)):
                    prompts_point_iter, prompts_label_iter = prompts_point[iter_i], prompts_label[iter_i]
                    for iter_j in range(0, len(prompts_point_iter[0, :])):
                        point_iter_j = prompts_point_iter[0, iter_j, :].cpu().numpy()
                        label_iter_j = prompts_label_iter[0, iter_j].cpu().numpy()

                        label_iter_j = label_iter_j + (iter_i * 2) + 1
                        label_iter_j = int(label_iter_j)


                        # if self.args.scribble_sagittal:
                        #     # sagittal
                        #     b[0, 0, point_iter_j[2], point_iter_j[0], point_iter_j[1]] = label_iter_j
                        # else:
                        #     # axial
                        b[0, 0, point_iter_j[0], point_iter_j[1], point_iter_j[2]] = label_iter_j

                        # coronal
                        # b[0, 0, point_iter_j[1], point_iter_j[2], point_iter_j[0]] = label_iter_j

                print('iteration: {}, cumulative prompt voxel number: {}'.format(iter_num, np.count_nonzero(b)))
                prompts_itk_image = sitk.GetImageFromArray(b)
                prompt_name = base_name.replace('.nii.gz', '_prompt.nii.gz')
                save_path = os.path.join(save_test_dir, prompt_name)
                sitk.WriteImage(prompts_itk_image, save_path)
                print(1)

                if iter_num == 0:
                    if self.args.refine_test:
                        image_name = base_name.replace('.nii.gz', '_image.nii.gz')
                    else:
                        image_name = base_name.replace('.nii.gz', '_image_no_refine.nii.gz')
                    b = self.subject_dict_save['image']['data'][0][0].float().cpu().numpy()
                    image_save = sitk.GetImageFromArray(b)
                    sitk.WriteImage(image_save, os.path.join(save_test_dir, image_name))

                    if self.args.use_penn:
                        image_name = base_name.replace('.nii.gz', '_penn.nii.gz')
                        b = self.subject_dict_save['seg']['data'][0][0].float().cpu().numpy()
                        image_save = sitk.GetImageFromArray(b)
                        sitk.WriteImage(image_save, os.path.join(save_test_dir, image_name))

                    if self.args.refine_test:
                        label_name = base_name.replace('.nii.gz', '_label.nii.gz')
                    else:
                        label_name = base_name.replace('.nii.gz', '_label_no_refine.nii.gz')
                    c = self.subject_dict_save['label']['data'][0][0].float().cpu().numpy()
                    label_save = sitk.GetImageFromArray(c)
                    sitk.WriteImage(label_save, os.path.join(save_test_dir, label_name))

                    label_name = base_name.replace('.nii.gz', '_initial_mask.nii.gz')
                    c = initial_mask[0][0].float().cpu().numpy()
                    label_save = sitk.GetImageFromArray(c)
                    sitk.WriteImage(label_save, os.path.join(save_test_dir, label_name))

        # prompts = {'points': self.click_points, 'labels': self.click_labels}
        return prev_masks, prompts



    def _interaction(self, sam_model, image, label, iter_nums, train=False, return_each_iter=False):
        if return_each_iter:
            return_mask_total_iter = torch.zeros([iter_nums, 1, image.size(2), image.size(3), image.size(4)])

        image_embedding, feature_list = self.sam.image_encoder(image)
        self.click_points = []
        self.click_labels = []
        return_loss = 0
        prev_masks = torch.zeros_like(label, dtype=torch.float).to(label.device)
        for iter_num in range(iter_nums):
            prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

            if self.args.init_learning and iter_num == 0:
                boundary, margin, content = boundary_selection.find_boundary_map(label)
                use_content = True
                for batch_index in range(label.size(0)):
                    if torch.count_nonzero(content[batch_index]) < self.args.num_clicks:
                        use_content = False
                if use_content:
                    label_sample = content
                else:
                    label_sample = label
            else:
                label_sample = label

            points_input, labels_input, box_input = self.get_points(prev_masks_sigmoid, label_sample, label)
            mask, dice_pred = self.batch_forward(sam_model, feature_list, image_embedding, image, prev_masks, points=[points_input, labels_input], boxes=box_input)

            # ========================================================
            if self.args.multiple_outputs:
                dice_pred_best, max_label_index = torch.max(dice_pred, dim=1)
                mask_list = [mask[i, max_label_index[i], :].unsqueeze(0) for i in range(mask.size(0))]
                mask_best = torch.stack(mask_list, dim=0)
            else:
                mask_best = mask

            # ========================================================

            if self.args.refine and self.args.refine_test:
                if self.args.no_detach:
                    mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best,
                                                                          [self.click_points, self.click_labels],
                                                                          mask_best.detach())
                else:
                    mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best,
                                                                          [self.click_points, self.click_labels],
                                                                          mask_best.detach())
                self.logger.info('dice before refine {} and after {}, label 0: {}, label 1: {}'.format(
                    self.get_dice_score(torch.sigmoid(mask_best), label), self.get_dice_score(torch.sigmoid(mask_refine), label),
                    str(labels_input.numel() - torch.count_nonzero(labels_input)), str(torch.count_nonzero(labels_input)) ) )
                mask_best = mask_refine  # FIXME refine or not

            loss = self.get_dice_score(torch.sigmoid(mask_best), label) # dice

            return_loss += loss
            prev_masks = mask_best

            if return_each_iter:
                return_mask_total_iter[iter_num, :] = mask_best
        if return_each_iter:
            print(return_mask_total_iter.shape)
            return return_loss / iter_nums, return_mask_total_iter
        else:
            return return_loss / iter_nums, prev_masks

    def get_dice_score(self, prev_masks, label, binary=True):
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
        if binary:
            true_masks = (label > 0)
        else:
            squeezed_array = np.squeeze(label, axis=1)
            one_hot_encoded = np.eye(np.max(label) + 1)[squeezed_array]
            true_masks = np.transpose(one_hot_encoded, (0, 4, 1, 2, 3))

        dice_list = []

        if binary:
            for i in range(true_masks.shape[0]):
                dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
            return (sum(dice_list) / len(dice_list)).item()
        else:
            for i in range(true_masks.shape[0]):
                for j in range(1, true_masks.shape[1]):
                    dice_list.append(compute_dice(pred_masks[i][j], true_masks[i][j]))
            return (sum(dice_list) / len(dice_list)).item(), dice_list




def main():
    init_seeds()
    args = parser.parse_args()
    check_and_setup_parser(args)

    log_name = 'test_' + args.save_name
    setup_logger(logger_name=log_name, root=args.save_dir, screen=True, tofile=True)
    logger = logging.getLogger(log_name)
    logger.info(str(args))

    #ckpt = '/home/hao/Hao/3D_medical_foundation_model/src/implementation/log/colon/3DSAM/best.pth.tar'
    ckpt = os.path.join(args.save_dir, args.checkpoint + '.pth.tar')
    with torch.no_grad():
        tester = Tester(args, logger, ckpt)
        loss = tester.validate(epoch_num=0)

        print(loss)

    logger.info("- Test done")

if __name__ == "__main__":
    main()