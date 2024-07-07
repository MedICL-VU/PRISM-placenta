from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
import pickle
import numpy as np
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)
import cc3d, math

class Dataset_promise(Dataset):
    def __init__(self, data, data_dir, split='train', image_size=128, transform=None, pcc=False, args=None):
        self.args = args
        self.data = data
        self.paths = data_dir

        self._set_file_paths(self.paths, split)
        self._set_dataset_stat()

        self.image_size = (image_size, image_size, image_size)
        self.transform = transform
        self.threshold = 0
        self.split = split
        self.pcc = pcc

        self.monai_transforms = self._get_train_transforms(split=split)

        self.cc = 1

    def __len__(self):
        return len(self.label_paths)

    # if self.args.scribble_sagittal:
    #     fg, bg = fg.permute(1, 2, 3, 0).float(), bg.permute(1, 2, 3, 0).float()
    #     fg, bg = fg.permute(1, 0, 2, 3).float(), bg.permute(1, 0, 2, 3).float()
    #     print('sagittal')
    # elif self.args.scribble_coronal:
    #     fg, bg = fg.permute(1, 2, 3, 0).float(), bg.permute(1, 2, 0).float()
    #     fg, bg = fg.permute(2, 0, 1, 3).float(), bg.permute(2, 0, 1, 3).float()
    #     print('coronal')
    # else:
    #     print('axial')
    def swap_axis(self, sitk_image, view='sagittal'):
        a = sitk.GetArrayFromImage(sitk_image)
        if view == 'sagittal':
            a_swap = np.swapaxes(a, 0, 1)
            print('sagittal')
        else:
            a_swap = np.swapaxes(a, 0, 2)
            print('coronal')
        swapped_sitk_image = sitk.GetImageFromArray(a_swap)
        swapped_sitk_image.SetOrigin(sitk_image.GetOrigin())
        swapped_sitk_image.SetSpacing(sitk_image.GetSpacing())
        #b = sitk.GetArrayFromImage(swapped_sitk_image)
        return swapped_sitk_image

    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])
        sitk_seg = sitk.ReadImage(self.seg_paths[index])
        if self.args.scribble_sagittal or self.args.scribble_coronal:
            if self.args.scribble_sagittal:
                view = 'sagittal'
            else:
                view = 'coronal'
            sitk_image = self.swap_axis(sitk_image, view=view)
            sitk_label = self.swap_axis(sitk_label, view=view)
            sitk_seg = self.swap_axis(sitk_seg, view=view)

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())
        if sitk_image.GetSpacing() != sitk_label.GetSpacing():
            sitk_label.SetSpacing(sitk_image.GetSpacing())

        if sitk_seg.GetOrigin() != sitk_label.GetOrigin():
            sitk_seg.SetOrigin(sitk_label.GetOrigin())
        if sitk_seg.GetDirection() != sitk_label.GetDirection():
            sitk_seg.SetDirection(sitk_label.GetDirection())
        if sitk_image.GetSpacing() != sitk_seg.GetSpacing():
            sitk_seg.SetSpacing(sitk_image.GetSpacing())
        #
        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
            seg=tio.LabelMap.from_sitk(sitk_seg),
        )

        subject_save = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
            seg=tio.LabelMap.from_sitk(sitk_seg),
        )


        if self.transform:
            try:
                subject = self.transform(subject)
                subject_save = self.transform(subject_save)
            except:
                print(self.image_paths[index])


        if self.pcc:
            subject = self._pcc(subject)


        if subject.label.data.sum() <= self.threshold:
            print(self.image_paths[index], 'label volume too small')
            if self.split == 'train':
                return self.__getitem__(np.random.randint(self.__len__()))
            else:
                return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]

        if self.split == 'train':
            crop_transform = tio.CropOrPad(mask_name='label', target_shape=self.image_size)
            subject = crop_transform(subject)

            # trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach(),
            #                                     "label": subject.label.data.clone().detach()})[0]
            # img_aug, seg_aug = trans_dict["image"], trans_dict["label"]

            trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach()})
            img_aug = trans_dict["image"]

            # print(self.image_paths[index])

            data_output = {'initial_seg': subject.seg.data.clone().detach().float(), 'image_path': self.image_paths[index]}

            return img_aug.float(), subject.label.data.clone().detach().float(), data_output
        else:
            crop_transform = tio.CropOrPad(mask_name='label', target_shape=self.image_size)
            subject = crop_transform(subject)
            subject_save = crop_transform(subject_save)

            trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach()})
            img_aug = trans_dict["image"]

            data_output = {'initial_seg': subject.seg.data.clone().detach().float(),
                           'image_path': self.image_paths[index], 'subject_save': subject_save}

            return img_aug, subject.label.data.clone().detach().float(), data_output




    def _set_file_paths(self, data_dir, split):
        self.image_paths = []
        self.label_paths = []
        self.seg_paths = []
        if self.args.small25:
            split_file = "split_small25.pkl"
        elif self.args.small50:
            split_file = "split_small50.pkl"
        else:
            split_file = "split.pkl"
        dataset_split = os.path.join(data_dir, split_file)

        with open(dataset_split, "rb") as f:
            d = pickle.load(f)[split]

        self.image_paths = [d[i][0] for i in list(d.keys())]
        self.label_paths = [d[i][1] for i in list(d.keys())]
        self.seg_paths = [d[i][2] for i in list(d.keys())]

    def _set_dataset_stat(self):
        self.target_label = 0
        if self.data == 'colon':
            self.intensity_range, self.global_mean, self.global_std = (-57, 175), 65.175035, 32.651197

        elif self.data == 'pancreas':
            self.intensity_range, self.global_mean, self.global_std = (-39, 204), 68.45214, 63.422806
            self.target_label = 2
            if self.args.multi_class:
                self.target_label = 0

        elif self.data == 'lits':
            self.intensity_range, self.global_mean, self.global_std = (-48, 163), 60.057533, 40.198017
            self.target_label = 2

        elif self.data == 'kits':
            self.intensity_range, self.global_mean, self.global_std = (-54, 247), 59.53867, 55.457336
            self.target_label = 2

        elif self.data == 'ultrasound':
            self.intensity_range, self.global_mean, self.global_std = (26, 179), 104.3259, 31.1875



    def _get_train_transforms(self, split):
        # if split == "train":
        #     transforms = Compose(
        #         [
        #             ScaleIntensityRanged(
        #                 keys=["image"],
        #                 a_min=self.intensity_range[0],
        #                 a_max=self.intensity_range[1],
        #                 b_min=self.intensity_range[0],
        #                 b_max=self.intensity_range[1],
        #                 clip=True,
        #             ),
        #             RandCropByPosNegLabeld(
        #                 keys=["image", "label"],
        #                 spatial_size=(128, 128, 128),
        #                 label_key="label",
        #                 pos=2,
        #                 neg=0,
        #                 num_samples=1,
        #             ),
        #             RandShiftIntensityd(keys=["image"], offsets=20, prob=0.5),
        #             NormalizeIntensityd(keys=["image"], subtrahend=self.global_mean, divisor=self.global_std),
        #             RandZoomd(keys=["image", "label"], prob=0.8, min_zoom=0.85, max_zoom=1.25,
        #                       mode=["trilinear", "nearest"]),
        #              ])
        # else:
        transforms = Compose(
            [
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=self.intensity_range[0],
                    a_max=self.intensity_range[1],
                    b_min=self.intensity_range[0],
                    b_max=self.intensity_range[1],
                    clip=True,
                ),
                NormalizeIntensityd(keys=["image"], subtrahend=self.global_mean, divisor=self.global_std),
            ]
        )
        return transforms

    def _binary_label(self, subject):
        label = subject.label.data
        label = (label == self.target_label)
        subject.label.data = label.float()
        return subject

    def _pcc(self, subject):
        print("using pcc setting")
        # crop from random click point
        random_index = torch.argwhere(subject.label.data == 1)
        if (len(random_index) >= 1):
            random_index = random_index[np.random.randint(0, len(random_index))]
            # print(random_index)
            crop_mask = torch.zeros_like(subject.label.data)
            # print(crop_mask.shape)
            crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
            subject.add_image(tio.LabelMap(tensor=crop_mask, affine=subject.label.affine), image_name="crop_mask")
            subject = tio.CropOrPad(mask_name='crop_mask', target_shape=self.image_size)(subject)

        return subject


class Dataloader_promise(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())




