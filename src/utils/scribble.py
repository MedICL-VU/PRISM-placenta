from typing import Union, Tuple, List, Optional

import numpy as np
import torch

import kornia
import cv2
import src.voxynth.noise as v_noise
import src.voxynth.transform as v_transform

import os

# Prevent neurite from trying to load tensorflow
os.environ['NEURITE_BACKEND'] = 'pytorch'


# -----------------------------------------------------------------------------
# Parent class
# -----------------------------------------------------------------------------

class WarpScribble:
    """
    Parent scribble class with shared functions for generating noise masks (useful for breaking up scribbles) and applying deformation fields (to warp scribbles)
    """

    def __init__(self,
                 warp: bool = True,
                 warp_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 warp_magnitude: Union[int, Tuple[int], List[int]] = (1, 6),
                 mask_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 ):
        if isinstance(warp_smoothing, int):
            warp_smoothing = [warp_smoothing, warp_smoothing]
        if isinstance(warp_magnitude, int):
            warp_magnitude = [warp_magnitude, warp_magnitude]
        # Warp settings
        self.warp = warp
        self.warp_smoothing = list(warp_smoothing)
        self.warp_magnitude = list(warp_magnitude)
        # Noise mask settings
        self.mask_smoothing = mask_smoothing

    def noise_mask(self, shape: Union[Tuple[int], List[int]] = (8, 128, 128), device=None):
        """
        Get a random binary mask by thresholding smoothed noise. The mask is used to break up the scribbles
        """
        if isinstance(self.mask_smoothing, tuple):
            get_smoothing = lambda: np.random.uniform(*self.mask_smoothing)
        else:
            get_smoothing = lambda: self.mask_smoothing

        noise = torch.stack([
            v_noise.perlin(shape=shape[-2:], smoothing=get_smoothing(), magnitude=1, device=device) for _ in
            range(shape[0])
        ])  # shape: b x H x W
        noise_mask = (noise > 0.0).int().unsqueeze(1)

        return noise_mask  # shaoe: b x 1 x H x W

    def apply_warp(self, x: torch.Tensor):
        """
        Warp a given mask x using a random deformation field
        """
        if x.sum() > 0:
            # warp scribbles using a deformation field
            deformation_field = v_transform.random_transform(
                shape=x.shape[-2:],
                affine_probability=0.0,
                warp_probability=1.0,
                warp_integrations=0,
                warp_smoothing_range=self.warp_smoothing,
                warp_magnitude_range=self.warp_magnitude,
                voxsize=1,
                device=x.device,
                isdisp=False
            )

            warped = v_transform.spatial_transform(x, trf=deformation_field, isdisp=False)
            if warped.sum() == 0:
                return x
            else:
                return (warped - warped.min()) / (warped.max() - warped.min())
        else:
            # Don't need to warp if mask is empty
            return x

    def batch_scribble(self, mask: torch.Tensor, n_scribbles: int = 1):
        """
        Simulate scribbles for a batch of examples (mask).
        """
        raise NotImplementedError

    def __call__(self, mask: torch.Tensor, n_scribbles: int = 1) -> torch.Tensor:
        """
        Args:
            mask: (b,1,H,W) or (1,H,W) mask in [0,1] to sample scribbles from
        Returns:
            scribble_mask: (b,1,H,W) or (1,H,W) mask(s) of scribbles on [0,1]
        """
        assert len(mask.shape) in [3, 4], f"mask must be b x 1 x h x w or 1 x h x w. currently {mask.shape}"

        if len(mask.shape) == 3:
            # shape: 1 x h x w
            return self.batch_scribble(mask[None, ...], n_scribbles=n_scribbles)[0, ...]
        else:
            # shape: b x 1 x h x w
            return self.batch_scribble(mask, n_scribbles=n_scribbles)


# -----------------------------------------------------------------------------
# Line Scribbles
# -----------------------------------------------------------------------------

class LineScribble(WarpScribble):
    """
    Generates scribbles by
        1) drawing lines connecting random points on the mask
        2) warping with a random deformation field
        3) then correcting any scribbles outside the mask
        5) optionally, limiting the max area of scribbles to k pixels
    """

    def __init__(self,
                 # Warp settings
                 warp: bool = True,
                 warp_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 warp_magnitude: Union[int, Tuple[int], List[int]] = (1, 6),
                 mask_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 # Line scribble settings
                 thickness: int = 1,
                 preserve_scribble: bool = True,  # if True, prevents empty scribble masks from being returned
                 max_pixels: Optional[int] = None,  # per "scribble"
                 max_pixels_smooth: Optional[int] = 42,
                 # Viz
                 show: bool = False
                 ):

        super().__init__(
            warp=warp,
            warp_smoothing=warp_smoothing,
            warp_magnitude=warp_magnitude,
            mask_smoothing=mask_smoothing,
        )
        self.thickness = thickness
        self.preserve_scribble = preserve_scribble
        self.max_pixels = max_pixels
        self.max_pixels_smooth = max_pixels_smooth
        self.show = show

    def batch_scribble(self, mask: torch.Tensor, n_scribbles: int = 1) -> torch.Tensor:
        """
        Args:
            mask: (b,1,H,W) mask in [0,1] to sample scribbles from
            n_scribbles: number of line scribbles to sample initially
        Returns:
            scribble_mask: (b,1,H,W) mask(s) of scribbles in [0,1]
        """
        bs = mask.shape[0]

        # Points to sample line endpoints from
        points = torch.nonzero(mask[:, 0, ...])

        def sample_lines(indices):

            image = np.zeros(mask.shape[-2:] + (1,))

            if len(indices) > 0:
                # Sample points for each example in the batch
                idx = np.random.randint(low=0, high=len(indices), size=2 * n_scribbles)
                endpoints = points[indices, 1:][idx, 0, ...]
                # Flip order of coordinates to be xy
                endpoints = torch.flip(endpoints, dims=(1,)).cpu().numpy()
                # Draw lines between the sample points
                for i in range(n_scribbles):

                    image = cv2.line(image, tuple(endpoints[i * 2]), tuple(endpoints[i * 2 + 1]), color=1,
                                     thickness=1)

            return torch.from_numpy(image)  # shape: H x W x 1

        scribbles = torch.stack([
            sample_lines(torch.argwhere(points[:, 0] == i)) for i in range(bs)
        ]).to(mask.device).moveaxis(-1, 1).float()  # shape: b x 1 x H x W

        if self.warp:
            warped_scribbles = torch.stack(
                [self.apply_warp(scribbles[b, ...]) for b in range(bs)])  # shape: b x 1 x H x W
        else:
            warped_scribbles = scribbles

        # Remove lines outside the mask
        corrected_warped_scribbles = mask * warped_scribbles

        if self.preserve_scribble:
            # If none of the scribble falls in the mask after warping, undo warping
            idx = torch.where(torch.sum(corrected_warped_scribbles, dim=(1, 2, 3)) == 0)
            corrected_warped_scribbles[idx] = mask[idx] * scribbles[idx]

        if self.max_pixels is not None:

            noise = torch.stack([
                v_noise.perlin(shape=mask.shape[-2:], smoothing=self.max_pixels_smooth, magnitude=1,
                                     device=mask.device) for _ in range(bs)
            ]).unsqueeze(1)  # shape: b x 1 x H x W

            # Shift all noise to be positive
            if noise.min() < 0:
                noise = noise - noise.min()

            # Get the top k pixels
            flat_mask = (noise * corrected_warped_scribbles).view(bs, -1)
            vals, idx = flat_mask.topk(k=(self.max_pixels * n_scribbles), dim=1)

            binary_mask = torch.zeros_like(flat_mask)
            binary_mask.scatter_(dim=1, index=idx, src=torch.ones_like(flat_mask))

            corrected_warped_scribbles = binary_mask.view(*mask.shape) * corrected_warped_scribbles

        if self.show:

            import neurite as ne
            import matplotlib.pyplot as plt
            from .plot import show_scribbles

            if self.max_pixels is not None:
                binary_mask = binary_mask.reshape(*mask.shape)
                tensors = [mask, scribbles, warped_scribbles, noise, binary_mask, corrected_warped_scribbles, mask]
                titles = ["Mask", "Lines", "Warped Lines", 'Smooth Noise', 'Top k Pixels', 'Corrected Scribbles',
                          'Corrected Scribbles']
            else:
                tensors = [mask, scribbles, warped_scribbles, corrected_warped_scribbles, mask]
                titles = ["Mask", "Lines", "Warped Lines", 'Corrected Scribbles', 'Corrected Scribbles']

            fig, axes = ne.plot.slices(
                sum([[x[i, 0, ...].cpu() for x in tensors] for i in range(bs)], []),
                sum([titles for _ in range(bs)], []),
                show=False, grid=(bs, len(titles)), width=3 * len(titles), do_colorbars=False
            )

            if bs > 1:
                for i in range(bs):
                    show_scribbles(corrected_warped_scribbles[i, 0, ...].cpu(), axes[i, -1])
            else:
                show_scribbles(corrected_warped_scribbles[0, 0, ...].cpu(), axes[-1])
            plt.show()

        return corrected_warped_scribbles  # b x 1 x H x W


# -----------------------------------------------------------------------------
# Median Axis Scribble
# -----------------------------------------------------------------------------

class CenterlineScribble(WarpScribble):
    """
    Generates scribbles by
        1) skeletonizing the mask
        2) chopping up with a random noise mask
        3) warping with a random deformation field
        4) then correcting any scribbles that fall outside the mask
        5) optionally, limiting the max area of scribbles to k pixels
    """

    def __init__(self,
                 # Warp settings
                 warp: bool = True,
                 warp_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 warp_magnitude: Union[int, Tuple[int], List[int]] = (1, 6),
                 mask_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 # Thickness of skeleton
                 break_mask: bool = True,
                 dilate_kernel_size: Optional[int] = None,
                 preserve_scribble: bool = True,  # if True, prevents empty scribble masks from being returned
                 max_pixels: Optional[int] = None,  # per "scribble"
                 max_pixels_smooth: int = 42,
                 # Viz
                 show: bool = False,
                 save_data = None
                 ):

        super().__init__(
            warp=warp,
            warp_smoothing=warp_smoothing,
            warp_magnitude=warp_magnitude,
            mask_smoothing=mask_smoothing,
        )
        self.dilate_kernel_size = dilate_kernel_size
        self.preserve_scribble = preserve_scribble
        self.max_pixels = max_pixels
        self.max_pixels_smooth = max_pixels_smooth
        self.show = show

        self.break_mask = break_mask
        self.save_data = save_data
    def batch_scribble(self, mask: torch.Tensor, n_scribbles: Optional[int] = 1):
        """
        Simulate scribbles for a batch of examples.
        Args:
            mask: (b,1,H,W) mask in [0,1] to sample scribbles from. torch.int32
            n_scribbles: (int) only used when max_pixels is set as a multiplier for total area of the scribbles
                currently, this argument does not control the number of components in the scribble mask
        Returns:
            scribble_mask: (b,1,H,W) mask(s) of scribbles in [0,1]
        """
        assert len(mask.shape) == 4, f"mask must be b x 1 x h x w. currently {mask.shape}"
        bs = mask.shape[0]

        mask_w_border = 255 * mask.clone().moveaxis(1, -1)
        mask_w_border[:, :, 0, :] = 0
        mask_w_border[:, :, -1, :] = 0
        mask_w_border[:, 0, :, :] = 0
        mask_w_border[:, -1, :, :] = 0

        # Skeletonize the mask
        skeleton = torch.from_numpy(
            np.stack([
                cv2.ximgproc.thinning(mask_w_border[i, ...].cpu().numpy().astype(np.uint8)) / 255 for i in range(bs)
            ])
        ).squeeze(-1).unsqueeze(1).to(mask.device).float()  # shape: b x 1 x H x W

        if self.save_data is not None:
            import nibabel as nib
            save_skeleton = skeleton.permute(1, 0, 2, 3).cpu().numpy()[0, :]
            new_I = nib.Nifti1Image(save_skeleton, self.save_data.affine, self.save_data.header)
            nib.save(new_I, '/home/hao/Hao/PRISM_ultrasound/src/post_processing/generate_scribble/positive_scribble_0_skeleton.nii.gz')
            print('skeleton save')


        if self.dilate_kernel_size is not None:
            # Dilate the boundary to make it thicker
            #k = _as_single_val(self.dilate_kernel_size)
            k = 3
            if k > 0:
                kernel = torch.ones((k, k), device=mask.device)
                dilated_skeleton = kornia.morphology.dilation(skeleton, kernel=kernel, engine='convolution')
        else:
            dilated_skeleton = skeleton

        noise_mask = self.noise_mask(shape=mask.shape, device=mask.device)

        if self.save_data is not None:
            import nibabel as nib
            save_noise_mask = noise_mask.permute(1, 0, 2, 3).cpu().numpy()[0, :]
            new_I = nib.Nifti1Image(save_noise_mask, self.save_data.affine, self.save_data.header)
            nib.save(new_I,
                     '/home/hao/Hao/PRISM_ultrasound/src/post_processing/generate_scribble/positive_scribble_0_noise_mask.nii.gz')
            print('noise_mask save')

        # Break up the boundary contours
        #scribbles = (dilated_skeleton * noise_mask)  # shape: b x 1 x H x W
        if self.break_mask:
            scribbles = dilated_skeleton * noise_mask  # shape: b x 1 x H x W
        else:
            scribbles = dilated_skeleton

        if self.save_data is not None:
            import nibabel as nib
            save_scribbles = scribbles.permute(1, 0, 2, 3).cpu().numpy()[0, :]
            new_I = nib.Nifti1Image(save_scribbles, self.save_data.affine, self.save_data.header)
            nib.save(new_I,
                     '/home/hao/Hao/PRISM_ultrasound/src/post_processing/generate_scribble/positive_scribble_0_break_scribbles.nii.gz')
            print('break scribbles save')

        if self.preserve_scribble:
            # If none of the scribbles fall in the random mask, keep the whole scribble
            idx = torch.where(torch.sum(scribbles, dim=(1, 2, 3)) == 0)
            scribbles[idx] = skeleton[idx]

        if self.warp:
            warped_scribbles = torch.stack([self.apply_warp(scribbles[b, ...]) for b in range(bs)])
        else:
            warped_scribbles = scribbles

        corrected_warped_scribbles = mask * warped_scribbles  # shape: b x 1 x H x W



        if self.preserve_scribble:
            # If none of the scribble falls in the mask after warping, remove the warping
            idx = torch.where(torch.sum(corrected_warped_scribbles, dim=(1, 2, 3)) == 0)
            corrected_warped_scribbles[idx] = mask[idx] * scribbles[idx]

        if self.max_pixels is not None:

            noise = torch.stack([
                v_noise.perlin(shape=mask.shape[-2:], smoothing=self.max_pixels_smooth, magnitude=1,
                                     device=mask.device) for _ in range(bs)
            ]).unsqueeze(1)  # shape: b x 1 x H x W

            # Shift all noise mask to be positive
            if noise.min() < 0:
                noise = noise - noise.min()

            flat_mask = (noise * corrected_warped_scribbles).view(bs, -1)
            vals, idx = flat_mask.topk(k=(self.max_pixels * n_scribbles), dim=1)

            binary_mask = torch.zeros_like(flat_mask)
            binary_mask.scatter_(dim=1, index=idx, src=torch.ones_like(flat_mask))

            corrected_warped_scribbles = binary_mask.view(*mask.shape) * corrected_warped_scribbles

        if self.show:

            import neurite as ne
            from .plot import show_scribbles
            import matplotlib.pyplot as plt

            tensors = [mask, skeleton]
            titles = ["Input Mask", "Skeleton"]

            if self.dilate_kernel_size is not None:
                tensors.append(dilated_skeleton)
                titles.append('Dilated Skeleton')

            if self.max_pixels is not None:
                tensors += [noise_mask, scribbles, warped_scribbles, noise, binary_mask.reshape(*mask.shape),
                            corrected_warped_scribbles, mask]
                titles += ["Noise Mask", 'Broken Skeleton', 'Warped Scribbles', 'Smooth Noise', 'Top k Pixels',
                           'Corrected Scribbles', 'Corrected Scribbles']
            else:
                tensors += [noise_mask, scribbles, warped_scribbles, corrected_warped_scribbles, mask]
                titles += ["Noise Mask", 'Broken Skeleton', 'Warped Scribbles', 'Corrected Scribbles',
                           'Corrected Scribbles']

            fig, axes = ne.plot.slices(
                sum([[x[i, ...].squeeze().cpu() for x in tensors] for i in range(bs)], []),
                sum([titles for _ in range(bs)], []),
                show=False, grid=(bs, len(titles)), width=3 * len(titles)
            )

            if bs > 1:
                for i in range(bs):
                    show_scribbles(corrected_warped_scribbles[i, 0, ...].cpu(), axes[i, -1])
            else:
                show_scribbles(corrected_warped_scribbles[0, 0, ...].cpu(), axes[-1])

            plt.show()

        return corrected_warped_scribbles

    # -----------------------------------------------------------------------------


# Contour Scribbles
# -----------------------------------------------------------------------------

class ContourScribble(WarpScribble):
    """
    Generates scribbles by
        1) blurring and thresholding the mask, then getting the contours
        2) chopping up the contour scribbles with a random noise mask
        3) warping with a random deformation field
        4) then correcting any scribbles that fall outside the mask
        5) optionally, limiting the max area of scribbles to k pixels
    """

    def __init__(self,
                 # Warp settings
                 warp: bool = True,
                 warp_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 warp_magnitude: Union[int, Tuple[int], List[int]] = (1, 6),
                 mask_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 # Blur settings
                 blur_mask: bool = True,
                 break_mask: bool = True,
                 blur_kernel_size: int = 33,
                 blur_sigma: Union[float, Tuple[float], List[float]] = (5.0, 20.0),
                 # Other settings
                 dilate_kernel_size: Optional[Union[int, Tuple[int]]] = None,
                 preserve_scribble: bool = True,  # if True, prevents empty scribble masks from being returned
                 max_pixels: Optional[int] = None,  # per "scribble"
                 max_pixels_smooth: Optional[int] = 42,
                 # Viz
                 show: bool = False,
                 save_data = None
                 ):

        super().__init__(
            warp=warp,
            warp_smoothing=warp_smoothing,
            warp_magnitude=warp_magnitude,
            mask_smoothing=mask_smoothing,
        )

        # Blur settings
        if isinstance(blur_sigma, float) or isinstance(blur_sigma, int):
            blur_sigma = (blur_sigma, blur_sigma + 1e-7)

        self.blur_fn = kornia.augmentation.RandomGaussianBlur(
            kernel_size=(blur_kernel_size, blur_kernel_size), sigma=blur_sigma, p=1.
        )
        # Line thickness
        self.dilate_kernel_size = dilate_kernel_size
        # Corrections
        self.preserve_scribble = preserve_scribble
        self.max_pixels = max_pixels
        self.max_pixels_smooth = max_pixels_smooth
        # Viz
        self.show = show

        self.blur_mask = blur_mask
        self.break_mask = break_mask
        self.save_data = save_data
    def batch_scribble(self, mask: torch.Tensor, n_scribbles: Optional[int] = 1):
        """
        Args:
            mask: (b,1,H,W) mask in [0,1] to sample scribbles from
            n_scribbles: (int) only used when max_pixels is set as a multiplier for total area of the scribbles
                currently, this argument does not control the number of components in the scribble mask
        Returns:
            scribble_mask: (b,1,H,W) mask(s) of scribbles in [0,1]
        """
        assert len(mask.shape) == 4, f"mask must be b x 1 x h x w. currently {mask.shape}"
        bs = mask.shape[0]

        rev_mask = (1 - mask)
        blur_mask = self.blur_fn(rev_mask)
        corrected_blur_mask = torch.reshape(torch.maximum(blur_mask, rev_mask), (bs, -1))

        # Randomly sample a threshold for each example
        min_bs = corrected_blur_mask.min(1)[0].cpu().numpy()
        binary_mask = (torch.reshape(mask, (bs, -1)) > 0) * corrected_blur_mask
        max_bs = torch.reshape(binary_mask, (bs, -1)).max(1)[0].cpu().numpy()
        thresh = torch.from_numpy(np.random.uniform(min_bs, max_bs, size=bs)).to(mask.device)

        # Apply threshold
        thresh = thresh[..., None].repeat(1, mask.shape[-2] * mask.shape[-1])
        binary_blur_mask = (corrected_blur_mask <= thresh).view(mask.shape).float()

        if self.save_data is not None:
            import nibabel as nib
            a = binary_blur_mask.permute(1, 2, 0, 3).cpu().numpy()[0, :]
            new_I = nib.Nifti1Image(a, self.save_data.affine, self.save_data.header)
            nib.save(new_I, '/home/hao/Hao/PRISM_ultrasound/src/post_processing/generate_scribble/positive_scribble_0_binary_blur_mask.nii.gz')
            print('binary_blur_mask save')

        # Use filter to get contours

        if self.blur_mask:
            _, boundary = kornia.filters.canny(binary_blur_mask, hysteresis=False)
        else:
            _, boundary = kornia.filters.canny(mask, hysteresis=False)

        if self.save_data is not None:
            import nibabel as nib
            a = boundary.permute(1, 2, 0, 3).cpu().numpy()[0, :]
            new_I = nib.Nifti1Image(a, self.save_data.affine, self.save_data.header)
            nib.save(new_I, '/home/hao/Hao/PRISM_ultrasound/src/post_processing/generate_scribble/positive_scribble_0_boundary.nii.gz')
            print('boundary save')

        if self.dilate_kernel_size is not None:
            # Dilate the boundary to make it thicker
            #k = _as_single_val(self.dilate_kernel_size)
            k = 0
            if k > 0:
                kernel = torch.ones((k, k), device=boundary.device)
                dilated_boundary = kornia.morphology.dilation(boundary, kernel=kernel, engine='convolution')
            else:
                dilated_boundary = boundary
        else:
            dilated_boundary = boundary

        # Get noise mask to break up the contours
        noise_mask = self.noise_mask(shape=mask.shape, device=mask.device)

        if self.save_data is not None:
            import nibabel as nib
            a = noise_mask.permute(1, 2, 0, 3).cpu().numpy()[0, :]
            new_I = nib.Nifti1Image(a, self.save_data.affine, self.save_data.header)
            nib.save(new_I, '/home/hao/Hao/PRISM_ultrasound/src/post_processing/generate_scribble/positive_scribble_0_noise_mask.nii.gz')
            print('noise_mask save')

        # Break up the boundary contours
        if self.break_mask:
            scribbles = dilated_boundary * noise_mask  # shape: b x 1 x H x W
        else:
            scribbles = dilated_boundary
        if self.save_data is not None:
            import nibabel as nib
            a = scribbles.permute(1, 2, 0, 3).cpu().numpy()[0, :]
            new_I = nib.Nifti1Image(a, self.save_data.affine, self.save_data.header)
            nib.save(new_I, '/home/hao/Hao/PRISM_ultrasound/src/post_processing/generate_scribble/positive_scribble_0_break_scribbles.nii.gz')
            print('break_scribbles save')

        if self.preserve_scribble:
            # If none of the scribbles fall in the noise mask, keep the whole scribble
            idx = torch.where(torch.sum(scribbles, dim=(1, 2, 3)) == 0)[0]
            scribbles[idx, ...] = dilated_boundary[idx, ...]

        if self.warp:
            warped_scribbles = torch.stack([self.apply_warp(scribbles[b, ...]) for b in range(bs)])
        else:
            warped_scribbles = scribbles

        # Remove scribbles that are outside the mask
        corrected_warped_scribbles = mask * warped_scribbles

        if self.preserve_scribble:
            # If none of the scribble falls in the mask after warping, remove the warping
            idx = torch.where(torch.sum(corrected_warped_scribbles, dim=(1, 2, 3)) == 0)[0]
            corrected_warped_scribbles[idx, ...] = mask[idx, ...] * scribbles[idx, ...]

        if self.max_pixels is not None:

            noise = torch.stack([
                v_noise.perlin(shape=mask.shape[-2:], smoothing=self.max_pixels_smooth, magnitude=1,
                                     device=mask.device) for _ in range(bs)
            ]).unsqueeze(1)  # shape: b x 1 x H x W

            # Shift noise mask to be positive
            if noise.min() < 0:
                noise = noise - noise.min()

            flat_mask = (noise * corrected_warped_scribbles).view(bs, -1)
            vals, idx = flat_mask.topk(k=(self.max_pixels * n_scribbles), dim=1)

            binary_mask = torch.zeros_like(flat_mask)
            binary_mask.scatter_(dim=1, index=idx, src=torch.ones_like(flat_mask))

            corrected_warped_scribbles = binary_mask.view(*mask.shape) * corrected_warped_scribbles

        return corrected_warped_scribbles


if __name__ == '__main__':
    import torch
    import torch.distributed as dist
    import numpy as np
    import random
    from torch.backends import cudnn


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

    import SimpleITK as sitk
    import torchio as tio

    init_seeds()
    mask_path = '/home/hao/Hao/data/0SAM_data/Task10_colon/labelsTr/colon_001.nii.gz'
    sitk_label = sitk.ReadImage(mask_path)
    subject = tio.Subject(
        image=tio.ScalarImage.from_sitk(sitk_label),
        label=tio.LabelMap.from_sitk(sitk_label),
    )
    transforms_list = [
        tio.Resample(1), ]
    transforms_list.append(tio.CropOrPad(mask_name='label', target_shape=(128,64,96)))
    transforms = tio.Compose(transforms_list)
    subject = transforms(subject)
    mask = subject.label.data.clone().detach()

    # mask[:, :, :, 39] = 0
    # mask[:, 68:72, 26:30, 39] = 1
    # mask[mask > 0] = 0
    # print(torch.unique(mask))

    sitk_label_new = mask.squeeze(0)
    result_image = sitk.GetImageFromArray(sitk_label_new)
    sitk.WriteImage(result_image, './maskresult.nii.gz')


    mask = mask.permute(3, 0, 1, 2)
    #mask = (torch.ones_like(mask) - mask).permute(3, 0, 1, 2)
    import time
    a = time.time()
    #CenterlineScribble
    #LineScribble
    ContourScribble_mask = LineScribble().batch_scribble(mask)
    print(time.time() - a )


    sitk_label_new = ContourScribble_mask.permute(1, 2, 3, 0).squeeze(0)
    sitk_label_new[sitk_label_new > 0] = 1
    result_image = sitk.GetImageFromArray(sitk_label_new)
    sitk.WriteImage(result_image, './result.nii.gz')
    print(1)