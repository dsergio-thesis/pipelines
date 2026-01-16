

import sys
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from skimage import color, measure, transform


if (sys.modules.get('morphometryka') is not None): 
    del sys.modules['morphometryka']
from morphometryka import simple_segmentation, \
    measure_morfometry, \
    centroid_flux


class AddGaussianNoise:
    """Custom transform that adds Gaussian noise to a tensor image."""
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class SegmentationTransform:
    """Custom transform for image segmentation."""
    def __init__(self, nsigma=1.5, min_area=20):
        self.nsigma = nsigma
        self.min_area = min_area

    def __call__(self, tensor):
        # print(f"Applying SegmentationTransform... {tensor.numpy().shape}")

        transformed_channels = []

        all_channels = tensor.numpy().squeeze()
        for c in range(all_channels.shape[0]):
            image = all_channels[c, :, :]
            mask = simple_segmentation(image, nsigma=self.nsigma, min_area=self.min_area)

            transformed_channels.append(image * mask)

        return torch.from_numpy(np.array(transformed_channels))

    def __repr__(self):
        return f"{self.__class__.__name__}"


class MorphometryFeatures:

    def __init__(self):
        pass

    def __call__(self, image):

        # Morphometric features extraction

        all_channels = image.squeeze()

        cash_images = torch.empty((all_channels.shape[0], 4))
        # print(f"cash_images.shape: {cash_images.shape}")

        for c in range(all_channels.shape[0]):

            image = all_channels[c, :, :]

            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()

            mask = simple_segmentation(image)
            res = measure_morfometry(image, mask)

            # print(f"Channel {c} - mask.shape: {mask.shape}")


            cash_images[c, 0] = torch.tensor(res['C'])
            cash_images[c, 1] = torch.tensor(res['A'])
            cash_images[c, 2] = torch.tensor(res['S'])
            cash_images[c, 3] = torch.tensor(res['H'])



        return cash_images


class PolarTransform:
    def __init__(self, output_size=(128, 128)):
        """
        output_size: (radial_steps, angular_steps)
        """
        self.output_size = output_size

    def __call__(self, input_image):
        """
        input_image: input image as a numpy array (H x W) or (H x W x C)
        returns: polar-transformed image as numpy array
        """
        
        transformed_channels = []
        all_channels = input_image.numpy().squeeze()

        for c in range(all_channels.shape[0]):

            image = all_channels[c, :, :]

            # 1. Threshold to find bright regions (Otsu)
            thresh_val = np.mean(image) + np.std(image)
            binary = image > thresh_val

            # 2. Find largest connected component
            labels = measure.label(binary)
            regions = measure.regionprops(labels, intensity_image=image)

            if regions:
                largest_region = max(regions, key=lambda r: r.area)
                cy, cx = largest_region.centroid  # skimage uses (row, col)
                cx = int(cx)
                cy = int(cy)
            else:
                # fallback to center
                cx, cy = image.shape[1] // 2, image.shape[0] // 2

            # 3. Polar transform using skimage
            max_radius = np.sqrt(max(cx, image.shape[1]-cx)**2 + max(cy, image.shape[0]-cy)**2)
            polar_img = transform.warp_polar(
                image, 
                center=(cy, cx), 
                radius=max_radius, 
                output_shape=self.output_size,
                scaling='linear'
            )
            
            transformed_channels.append(polar_img)

        return torch.from_numpy(np.array(transformed_channels))



class CropZeros:
    def __init__(self, output_size=None):
        """
        output_size: (width, height) to resize after cropping zeros.
                     If None, return the cropped area without resizing.
        """
        self.output_size = output_size

    def __call__(self, img):
        # Convert to numpy if tensor
        if isinstance(img, torch.Tensor):
            img_np = img.squeeze(0).cpu().numpy() if img.ndim == 3 and img.shape[0]==1 else img.permute(1,2,0).cpu().numpy()
        else:
            img_np = img

        # For multi-channel images, find pixels where all channels are zero
        if img_np.ndim == 3:
            mask = np.any(img_np != 0, axis=2)
        else:
            mask = img_np != 0

        # If all pixels are zero, return original
        if not np.any(mask):
            cropped = img_np
        else:
            coords = np.argwhere(mask)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1  # slice end is exclusive
            cropped = img_np[y0:y1, x0:x1] if img_np.ndim==2 else img_np[y0:y1, x0:x1, :]

        # Resize to fixed output size if needed
        if self.output_size is not None:
            cropped = transform.resize(
                cropped,
                (self.output_size[1], self.output_size[0]),  # skimage uses (rows, cols)
                anti_aliasing=True
            )

        # Convert back to tensor
        if isinstance(img, torch.Tensor):
            if cropped.ndim == 2:
                cropped = torch.from_numpy(cropped).unsqueeze(0).float()
            else:
                cropped = torch.from_numpy(cropped).permute(2,0,1).float()

        return cropped



class CropAroundCentroid:
    def __init__(self, crop_size=(40, 40)):
        """
        crop_size: (width, height) of the output crop
        """
        self.crop_width, self.crop_height = crop_size

    def __call__(self, input_image):



        transformed_channels = []
        all_channels = input_image.squeeze()

        for c in range(all_channels.shape[0]):

            image = all_channels[c, :, :]

            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()

            # Compute centroid: intensity-weighted center
            mask = simple_segmentation(image)
            cy, cx = centroid_flux(image, mask)

            if cx is None or cy is None or np.isnan(cx) or np.isnan(cy):
                # fallback to image center
                cx, cy = image.shape[1] // 2, image.shape[0] // 2
            else:
                cx = int(cx)
                cy = int(cy)

            # Compute crop boundaries
            y0 = cy - self.crop_height // 2
            y1 = y0 + self.crop_height
            x0 = cx - self.crop_width // 2
            x1 = x0 + self.crop_width

            # Pad if needed
            pad_top = max(0, -y0)
            pad_left = max(0, -x0)
            pad_bottom = max(0, y1 - image.shape[0])
            pad_right = max(0, x1 - image.shape[1])

            y0 = max(0, y0)
            y1 = min(image.shape[0], y1)
            x0 = max(0, x0)
            x1 = min(image.shape[1], x1)

            cropped = image[y0:y1, x0:x1]

            if cropped.ndim == 2:
                cropped = np.pad(
                    cropped,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant'
                )
            elif cropped.ndim == 3:
                cropped = np.pad(
                    cropped,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant'
                )
            else:
                raise ValueError(f"Unexpected image ndim={cropped.ndim}")

            transformed_channels.append(torch.from_numpy(cropped))

        # Stack all transformed channels back into a single tensor
        return torch.stack(transformed_channels) if transformed_channels else input_image
