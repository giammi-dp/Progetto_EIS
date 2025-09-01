from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


import gc
import monai
import os
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


from monai.transforms import (
    LoadImaged, Spacingd, Orientationd,
    ToTensord, Compose, ConcatItemsd, DeleteItemsd, Resized, NormalizeIntensityd
)
from monai.networks.nets import SegResNet
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset
from torchcam.methods import GradCAMpp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Segmentator:
    def __init__(self, image, model):
        self.image = image
        self.model = model

    def _build_image_paths(self):
        """
        Mappa le modalità alle immagini .nii.gz nella cartella del paziente
        """
        paths = {}
        for modality in ["seg", "t1c", "t1n", "t2f", "t2w"]:
            found = [f for f in os.listdir(self.image)
                     if f.lower().endswith(f"-{modality}.nii") or f.lower().endswith(f"-{modality}.nii.gz")]
            if not found:
                raise FileNotFoundError(f"File per la modalità '{modality}' non trovato in {self.image}")
            paths[modality] = os.path.join(self.image, found[0])
        return paths


    def preprocessing_step(self):

        paths = self._build_image_paths()

        keys_input_modalities = ["image_t1c", "image_t1n", "image_t2w", "image_t2f"]
        inference_spatial_size = (240, 240, 160)

        val_transforms = Compose([
            LoadImaged(keys=keys_input_modalities + ["label"]),
            monai.transforms.EnsureChannelFirstd(keys=keys_input_modalities + ["label"]),
            Orientationd(keys=keys_input_modalities + ["label"], axcodes="RAS"),
            Spacingd(keys=keys_input_modalities + ["label"], pixdim=(1.0, 1.0, 1.0),
                     mode=("trilinear", "trilinear", "trilinear", "trilinear", "nearest")),  # Probabilmente necessario
            Resized(keys=keys_input_modalities + ["label"], spatial_size=inference_spatial_size,
                    mode=("trilinear", "trilinear", "trilinear", "trilinear", "nearest")),  # Cruciale per roi_size
            NormalizeIntensityd(keys=keys_input_modalities, nonzero=True, channel_wise=True),
            ConcatItemsd(keys=keys_input_modalities, dim=0, name="image"),
            DeleteItemsd(keys=keys_input_modalities),
            ToTensord(keys=["image", "label"])
        ])

        data_dict = [{
            "image_t1c": paths["t1c"],
            "image_t1n": paths["t1n"],
            "image_t2w": paths["t2w"],
            "image_t2f": paths["t2f"],
            "label": paths["seg"]
        }]
        dataset = Dataset(data=data_dict, transform=val_transforms)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataloader


    def inference(self, dataloader):
        batch = next(iter(dataloader))
        image = batch["image"].to(device)
        self.model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            gc.collect()
            output = self.model(image)

        output_sigmoid = torch.sigmoid(output)
        output_binary = (output_sigmoid > 0.5).int().squeeze(0).cpu().numpy()

        net_mask = output_binary[0]
        ed_mask = output_binary[1]
        et_mask = output_binary[2]

        segmentation = np.zeros_like(net_mask, dtype=np.uint8)
        #print('segmentation shape', segmentation.shape)
        segmentation[et_mask > 0] = 4
        segmentation[net_mask > 0] = 1
        segmentation[ed_mask > 0] = 2

        final_segmentation_from_lambda = np.zeros_like(et_mask, dtype=np.uint8)

        final_segmentation_from_lambda[ed_mask > 0] = 2

        final_segmentation_from_lambda[net_mask > 0] = 1

        final_segmentation_from_lambda[et_mask > 0] = 4

        segmentation = final_segmentation_from_lambda
        #print('segmentation shape', segmentation.shape)
        return segmentation, batch

    def generate_heatmap(self, batch):
        image = batch["image"].to(device)
        self.model.eval()

        target_layer = self.model.up_layers[2][0].conv1.conv


        cam_extractor = GradCAMpp(self.model, target_layer=target_layer)

        image.requires_grad = True
        with torch.set_grad_enabled(True):
            torch.cuda.empty_cache()
            gc.collect()
            output = self.model(image)

        class_idx = 2
        torch.cuda.empty_cache()
        gc.collect()
        heatmap = cam_extractor(class_idx, output)[0].squeeze().cpu().numpy()


        cam_extractor.remove_hooks()

        return heatmap

    def plot_slice(self, segmentation, batch):
        display_channel_idx = 0
        tumor_voxel_counts = np.sum(segmentation > 0, axis=(0, 1))

        slice_idx = np.argmax(tumor_voxel_counts)
        #print(f"Slice selezionata automaticamente: {slice_idx}, con {tumor_voxel_counts[slice_idx]} voxels tumorali.")


        image = batch["image"].to(device)
        input_slice = image[0, display_channel_idx, :, :, slice_idx].detach().cpu().numpy()
        label = batch["label"].to(device)
        label_slice = label[0, display_channel_idx, :, :, slice_idx].detach().cpu().numpy()
        mask_slice = segmentation[:, :, slice_idx]

        fig = Figure(figsize=(4, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(input_slice, cmap="gray")
        tumor_mask = mask_slice != 0

        masked_tumor = np.where(tumor_mask, mask_slice, np.nan)

        ax.imshow(masked_tumor, cmap="jet", alpha=0.6)

        ax.axis("off")
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        argb = np.frombuffer(canvas.tostring_argb(), dtype='uint8').reshape(int(height), int(width), 4)
        image_array = argb[:, :, 1:]  # Rimuove il canale alpha


        '''
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(input_slice, cmap="gray")
        plt.title(f"Original MRI Slice (Channel {display_channel_idx})")

        plt.subplot(1, 3, 2)
        plt.imshow(input_slice, cmap="gray")  # Mostra l'immagine originale sotto
        plt.imshow(mask_slice, cmap="jet", alpha=0.6)  # Sovrapponi la maschera
        plt.title("Tumor Segmentation")



        plt.subplot(1, 3, 3)
        plt.imshow(label_slice, cmap="gray")
        plt.title("Label Segmentation")

        plt.tight_layout()
        plt.show()
        '''
        return image_array, slice_idx


    def save_mask(self, paths, segmentation, name):
        #print("\n--- Salvataggio della maschera segmentata")

        original_image_nifti = nib.load(paths["t1c"])

        transposed_segmentation = np.transpose(segmentation, (1, 2, 0))
        #print(f"Shape della maschera trasposta per salvataggio: {transposed_segmentation.shape}")

        target_depth_orig = original_image_nifti.shape[2]
        if transposed_segmentation.shape[2] > target_depth_orig:
            start_d = (transposed_segmentation.shape[2] - target_depth_orig) // 2
            end_d = start_d + target_depth_orig
            final_segmentation_for_save = transposed_segmentation[:, :, start_d:end_d]
        else:
            final_segmentation_for_save = transposed_segmentation

        # Save the segmented mask
        predicted_nifti = nib.Nifti1Image(final_segmentation_for_save.astype(np.uint8),
                                          original_image_nifti.affine,
                                          original_image_nifti.header)
        nib.save(predicted_nifti, f"{name}_seg.nii.gz")
        #print("Maschera segmentata salvata come predicted_segmentation_brats.nii.gz")


    def calculate_volume(self, segmentation, batch):
        #print("\n ---Volumi della segmentazione---")

        tumor_voxels = np.where(segmentation != 0)  # Use the D,H,W segmented mask
        num_tumor_voxels = len(tumor_voxels[0])
        voxel_volume_mm3 = 1.0 * 1.0 * 1.0  # Assuming MONAI Spacingd to 1mm isovoxel
        total_tumor_volume_seg = num_tumor_voxels * voxel_volume_mm3
        #print(f"Volume totale stimato del tumore (basato su spacing 1x1x1mm): {total_tumor_volume_seg:.2f} mm^3")
        #print(f"Volume totale stimato del tumore (basato su spacing 1x1x1mm): {total_tumor_volume_seg / 1000:.2f} cm^3")

        # Detailed sub-region volumes:
        volume_net_seg = np.sum(segmentation == 1) * voxel_volume_mm3
        volume_ed_seg = np.sum(segmentation == 2) * voxel_volume_mm3
        volume_et_seg = np.sum(segmentation == 4) * voxel_volume_mm3  # Class 4 after re-mapping
        #print(f"Volume NET/NCR: {volume_net_seg:.2f} mm^3")
        #print(f"Volume ED: {volume_ed_seg:.2f} mm^3")
        #print(f"Volume ET: {volume_et_seg:.2f} mm^3")

        #print("\n ---Volumi della label---")
        label_preprocessed = batch["label"]
        label_np = label_preprocessed.squeeze().cpu().numpy()

        total_tumor_volume_gt = np.sum(label_np != 0) * voxel_volume_mm3
        #print(f"\nVolume totale del tumore (Ground Truth): {total_tumor_volume_gt:.2f} mm^3")
        #print(f"Volume totale del tumore (Ground Truth): {total_tumor_volume_gt / 1000:.2f} cm^3")

        volume_net_gt = np.sum(label_np == 1) * voxel_volume_mm3
        volume_ed_gt = np.sum(label_np == 2) * voxel_volume_mm3
        volume_et_gt = np.sum(label_np == 4) * voxel_volume_mm3
        #print(f"Volume NET/NCR (Ground Truth): {volume_net_gt:.2f} mm^3")
        #print(f"Volume ED (Ground Truth): {volume_ed_gt:.2f} mm^3")
        #print(f"Volume ET (Ground Truth): {volume_et_gt:.2f} mm^3")

        return total_tumor_volume_seg, volume_net_seg, volume_ed_seg, volume_et_seg, total_tumor_volume_gt, volume_net_gt, volume_ed_gt, volume_et_gt


    def bounding_box(self, segmentation, paths):
        tumor_mask = (segmentation != 0)
        coords = np.argwhere(tumor_mask)

        if coords.size == 0:
            return None, None, None  # maschera vuota

        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        coords_x = (xmin, xmax)
        coords_z = (zmin, zmax)
        bbox = (zmin, zmax, ymin, ymax, xmin, xmax)
        #print("Bounding box tumore (voxel):", bbox)

        z_center_voxel = (coords_z[0] + coords_z[1]) / 2
        y_center_voxel = (bbox[2] + bbox[3]) / 2
        x_center_voxel = (coords_x[0] + coords_x[1]) / 2
        center_voxel = np.array([z_center_voxel, y_center_voxel, x_center_voxel, 1.0])  # omogenea

        # Step 4: conversione in coordinate fisiche (mm)
        original_image_nifti = nib.load(paths["t1c"])
        center_mm = original_image_nifti.affine @ center_voxel
        x_mm, y_mm, z_mm = center_mm[:3]
        #print(f"Centro tumore in coordinate spaziali (mm): x={x_mm:.2f}, y={y_mm:.2f}, z={z_mm:.2f}")

        # Step 5: stima emisfero/lobo con coordinate reali
        hemisphere = "left" if x_mm < 0 else "right"  # 0 = piano sagittale
        if z_mm > 40:
            lobe = "frontal"
        elif z_mm > 10:
            lobe = "parietal"
        elif z_mm > -20:
            lobe = "temporal"
        else:
            lobe = "occipital"

        return hemisphere, lobe
