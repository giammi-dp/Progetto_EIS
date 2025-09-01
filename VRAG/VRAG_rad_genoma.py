import os
import faiss
import torch
import json
import nibabel as nib
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class VRAG:
    def __init__(self, query, image_id, dataset_root='../ASNR-MICCAI-BraTS2023-Challenge-TrainingData',
                 top_k=3):
        self.query = query
        self.image_id = image_id
        self.dataset_root = dataset_root
        self.top_k = top_k

        self.report_dict = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",  use_safetensors=True).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_index_path = "image_rad_genoma.index"
        self.text_path = "global_finding.json"

    def load_metadata(self):
        # Caricamento metadati immagini
        with open(self.text_path) as f:
            image_meta = json.load(f)

        self.image_files = [id for id in image_meta.keys()]
        self.case_texts = [image_meta.get(id) for id in self.image_files]


    def _build_image_paths(self, case_dir: str) -> dict:

        paths = {}
        modalities = ["seg", "t1c", "t1n", "t2f", "t2w"]

        for modality in modalities:
            found = [f for f in os.listdir(case_dir)
                     if f.lower().endswith(f"-{modality}.nii.gz") or f.lower().endswith(f"-{modality}.nii")]
            if not found:
                raise FileNotFoundError(f"File for modality '{modality}' not found in {case_dir}")
            paths[modality] = os.path.join(case_dir, found[0])
        return paths

    def get_tumor_slice(self, mri_path: str, seg_path: str, slice_idx: int) -> Image.Image:
        # Caricamento volumi
        mri_img = nib.load(mri_path).get_fdata()
        seg_img = nib.load(seg_path).get_fdata()

        if not slice_idx:
            # Trova la slice che contiene più tumore
            tumor_presence = np.sum(seg_img > 0, axis=(0, 1))
            best_slice_idx = np.argmax(tumor_presence)
        else:
            best_slice_idx = slice_idx

        # Estrazione slice
        image_slice = mri_img[:, :, best_slice_idx]


        p1, p99 = np.percentile(image_slice[image_slice > 0], [1, 99])
        image_slice = np.clip(image_slice, p1, p99)

        # Normalizzazione 0-255
        if np.ptp(image_slice) > 0:
            image_slice = ((image_slice - np.min(image_slice)) / np.ptp(image_slice) * 255).astype(np.uint8)
        else:
            image_slice = np.zeros_like(image_slice, dtype=np.uint8)

        # Conversione a immagine RGB PIL
        pil_image = Image.fromarray(image_slice).convert('RGB')

        return pil_image



    def build_or_load_image_index(self):
        if os.path.exists(self.image_index_path):
            #print("Caricamento Image VectorDB...")
            self.image_index = faiss.read_index(self.image_index_path)
        else:
            #print("Creazione Image VectorDB...")

            images = []
            for img_id in self.image_files:
                image_path = os.path.join(self.dataset_root, img_id)
                paths = self._build_image_paths(image_path)
                img_slice = self.get_tumor_slice(paths["t1c"], paths["seg"], None)
                images.append(img_slice)

            inputs = self.clip_processor(text=self.case_texts, images=images, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                img_feats = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])

                img_feats = torch.nn.functional.normalize(img_feats, p=2, dim=1)


            self.image_embeds = img_feats.cpu().numpy().astype("float32")
            self.image_index = faiss.IndexFlatIP(self.image_embeds.shape[1])
            self.image_index.add(self.image_embeds)
            faiss.write_index(self.image_index, self.image_index_path)

    def retrieve_images_and_text(self, slice_idx):
        #print("Retrieval immagini + testo associato...")

        # Encode query immagine
        query_dir = os.path.join(self.dataset_root, self.image_id)
        paths = self._build_image_paths(query_dir)
        query_image = self.get_tumor_slice(paths['t1c'], paths['seg'], slice_idx)
        inputs = self.clip_processor(text=[self.query], images=[query_image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            query_img_emb = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            query_img_emb = torch.nn.functional.normalize(query_img_emb, p=2, dim=1)
        query_img_emb = query_img_emb.cpu().numpy().astype("float32")

        img_scores, img_indices = self.image_index.search(query_img_emb, self.top_k)

        results = []
        for score, idx in zip(img_scores[0], img_indices[0]):
            img_path = self.image_files[idx]
            case_text = self.case_texts[idx]
            results.append({
                "image_path": img_path,
                "score": score,
                "case_text": case_text
            })
        return results

    def run(self, slice_idx):
        #print(" Avvio VRAG (solo immagini)")
        self.load_metadata()
        self.build_or_load_image_index()
        results = self.retrieve_images_and_text(slice_idx)

        #print("Risultati:")
        '''
        for i, res in enumerate(results):
            print(f" Risultato {i+1} - Score: {res['score']:.4f}")
            print(f"Immagine: {res['image_path']}")


            print(f"Testo associato: {res['case_text'][:400]}...")
        '''
        return results


