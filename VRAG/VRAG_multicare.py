import os
import faiss
import torch
import json
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class VRAG:
    def __init__(self, query, image_path, dataset_root='../medical_datasets/brain_tumor_multimodal',
                 top_k=3, use_gpu=True):
        self.query = query
        self.image_path = image_path
        self.dataset_root = dataset_root
        self.top_k = top_k
        self.use_gpu = use_gpu and torch.cuda.is_available()

        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_index_path = os.path.join(dataset_root, "image_multicare.index")
        self.metadata_path = os.path.join(dataset_root, "image_metadata.json")
        self.cases_path = os.path.join(dataset_root, "cases.csv")

    def load_metadata(self):
        # Caricamento metadati immagini
        with open(self.metadata_path) as f:
            self.image_meta = [json.loads(line) for line in f if line.strip()]

        self.image_files = [item.get('file_path') for item in self.image_meta]
        self.captions = [item.get('caption') for item in self.image_meta]
        self.case_ids = [item.get('case_id') for item in self.image_meta]

        # Caricamento mappa case_id → case_text
        df = pd.read_csv(self.cases_path)
        self.case_text_map = dict(zip(df['case_id'], df['case_text'].fillna('')))

    def build_or_load_image_index(self):
        if os.path.exists(self.image_index_path):
            #print("Caricamento Image VectorDB...")
            self.image_index = faiss.read_index(self.image_index_path)
        else:
            #print("Creazione Image VectorDB...")
            images = [Image.open(p).convert("RGB") for p in self.image_files]
            inputs = self.clip_processor(text=self.captions, images=images, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                img_feats = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])

            img_feats = torch.nn.functional.normalize(img_feats, p=2, dim=1)
            self.image_embeds = img_feats.cpu().numpy().astype("float32")
            self.image_index = faiss.IndexFlatIP(self.image_embeds.shape[1])
            self.image_index.add(self.image_embeds)
            faiss.write_index(self.image_index, self.image_index_path)

    def retrieve_images_and_text(self):
        #print("Retrieval immagini + testo associato...")

        # Encode query immagine
        query_image = Image.open(self.image_path).convert("RGB")
        inputs = self.clip_processor(text=[self.query], images=[query_image], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            query_img_emb = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])

        query_img_emb = torch.nn.functional.normalize(query_img_emb, p=2, dim=1).cpu().numpy().astype("float32")

        img_scores, img_indices = self.image_index.search(query_img_emb, self.top_k)

        results = []
        for score, idx in zip(img_scores[0], img_indices[0]):
            caption = self.captions[idx]
            case_id = self.case_ids[idx]
            case_text = self.case_text_map.get(case_id, "[No case text found]")
            results.append({
                "score": score,
                "caption": caption,
                "case_id": case_id,
                "case_text": case_text
            })
        return results

    def run(self):
        #print("Avvio VRAG (solo immagini)")
        self.load_metadata()
        self.build_or_load_image_index()
        results = self.retrieve_images_and_text()
        '''
        print("Risultati:")
        for i, res in enumerate(results):
            print(f"Risultato {i+1} - Score: {res['score']:.4f}")
            print(f"Immagine: {res['image_path']}")
            print(f"Case ID: {res['case_id']}")
            print(f"Caption: {res['caption']}")
            print(f"Testo associato: {res['case_text'][:400]}...")
        '''
        return results
