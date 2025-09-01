import os.path
import numpy as np
import faiss
import torch
import json
from open_clip import create_model_from_pretrained, get_tokenizer
import nibabel as nib
from PIL import Image
import pickle
from typing import List, Tuple
from MMRAG.attention_file import CrossModalAttentionN
import pandas as pd



class ImprovedMRAG:
    def __init__(self, query_path: str, top_k: int = 3, approach: str = "multimodal", attention_type : str = 'cross_modal',):
        """
        Args:
            query_path: Path al file MRI
            top_k: Numero di risultati da recuperare
            approach: "cross_modal" or "multimodal"
            attention_type: Type of attention mechanism
        """
        self.query_path = query_path
        self.top_k = top_k
        self.approach = approach
        self.attention_type = attention_type
        self.report_texts = []
        self.image_ids = []

        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        #print(f"Using device: {self.device}")

        # Load model
        self._load_model()

        # Paths for saving/loading
        self.db_path = f"vector_db/vector_db_{approach}_multicare.index"
        self.metadata_path = f"json_e_metadata_report_medici/metadata_{approach}_multicare.pkl"

        if self.approach == 'multimodal' and self.attention_type != None:
            embed_dim = 512
            self.attention_fusion_model = CrossModalAttentionN(embed_dim)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

    def _load_model(self):
        """Load BiomedCLIP model and preprocessing"""
        try:
            model, preprocess = create_model_from_pretrained(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                return_transform=True
            )
            tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

            self.model = model.to(self.device)
            self.preprocess = preprocess
            self.tokenizer = tokenizer
            self.model.eval()  # Set to eval mode

            #print("Model loaded successfully")
        except Exception as e:
            #print(f"Error loading model: {e}")
            raise

    def _build_image_paths(self, case_dir: str) -> dict:
        """Map modalities to .nii.gz files in patient folder"""
        paths = {}
        modalities = ["seg", "t1c", "t1n", "t2f", "t2w"]

        for modality in modalities:
            found = [f for f in os.listdir(case_dir)
                     if f.lower().endswith(f"-{modality}.nii.gz")]
            if not found:
                raise FileNotFoundError(f"File for modality '{modality}' not found in {case_dir}")
            paths[modality] = os.path.join(case_dir, found[0])
        return paths

    def get_tumor_slice(self, mri_path: str, seg_path: str) -> Image.Image:
        """Extract the slice with maximum tumor content and preprocess it properly"""
        try:
            # Load volumes
            mri_img = nib.load(mri_path).get_fdata()
            seg_img = nib.load(seg_path).get_fdata()

            # Find slice with maximum tumor content
            tumor_presence = np.sum(seg_img > 0, axis=(0, 1))
            best_slice_idx = np.argmax(tumor_presence)

            # Extract slice
            image_slice = mri_img[:, :, best_slice_idx]

            # Improved preprocessing - preserve more information
            # Clip extreme values (remove outliers)
            p1, p99 = np.percentile(image_slice[image_slice > 0], [1, 99])
            image_slice = np.clip(image_slice, p1, p99)

            # Normalize to 0-255 with better contrast
            if np.ptp(image_slice) > 0:
                image_slice = ((image_slice - np.min(image_slice)) / np.ptp(image_slice) * 255).astype(np.uint8)
            else:
                image_slice = np.zeros_like(image_slice, dtype=np.uint8)

            # Convert to RGB PIL image
            pil_image = Image.fromarray(image_slice).convert('RGB')

            return pil_image

        except Exception as e:
            #print(f"Error processing image {mri_path}: {e}")
            raise

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts using BiomedCLIP"""
        inputs = self.tokenizer(texts=texts).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # L2 normalize
            return embeddings.detach().cpu().numpy().astype('float32')

    def _encode_image(self, images: List[Image.Image]) -> np.ndarray:
        """Encode images using BiomedCLIP"""
        # Preprocess images
        image_tensors = []
        for img in images:
            tensor = self.preprocess(img).unsqueeze(0)
            image_tensors.append(tensor)

        batch_tensor = torch.cat(image_tensors, dim=0).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # L2 normalize
            return embeddings.detach().cpu().numpy().astype('float32')

    def build_or_load_vector_db(self):
        """Build or load vector database based on chosen approach"""

        if os.path.exists(self.db_path) and os.path.exists(self.metadata_path):
            #print("Loading existing Vector DB...")
            self.index = faiss.read_index(self.db_path)

            with open(self.metadata_path, "rb") as f:
                metadata = pickle.load(f)
                self.report_texts = metadata['report_texts']
                self.image_ids = metadata['image_ids']
            return

        #print(f"Creating new Vector DB with {self.approach} approach...")

        # Load data
        with open("../medical_datasets/brain_tumor_multimodal/image_metadata_train.json") as f:
            image_meta = [json.loads(line) for line in f if line.strip()]

        image_dir = "../medical_datasets/brain_tumor_multimodal/images"
        embeddings = []

        # Process data in batches for efficiency
        batch_size = 32
        batch_reports = []
        batch_images = []
        batch_ids = []

        for item in image_meta:
            image_file = '../' + item['file_path']
            case_id = item.get('case_id')
            # Carica mappa case_id → case_text
            df = pd.read_csv('../medical_datasets/brain_tumor_multimodal/cases.csv')
            case_text_map = dict(zip(df['case_id'], df['case_text'].fillna('')))


            image = Image.open(image_file).convert("RGB")

            batch_reports.append(case_text_map.get(case_id))
            batch_images.append(image)
            batch_ids.append(case_id)


            # Process batch when full
            if len(batch_reports) >= batch_size:
                batch_embeddings = self._process_batch(batch_reports, batch_images)
                embeddings.extend(batch_embeddings)
                self.report_texts.extend(batch_reports)
                self.image_ids.extend(batch_ids)

                batch_reports, batch_images, batch_ids = [], [], []
                #print(f"Processed {len(embeddings)} samples...")



        # Process remaining batch
        if batch_reports:
            batch_embeddings = self._process_batch(batch_reports, batch_images)
            embeddings.extend(batch_embeddings)
            self.report_texts.extend(batch_reports)
            self.image_ids.extend(batch_ids)

        # Build and save FAISS index
        if not embeddings:
            raise ValueError("No embeddings created!")

        embeddings_array = np.array(embeddings).astype('float32')
        dim = embeddings_array.shape[1]

        # Use IndexFlatIP for cosine similarity (since we L2 normalized)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_array)
        faiss.write_index(index, self.db_path)

        # Save metadata
        metadata = {
            'report_texts': self.report_texts,
            'image_ids': self.image_ids
        }
        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        self.index = index
        #print(f"Vector DB created with {len(embeddings)} embeddings")

    def _process_batch(self, reports: List[str], images: List[Image.Image]) -> List[np.ndarray]:
        """Process a batch of reports and images based on the chosen approach"""

        if self.approach == "cross_modal":
            # Approach A: Store only text embeddings, query with image embeddings
            text_embeddings = self._encode_text(reports)
            return [emb for emb in text_embeddings]

        elif self.approach == "multimodal":
            # Approach B: Combine text and image embeddings
            text_embeddings = self._encode_text(reports)
            image_embeddings = self._encode_image(images)

            text_embeddings_torch = torch.from_numpy(text_embeddings).to(self.device)
            image_embeddings_torch = torch.from_numpy(image_embeddings).to(self.device)

            if self.attention_fusion_model:
                self.attention_fusion_model.eval()
                with torch.no_grad():
                    fused_emb_torch = self.attention_fusion_model(
                        text_embeddings_torch, image_embeddings_torch
                    )

                    # Molti modelli di attenzione restituiscono più valori. Gestiamo i casi.
                    if isinstance(fused_emb_torch, tuple):
                        fused_emb_torch = fused_emb_torch[0]

                fused_embeddings_np = fused_emb_torch.cpu().numpy()

                # Normalizzazione finale se non gestita dal modello di attenzione
                norms = np.linalg.norm(fused_embeddings_np, axis=1, keepdims=True)
                fused_embeddings_np = fused_embeddings_np / (norms + 1e-8)

                return [emb for emb in fused_embeddings_np]

        else:
            raise ValueError(f"Unknown approach: {self.approach}")

    def retrieve(self) -> List[Tuple[str, str, float]]:
        """Retrieve most similar reports for the query image"""

        query_image = Image.open(self.query_path).convert('RGB')


        # Encode query based on approach
        if self.approach == "cross_modal":
            # Query with image embedding against text embeddings
            query_embedding = self._encode_image([query_image])

        elif self.approach == "multimodal":
            # For multimodal, we only have image for query, so use image embedding
            # In a real scenario, you might want to include query text too
            query_embedding = self._encode_image([query_image])

        # Search
        scores, indices = self.index.search(query_embedding, self.top_k)

        # Return results with metadata
        results = []
        for j, idx in enumerate(indices[0]):
            if idx < len(self.report_texts):  # Safety check
                results.append({
                    "case_id": self.image_ids[idx],
                    "case_text": self.report_texts[idx],
                    "score": float(scores[0][j])
                })

        return results




    def run(self) -> List[Tuple[str, str, float]]:
        """Run the complete RAG pipeline"""
        #print(f'Running {self.approach} RAG...')

        self.build_or_load_vector_db()
        #print('Vector DB ready')

        results = self.retrieve()
        return results
