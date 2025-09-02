import os.path
import numpy as np
import torch
import json
from PIL import Image
from typing import List
import logging
from MMRAG.fusion_file import Fusion
import torch.nn.functional as F

from MMRAG.MMRAG_rad_genome import MRAG


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer del modulo di fusione per MMRAG
    """

    def __init__(self, fusion_model, device, type):
        self.fusion_model = fusion_model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.fusion_model.parameters(), lr=0.001)
        self.type = type

    def train_self_supervised(self, mrag_instance, epochs: int = 20):
        """
        Self-supervised training usando la hybrid cross modal loss
        """
        logger.info(f"Starting self-supervised training for {epochs} epochs...")

        # Carica i dati per training
        train_data = self._prepare_training_data(mrag_instance)

        if not train_data:
            logger.error("No training data prepared!")
            return

        self.fusion_model.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_texts, batch_images in train_data:
                if not batch_texts:  # Skip empty batches
                    continue

                self.optimizer.zero_grad()

                try:
                    # Forward pass
                    loss = self.hybrid_cross_modal_loss(batch_texts, batch_images, mrag_instance)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    logger.warning(f"Skipping batch due to error: {e}")
                    continue

            avg_loss = total_loss / num_batches if num_batches > 0 else 0

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        self.fusion_model.eval()
        logger.info("Training completed!")

    def _prepare_training_data(self, mrag_instance, batch_size: int = 16):
        """
        Prepara dati di training dai dati esistenti del MRAG
        """
        # Carica il JSON dei report
        try:
            with open(f"json_e_metadata_report_medici/global_finding_{self.type}.json", "r") as f:
                report_dict = json.load(f)
        except FileNotFoundError:
            logger.error(f"global_finding_{self.type}.json not found!")
            return []

        image_dir = "../ASNR-MICCAI-BraTS2023-Challenge-TrainingData"

        # Colleziona tutti i dati
        all_texts = []
        all_images = []

        count = 0


        logger.info("Preparing training data...")

        for image_id, report in report_dict.items():

            case_dir = os.path.join(image_dir, image_id)
            if not os.path.exists(case_dir):
                continue

            try:
                paths = mrag_instance._build_image_paths(case_dir)
                image_slice = mrag_instance.get_tumor_slice(paths['t1c'], paths['seg'], None)

                all_texts.append(report)
                all_images.append(image_slice)
                count += 1

                if count % 10 == 0:
                    logger.info(f"Loaded {count} samples...")

            except Exception as e:
                logger.debug(f"Skipping {image_id}: {e}")
                continue

        logger.info(f"Prepared {len(all_texts)} training samples")

        if not all_texts:
            logger.error("No valid training samples found!")
            return []

        # Crea batch
        batches = []
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_images = all_images[i:i + batch_size]
            batches.append((batch_texts, batch_images))

        return batches


    def hybrid_cross_modal_loss(self, texts: List[str], images: List, mrag_instance):
        """
        Combinazione di Alignment + Contrastive
        """

        text_embeddings = torch.tensor(
            mrag_instance._encode_text(texts),
            device=self.device,
            dtype=torch.float32
        )
        image_embeddings = torch.tensor(
            mrag_instance._encode_image(images),
            device=self.device,
            dtype=torch.float32
        )

        result = self.fusion_model(text_embeddings, image_embeddings)
        fused_emb = result[0] if isinstance(result, tuple) else result

        # === PARTE ALIGNMENT  ===
        text_norm = F.normalize(text_embeddings, p=2, dim=1)
        image_norm = F.normalize(image_embeddings, p=2, dim=1)
        fused_norm = F.normalize(fused_emb, p=2, dim=1)

        text_sim = torch.sum(text_norm * fused_norm, dim=1)
        image_sim = torch.sum(image_norm * fused_norm, dim=1)

        alignment_loss = (
                F.relu(0.8 - text_sim).mean() +  # Penalizza se similarità < 0.8
                F.relu(0.8 - image_sim).mean()
        )

        # === PARTE 2: CONTRASTIVE PER NEGATIVI DIFFICILI ===
        contrastive_loss = torch.tensor(0.0, device=self.device)
        batch_size = text_norm.size(0)


        for i in range(batch_size):
            #Trova il negativo più difficile per fused_emb considerando text_emb
            neg_text_sims = torch.sum(fused_norm[i:i + 1] * text_norm, dim=1)
            neg_text_sims[i] = -1.0
            hardest_neg_text_sim = neg_text_sims.max()

            #Trova il negativo più difficile per fused_emb considerando img_emb
            neg_image_sims = torch.sum(fused_norm[i:i + 1] * image_norm, dim=1)
            neg_image_sims[i] = -1.0  # Mask self
            hardest_neg_image_sim = neg_image_sims.max()

            #Applicazione del margine: il positivo deve superare il negativo di un certo margine
            margin = 0.2
            text_margin_loss = F.relu(hardest_neg_text_sim - text_sim[i] + margin)
            image_margin_loss = F.relu(hardest_neg_image_sim - image_sim[i] + margin)

            contrastive_loss += (text_margin_loss + image_margin_loss) / (2 * batch_size)

        return (alignment_loss + contrastive_loss) / 2



# Integrazione MRAG
class MRAGWithTraining(MRAG):
    """
    Versione del MRAG con training automatico del fusion model
    """

    def __init__(self, query_path: str, type, top_k: int = 3,
                 auto_train: bool = True):
        super().__init__(query_path, type, top_k)

        self.is_fusion_trained = False

        if auto_train:
            self._train_fusion_if_needed()

    def _train_fusion_if_needed(self):
        """
        Allena il fusion model se non è già stato fatto
        """
        model_path = f"pesi_fusion/fusion_trained_{self.type}.pth"

        if os.path.exists(model_path):
            logger.info(f"Loading pre-trained fusion model: {model_path}")
            try:
                self.fusion_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.is_fusion_trained = True
                logger.info("Pre-trained model loaded successfully!")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}. Training from scratch...")
                self._train_from_scratch(model_path)
        else:
            logger.info("No pre-trained model found. Training from scratch...")
            self._train_from_scratch(model_path)

    def _train_from_scratch(self, model_path):
        """Train the fusion model from scratch"""
        if not hasattr(self, 'fusion_model'):
            logger.error("No fusion model to train!")
            return

        logger.info("Training fusion model...")
        trainer = Trainer(self.fusion_model, self.device, type=self.type)
        trainer.train_self_supervised(self, epochs=20)

        # Salva il modello allenato
        try:
            torch.save(self.fusion_model.state_dict(), model_path)
            self.is_fusion_trained = True
            logger.info(f"Fusion model trained and saved to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            self.is_fusion_trained = True

