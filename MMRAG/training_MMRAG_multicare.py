import os.path
import numpy as np
import torch
import json
from PIL import Image
from typing import List
import logging
from MMRAG.fusion_file import Fusion
import torch.nn.functional as F
import pandas as pd
from MMRAG.MMRAG_multicare import MRAG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer per il fusion module del MRAG
    """

    def __init__(self, fusion_model, device):
        self.fusion_model = fusion_model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(fusion_model_model.parameters(), lr=0.001)

    def train_self_supervised(self, mrag_instance, epochs: int = 50):
        """
        Self-supervised training hybrid cross modal loss
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
            with open("../medical_datasets/brain_tumor_multimodal/image_metadata_train.json") as f:
                image_meta = [json.loads(line) for line in f if line.strip()]

        except FileNotFoundError:
            logger.error("medical_datasets/brain_tumor_multimodal/image_metadata_train.json not found!")
            return []

        image_dir = "../medical_datasets/brain_tumor_multimodal/images"

        # Colleziona tutti i dati
        all_texts = []
        all_images = []

        count = 0


        logger.info("Preparing training data...")

        for item in image_meta:
            image_file = '../' + item['file_path']
            case_id = item.get('case_id')

            # Carica mappa case_id → case_text
            df = pd.read_csv('../medical_datasets/brain_tumor_multimodal/cases.csv')
            case_text_map = dict(zip(df['case_id'], df['case_text'].fillna('')))

            all_texts.append(case_text_map.get(case_id, "[No case text found]"))
            image = Image.open(image_file).convert("RGB")
            all_images.append(image)
            count += 1

            if count % 10 == 0:
                logger.info(f"Loaded {count} samples...")

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

        # === PARTE CONTRASTIVE (NEGATIVI DIFFICILI) ===
        contrastive_loss = torch.tensor(0.0, device=self.device)
        batch_size = text_norm.size(0)


        for i in range(batch_size):
            # Trova il negativo più difficile per fused_emb considerando text_emb
            neg_text_sims = torch.sum(fused_norm[i:i + 1] * text_norm, dim=1)
            neg_text_sims[i] = -1.0
            hardest_neg_text_sim = neg_text_sims.max()

            # Trova il negativo più difficile per fused_emb considerando img_emb
            neg_image_sims = torch.sum(fused_norm[i:i + 1] * image_norm, dim=1)
            neg_image_sims[i] = -1.0
            hardest_neg_image_sim = neg_image_sims.max()

            # Applicazione del margine: il positivo deve superare il negativo di un certo margine
            margin = 0.2
            text_margin_loss = F.relu(hardest_neg_text_sim - text_sim[i] + margin)
            image_margin_loss = F.relu(hardest_neg_image_sim - image_sim[i] + margin)

            contrastive_loss += (text_margin_loss + image_margin_loss) / (2 * batch_size)

        return (alignment_loss + contrastive_loss) / 2



# Integrazione MRAG
class MRAGWithTraining(MRAG):
    """
    Versione del MRAG con training automatico del fusion module
    """

    def __init__(self, query_path: str, top_k: int = 3, approach: str = "multimodal",
                 auto_train: bool = True):
        super().__init__(query_path, top_k, approach)

        self.is_fusion_trained = False

        if self.approach == 'multimodal' and auto_train:
            self._train_fusion_if_needed()

    def _train_fusion_if_needed(self):
        """
        Allena fusion model se non è già stato fatto
        """

        model_path = f"./pesi_fusion/fusion_trained_multicare.pth"

        if os.path.exists(model_path):
            logger.info(f"Loading pre-trained fusion model: {model_path}")
            try:
                self.fusion_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.is_fusion_trained = True
                logger.info(" Pre-trained model loaded successfully!")
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
        trainer = Trainer(self.fusion_model, self.device)
        trainer.train_self_supervised(self, epochs=50)

        # Salva il modello allenato
        try:
            torch.save(self.fusion_model.state_dict(), model_path)
            self.is_fusion_trained = True
            logger.info(f" Fusion model trained and saved to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            self.is_fusion_trained = True  # Continue anyway


def train_and_test():
    """Funzione per testare"""
    query_path = "../medical_datasets/brain_tumor_multimodal/images/PMC1/PMC10/PMC10018421_crn-2023-0015-0001-529741_f1_a_1_4.webp"

    try:
        # Controlla che il file query esista
        if not os.path.exists(os.path.dirname(query_path)):
            logger.error(f"Query directory not found: {os.path.dirname(query_path)}")
            return None

        logger.info("Starting MRAG with fusion training...")

        # Usa la versione con training automatico
        rag = MRAGWithTraining(
            query_path,
            top_k=5,
            approach="multimodal",
            auto_train=True  # Training automatico
        )

        logger.info("Running RAG pipeline...")
        results = rag.run()

        if results:
            logger.info(f" Retrieved {len(results)} results")

            # Mostra primi risultati
            print(f"\ TOP RESULTS:")
            print("-" * 60)
            for i, (img_id, report, score) in enumerate(results[1:3]):
                print(f"[{i + 1}] Score: {score} | ID: {img_id}")
                print(f"    Report: {report[:100]}...")
                print()



        return results

    except Exception as e:
        logger.error(f"Error in train_and_test: {e}")
        import traceback
        traceback.print_exc()
        return None



if __name__ == "__main__":
    # Scegli il test da eseguire
    print(" Running fusion-based MRAG...")

    # Opzione 1: Test completo (può richiedere tempo)
    results = train_and_test()

    if results:
        print(f" Test completed! Retrieved {len(results)} results")
    else:
        print("Test failed!")
