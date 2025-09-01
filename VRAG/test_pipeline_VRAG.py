
from monai.networks.nets import SegResNet
from segmentator import Segmentator

from VRAG.VRAG_rad_genoma import VRAG
from utils import clean_reference_text, evaluate_all
import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaProcessor
import matplotlib.pyplot as plt
import gc
import torch

model_id = "Eren-Senoglu/llava-med-v1.5-mistral-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config, torch_dtype=torch.float16)
processor = LlavaProcessor.from_pretrained(model_id)
processor.patch_size = 14

transform = transforms.Compose([transforms.Resize((336, 336))])


model_checkpoint_path = "../models/monai_brats_mri_segmentation/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_seg = SegResNet(
    spatial_dims=3,
    init_filters=16,
    in_channels=4,
    out_channels=3,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    dropout_prob=0.2,
).to(device)


#Caricamento dei pesi del modello
try:
    model_seg.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model_seg.eval()
    print("Modello caricato con successo direttamente dal file .pt")
except FileNotFoundError:
    print(f"Errore: File checkpoint del modello non trovato a {model_checkpoint_path}. Assicurati che il percorso sia corretto.")
    exit()
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}. Controlla che l'architettura della UNet (in_channels, out_channels, channels, strides, etc.) corrisponda esattamente a quella del modello addestrato.")
    exit()

# Istanza VRAG
rag_img = VRAG(query="", image_id="", top_k=3)
results = []

with open('../MMRAG/json_e_metadata_report_medici/global_finding_test.json', 'r') as file:
    data = json.load(file)

test_meta = dict(list(data.items()))

for img_id in tqdm(test_meta.keys(), desc="Test loop"):
    image_path = os.path.join('../ASNR-MICCAI-BraTS2023-Challenge-TrainingData', img_id)
    #print(image_path)

    # Istanza Segmentator per segmentazione immagine
    segmentator = Segmentator(image=image_path, model=model_seg)

    paths = segmentator._build_image_paths()
    dataloader = segmentator.preprocessing_step()
    segmentation, batch = segmentator.inference(dataloader=dataloader)

    hemisphere, lobe = segmentator.bounding_box(segmentation=segmentation, paths=paths)
    image_array, slice_idx = segmentator.plot_slice(segmentation=segmentation,  batch=batch)

    (total_tumor_volume_seg,
     volume_net_seg,
     volume_ed_seg,
     volume_et_seg,
     total_tumor_volume_gt,
     volume_net_gt,
     volume_ed_gt,
     volume_et_gt) = segmentator.calculate_volume(segmentation=segmentation, batch=batch)

    image = Image.fromarray(image_array)

    # Info aggiunte al prompt del VLM
    query = (" Tumor located in the " + str(hemisphere) +
            " hemisphere and " + str(lobe) +
            " lobe with a volume of " + str(total_tumor_volume_seg) + " mm^3")



    ground_truth = test_meta.get(img_id, "")
    cleaned_ground_truth = clean_reference_text(ground_truth, False)
    path = os.path.join("../ASNR-MICCAI-BraTS2023-Challenge-TrainingData/", img_id)
    if 'MET' in img_id:
        query = 'metastasi'
    elif 'GLI' in img_id:
        query = 'glioma'
    else:
        query = 'meningioma'

    rag_img.image_id = img_id
    rag_img.query = query
    similar = rag_img.run(slice_idx)

    if similar[0]['case_text'] == ground_truth:
        case = similar[1]['case_text']

    else:
        case = similar[0]['case_text']

    prompt = f"""
    <image>
    You are an expert oncologist. The user has provided an image of the segmented tumor and the following clinical context from a similar, but not identical, case: {case}.
    There are also some information about the location of the tumor : {query}
    Using both the image and the given information, generate a *detailed, concise, and professional medical diagnostic report* suitable for clinical documentation. The report must include:

    1. A description of the pathology visible in the image.
    2. The diagnosis based on the image and clinical data.
    3. Possible treatments, therapies, or management options.

    *Critical instructions for accuracy:*
    - Always prioritize *verified clinical reports* over image interpretation if there is any discrepancy (e.g., lesion location, size, or characteristics).
    - Maintain precision on laterality, lesion dimensions, and edema.
    - Avoid any image reference tags like [<...>].
    - Start immediately by describing the case and generating the requested report.
    - Keep language professional, clear, and suitable for clinical documentation.
    - Note that the clinical report may refers to a different type of tumor, so always check the image to predict the tumor type.
    """


    image_resized = transform(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_resized)
    plt.show()
    inputs = processor(text=[prompt], images=[image_resized], return_tensors="pt").to(device)

    # Generazione report
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            min_new_tokens=len(cleaned_ground_truth),
            do_sample=False,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[-1]
    generated_tokens = output[0][prompt_len:]
    generated_report = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    cleaned_generated_report = clean_reference_text(generated_report, True)
    print(cleaned_generated_report)

    # Valutazione
    metrics = evaluate_all(reference=cleaned_ground_truth, hypothesis=cleaned_generated_report)

    result = {
        "image_path": path,
        "generated_report": cleaned_generated_report,
        "ground_truth": cleaned_ground_truth,
    }
    result.update(metrics)
    print(metrics)
    results.append(result)

# Salva tutti i risultati
df_out = pd.DataFrame(results)
df_out.to_csv("evaluation_pipeline_VRAG_1.csv", index=False)
print(" Test completato")

