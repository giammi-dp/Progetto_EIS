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
import transformers, torch, bitsandbytes
print(transformers.__version__)
print(torch.__version__)
print(bitsandbytes.__version__)
import nibabel as nib
import numpy as np

def _build_image_paths(case_dir: str) -> dict:
    """Map modalities to .nii.gz files in patient folder"""
    paths = {}
    modalities = ["seg", "t1c", "t1n", "t2f", "t2w"]

    for modality in modalities:
        found = [f for f in os.listdir(case_dir)
                 if f.lower().endswith(f"-{modality}.nii.gz") or f.lower().endswith(f"-{modality}.nii")]
        if not found:
            raise FileNotFoundError(f"File for modality '{modality}' not found in {case_dir}")
        paths[modality] = os.path.join(case_dir, found[0])
    return paths

def get_tumor_slice(mri_path: str, seg_path: str) -> Image.Image:
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
        print(f"Error processing image {mri_path}: {e}")
        raise

# Caricamenti

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


print("Modello caricato")


# Caricamenti
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((336, 336))])

# Carica test metadata e testo reale
with open('MMRAG/json_e_metadata_report_medici/global_finding_test.json') as f:
    test_meta = json.load(f)





results = []

test_meta = dict(list(test_meta.items()))

for img_id in tqdm(test_meta.keys(), desc="Test loop"):
    print(f'Processing {img_id}')
    ground_truth = test_meta.get(img_id, "")
    cleaned_ground_truth = clean_reference_text(ground_truth, False)
    path = os.path.join("./ASNR-MICCAI-BraTS2023-Challenge-TrainingData/", img_id)
    if 'MET' in img_id:
        type = 'MET'

    elif 'GLI' in img_id:
        type = 'GLI'
    elif 'MEN' in img_id:
        type = 'MEN'
    print(f'Tipo: {type}')


    prompt = f"""
<image>
You are an expert oncologist. The user has provided an image.

Using the image, generate a *detailed, concise, and professional medical diagnostic report* suitable for clinical documentation. The report must include:

1. A description of the pathology visible in the image.
2. The diagnosis based on the image and clinical data.
3. Possible treatments, therapies, or management options.

*Critical instructions for accuracy:*
- Always prioritize *verified clinical reports* over image interpretation if there is any discrepancy (e.g., lesion location, size, or characteristics).
- Maintain precision on laterality, lesion dimensions, and edema.
- Avoid any image reference tags like [<...>].
- Start immediately by describing the case and generating the requested report.
- Keep language professional, clear, and suitable for clinical documentation.
"""

    paths = _build_image_paths(path)
    # 4. Prepara immagine
    image = get_tumor_slice(paths['t1c'], paths['seg'])


    image_resized = transform(image)
    '''
    plt.figure(figsize=(10, 10))
    plt.imshow(image_resized)
    plt.show()
    '''
    inputs = processor(text=[prompt], images=[image_resized], return_tensors="pt").to(device)



    # 5. Generazione report
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
        output = model.generate(
            **inputs,
            max_new_tokens = 2048,
            min_new_tokens= len(cleaned_ground_truth),
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

    # 6. Valutazione
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
df_out.to_csv("evaluation_llava.csv", index=False)
print(" Test completato")
