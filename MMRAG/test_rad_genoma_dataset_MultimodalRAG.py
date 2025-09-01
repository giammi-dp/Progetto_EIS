
from training_MMRAG_rad_genome import ImprovedMRAGWithTraining
from utils import clean_reference_text, evaluate_all
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import LlavaProcessor
from torchvision import transforms
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaProcessor
import torch
import matplotlib.pyplot as plt
import gc
import bitsandbytes as bnb
import transformers, torch, bitsandbytes
print(transformers.__version__)
print(torch.__version__)
print(bitsandbytes.__version__)



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
with open('json_e_metadata_report_medici/global_finding_test.json') as f:
    test_meta = json.load(f)





results = []

test_meta = dict(list(test_meta.items())[1:])

for img_id in tqdm(test_meta.keys(), desc="Test loop"):
    print(f'Processing {img_id}')
    ground_truth = test_meta.get(img_id, "")
    cleaned_ground_truth = clean_reference_text(ground_truth, False)
    path = os.path.join("../ASNR-MICCAI-BraTS2023-Challenge-TrainingData/", img_id)
    if 'MET' in img_id:
        type = 'MET'

    elif 'GLI' in img_id:
        type = 'GLI'
    elif 'MEN' in img_id:
        type = 'MEN'
    print(f'Tipo: {type}')

    # Istanzia oggetti
    rag_img = ImprovedMRAGWithTraining(
        query_path=path,
        type=type,
        top_k=5,
        approach="multimodal",
        attention_type='cross_modal',
        auto_train=True)

    similar = rag_img.run(None)

    if similar[0][1] == ground_truth:
        case = similar[1][1]

    else:
        case = similar[0][1]

    prompt = f"""
<image>
You are an expert oncologist. The user has provided an image and the following clinical context from a similar, but not identical, case: {case}.

Using both the image and the clinical information, generate a *detailed, concise, and professional medical diagnostic report* suitable for clinical documentation. The report must include:

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

    paths = rag_img._build_image_paths(path)

    # 4. Prepara immagine
    image = rag_img.get_tumor_slice(paths['t1c'], paths['seg'], None)


    image_resized = transform(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_resized)
    plt.show()
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
df_out.to_csv("evaluation_rad_genoma_MMRAG_LL1.csv", index=False)
print(" Test completato")
