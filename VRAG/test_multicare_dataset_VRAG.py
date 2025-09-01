from VRAG.VRAG_multicare import VRAG
from utils import clean_reference_text, evaluate_all
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import gc
import torch


# Caricamenti

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaProcessor
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


print("Modello caricato")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((336, 336))])

# Carica test metadata e testo reale
with open('../medical_datasets/brain_tumor_multimodal/image_metadata_test.json') as f:
    test_meta = [json.loads(line) for line in f if line.strip()]
df = pd.read_csv('../medical_datasets/brain_tumor_multimodal/cases.csv')
case_text_map = dict(zip(df['case_id'], df['case_text'].fillna('')))


rag_img = VRAG(query="", image_path="", top_k=3)

results = []

for item in tqdm(test_meta[41:51], desc="Test loop"):
    image_path = '../' + item['file_path']
    case_id = item['case_id']
    ground_truth = case_text_map.get(case_id, "")
    cleaned_ground_truth = clean_reference_text(ground_truth, False)

    caption = item['caption']
    # VRAG → immagini simili
    rag_img.image_path = image_path
    rag_img.query = caption
    similar = rag_img.run()

    print(similar)

    prompt = f"""
<image>
You are an expert oncologist. The user has provided an image and the following clinical context from a similar, but not identical, case: {similar[0]['case_text']}.

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
"""


    # Adatta immagine al VLM
    image = Image.open(image_path).convert("RGB")

    image_resized = transform(image)

    inputs = processor(text=[prompt], images=[image_resized], return_tensors="pt").to(device)

    # Generazione report
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            min_new_tokens=700,
            do_sample=True,
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
        "image_path": image_path,
        "case_id": case_id,
        "generated_report": cleaned_generated_report,
        "ground_truth": cleaned_ground_truth,
    }
    result.update(metrics)

    results.append(result)

# Salva tutti i risultati
df_out = pd.DataFrame(results)
df_out.to_csv("eval_all_metrics_41-50_cases.csv", index=False)
print(" Test completato")
