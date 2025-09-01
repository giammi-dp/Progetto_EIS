
from utils import evaluate_all, clean_reference_text
from monai.networks.nets import SegResNet
from segmentator import Segmentator

from MMRAG.training_MMRAG_rad_genome import MRAGWithTraining
from utils import clean_reference_text
import os
import json
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaProcessor
import gc
import torch


def run(image_id, prompt_user):
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

    results = []

    model_checkpoint_path = "../models/monai_brats_mri_segmentation/model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_seg = SegResNet(
        spatial_dims=3,
        init_filters=16,
        in_channels=4,
        out_channels=3,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    ).to(device)


    #Carico i pesi del modello

    model_seg.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model_seg.eval()

    with open('../VRAG/global_finding.json', 'r') as file:
        data = json.load(file)


    image_path = os.path.join('../ASNR-MICCAI-BraTS2023-Challenge-TrainingData', image_id)

    segmentator = Segmentator(image=image_path, model=model_seg)

    paths = segmentator._build_image_paths()
    dataloader = segmentator.preprocessing_step()
    segmentation, batch = segmentator.inference(dataloader=dataloader)

    hemisphere, lobe = segmentator.bounding_box(segmentation=segmentation, paths=paths)
    image_array, slice_idx = segmentator.plot_slice(segmentation=segmentation,  batch=batch)
    #segmentator.save_mask(paths=paths, segmentation=segmentation, name =img_id)
    (total_tumor_volume_seg,
         volume_net_seg,
         volume_ed_seg,
         volume_et_seg,
         total_tumor_volume_gt,
         volume_net_gt,
         volume_ed_gt,
         volume_et_gt) = segmentator.calculate_volume(segmentation=segmentation, batch=batch)

    image = Image.fromarray(image_array)

    query = (" Tumor located in the " + str(hemisphere) +
            " hemisphere and " + str(lobe) +
            " lobe with a volume of " + str(total_tumor_volume_seg) + " mm^3")

    path = os.path.join("../ASNR-MICCAI-BraTS2023-Challenge-TrainingData/", image_id)
    if 'MET' in image_id:
        type = 'MET'
    elif 'GLI' in image_id:
        type = 'GLI'
    elif 'MEN' in image_id:
        type = 'MEN'

    rag_img = ImprovedMRAGWithTraining(
            query_path=path,
            type=type,
            top_k=3,
            approach="multimodal",
            attention_type='cross_modal',
            auto_train=True)

    similar = rag_img.run(slice_idx)

    case = similar[0][1]

    if prompt_user is None:
        prompt = f"""
            <image>
            You are an expert oncologist. The user has provided an image of the segmented tumor and the following clinical context from a similar, but not identical, case: {case}.
            There are also some information about the location of the tumor : {query}
            Using both the image and the given information, generate a detailed, concise, and professional medical diagnostic report, suitable for clinical documentation. The report must include:
            
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
    else:

        prompt = prompt_user + f"""
            <image>
            You are an expert oncologist. The user has provided an image of the segmented tumor and the following clinical context from a similar, but not identical, case: {case}.
            There are also some information about the location of the tumor : {query}
            Using both the image and the given information, generate a detailed, concise, and professional medical diagnostic report, suitable for clinical documentation. The report must include:
            
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

    inputs = processor(text=[prompt], images=[image_resized], return_tensors="pt").to(device)

    # 5. Generazione report
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            min_new_tokens=200,
            do_sample=False,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
            )
        input_ids = inputs["input_ids"].to(device)
        prompt_len = input_ids.shape[-1]
        generated_tokens = output[0][prompt_len:]
        generated_report = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        cleaned_generated_report = clean_reference_text(generated_report, True)

        return image_resized, cleaned_generated_report
