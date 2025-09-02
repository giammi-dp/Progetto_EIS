import gradio as gr
import os
import zipfile
from MMRAG.pipeline import run

DATA_DIR = "../ASNR-MICCAI-BraTS2023-Challenge-TrainingData"


def list_cases():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]


def upload_case(zip_file):
    if zip_file is None:
        return gr.update(choices=list_cases()), "Nessun file caricato"
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    return gr.update(choices=list_cases()), f"Caricato: {os.path.basename(zip_file)}"


def process_case(case_name, user_prompt, history):
    # Messaggio temporaneo
    history.append((user_prompt, " Generazione in corso..."))

    # Esegui la pipeline
    image, report = run(case_name, user_prompt)

    # Aggiorna la chat con il report
    history[-1] = (user_prompt, report)

    return history, gr.update(value=image, visible=True), gr.update(value="")  # mostra immagine e svuota textbox


with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Tumor Segmentation & Report Chat")

    with gr.Row():
        with gr.Column(scale=1):
            case_selector = gr.Dropdown(choices=list_cases(), label=" 📂 Seleziona un caso")
            upload = gr.File(label=" ⬆️ Carica nuovo caso (.zip)", type="filepath")
            upload_output = gr.Textbox(label=" ⌛ Log caricamento")
            upload.upload(upload_case, inputs=upload, outputs=[case_selector, upload_output])

        # Colonna unica per chat + immagine + input
        with gr.Column(scale=2):
            with gr.Group():
                chatbot = gr.Chatbot(label=" 🗒️ Chat diagnostica", height=400)
                image_output = gr.Image(label="Segmentazione Tumore", visible=False, height=250)
            with gr.Row():
                user_input = gr.Textbox(placeholder=" ✍️ Scrivi il tuo prompt qui...", lines=2, scale=4)
                send_button = gr.Button("Invia", scale=1)

    send_button.click(
        fn=process_case,
        inputs=[case_selector, user_input, chatbot],
        outputs=[chatbot, image_output, user_input]  # l'ultimo aggiorna la textbox
    )

demo.launch()
