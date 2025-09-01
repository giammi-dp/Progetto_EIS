#%%
import nltk
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pycocoevalcap.cider.cider import Cider
import evaluate

bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")



sbert_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")


def compute_bleu_score(reference, hypothesis):
    if not reference or not hypothesis:
        return -1
    bleu_res = bleu_metric.compute(predictions=[hypothesis.lower()], references=[[reference.lower()]])
    return bleu_res["bleu"]


def compute_meteor_score(reference, hypothesis):
    if not reference or not hypothesis:
        return -1
    meteor_res = meteor_metric.compute(predictions=[hypothesis.lower()], references=[reference.lower()])
    return meteor_res["meteor"]


def compute_cider_score(reference, hypothesis):
    if not reference or not hypothesis:
        return -1
    cider = Cider()
    gts = {"0": [reference]}
    res = {"0": [hypothesis]}
    score, _ = cider.compute_score(gts, res)
    return score


def compute_bert_score(reference, hypothesis):
    if not reference or not hypothesis:
        return -1
    P, R, F1 = bert_score([hypothesis], [reference], lang="en")
    return F1[0].item()


def compute_cosine_similarity(reference, hypothesis):
    if not reference or not hypothesis:
        return -1
    ref_emb = sbert_model.encode(reference, convert_to_tensor=True)
    hyp_emb = sbert_model.encode(hypothesis, convert_to_tensor=True)
    return cosine_similarity(ref_emb.unsqueeze(0).cpu(), hyp_emb.unsqueeze(0).cpu())[0][0]


def compute_rouge_scores(reference, hypothesis):
    if not reference or not hypothesis:
        return {'rouge_1': -1, 'rouge_2': -1, 'rouge_L': -1}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge_1': scores['rouge1'].fmeasure,
        'rouge_2': scores['rouge2'].fmeasure,
        'rouge_L': scores['rougeL'].fmeasure
    }


# Funzione generale per valutare tutto
def evaluate_all(reference, hypothesis):
    return {
        "BLEU-4": compute_bleu_score(reference, hypothesis),
        "METEOR": compute_meteor_score(reference, hypothesis),
        "CIDEr": compute_cider_score(reference, hypothesis),
        "BERTScore_F1": compute_bert_score(reference, hypothesis),
        "Cosine_Similarity": compute_cosine_similarity(reference, hypothesis),
        **compute_rouge_scores(reference, hypothesis)
    }


#-------------pulizia report------------------
#%%
import re
import nltk

def clean_reference_text(text, generated):

    # Rimuovi riferimenti a immagini tra <>
    text = re.sub(r"<[^<>]+>", "", text)

    # Rimuovi età e genere (pattern base: "A 19-year-old boy/girl/man/woman...")
    text = re.sub(r"\b[Aa]\s\d{1,3}-year-old\s\w+\b", "", text)


    # Rimuovi date (formati comuni come MM/DD/YYYY, DD-MM-YYYY, Month DD, YYYY)
    text = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", "", text)
    text = re.sub(r"\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2,4}\b", "", text)
    text = re.sub(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},\s\d{2,4}\b", "", text)
    text = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "", text)


    # Rimuovi riferimenti a figure e tabelle (es. Figure 1, Table 2)
    text = re.sub(r"\b(Figure|Fig|Table)\s\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Figures|Figs|Tables)\s\d+-\d+\b", "", text, flags=re.IGNORECASE)


    # Rimuovi riferimenti a sezioni (es. Methods, Results)
    text = re.sub(r"\b(Introduction|Methods|Results|Discussion|Conclusion|Case Presentation)\b", "", text, flags=re.IGNORECASE)


    # Rimuovi punteggiatura extra e caratteri speciali
    text = re.sub(r"[^\w\s.,;:]", "", text)


    # Rimuovi whitespace multiplo
    text = re.sub(r"\s+", " ", text).strip()

    return text
