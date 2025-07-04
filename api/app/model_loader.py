import os
import logging
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    def __init__(self, model_url: str):
        logger.info(f"Loading tokenizer from Hugging Face Model Hub: {model_url}")
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_url)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise RuntimeError("Tokenizer loading failed")
        logger.info(f"Loading model from Hugging Face Model Hub: {model_url}")
        try:
            self.model = RobertaForSequenceClassification.from_pretrained(model_url)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError("Model loading failed")
    
    def predict(self, text: str):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            outputs = self.model(**inputs)
            probs = outputs.logits.sigmoid().detach().numpy()[0]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError("Prediction failed")
        outputs = self.model(**inputs)
        probs = outputs.logits.sigmoid().detach().numpy()[0]
        labels = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat",]
        result = {label: float(prob) for label, prob in zip(labels, probs)}
        return result

def load_model():
    model_url = "https://huggingface.co/sofiasartori24/roberta-finetune-toxic"
    return Model(model_url)