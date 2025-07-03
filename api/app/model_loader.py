import os
import logging
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    def __init__(self, model_path: str):
        logger.info(f"Loading tokenizer from path: {model_path}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        logger.info(f"Loading model from path: {model_path}")
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
    
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        probs = outputs.logits.sigmoid().detach().numpy()[0]
        labels = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat",]
        result = {label: float(prob) for label, prob in zip(labels, probs)}
        return result

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "../model/roberta-finetune-toxic")
    return Model(model_path)