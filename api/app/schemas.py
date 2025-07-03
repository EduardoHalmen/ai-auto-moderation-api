from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class ToxicityResponse(BaseModel):
    toxicity: float
    severe_toxicity: float
    obscene: float
    identity_attack: float
    insult: float
    threat: float