from http.client import HTTPException
from fastapi import FastAPI, logger, Depends
from .schemas import TextInput, ToxicityResponse
from .dependencies import get_model, Model

app = FastAPI(
    title="Toxicity Detection API",
    description="API for detecting toxic content in text, it will give a probability for each toxicity category (toxicity, severe toxicity, insult, obscene, threat, sexual explicit and identity attack)",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {"message": "Hello World :)"}

@app.post("/evaluate_comment",
         summary="Evaluate Comment Toxicity",
         description="Analyzes the provided text comment and returns toxicity scores across multiple categories",
         response_model=ToxicityResponse)
async def evaluate_comment(input: TextInput, model: Model = Depends(get_model)):
    try:
        results = model.predict(input.text)
        return results
    except Exception as e:
        logger.error(f"Error predicting toxicity: {e}")
        raise HTTPException(status_code=500, detail="Error processing request.")
