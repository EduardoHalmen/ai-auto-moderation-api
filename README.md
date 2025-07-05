# ToxBlock: Toxicity Detection System

This project focuses on building a deployable toxicity detection system, including robust text classification models, an API, and a user-friendly interface for testing and visualization. The primary datasets used were the **Jigsaw Unintended Bias in Toxicity Classification** dataset and the **Toxic Comment Classification Challenge**.

## Project Goals

* **Data Exploration:** Analyze and visualize the distribution and characteristics of toxic comments.
* **Data Processing:** Clean, preprocess, and engineer features from the raw text data.
* **Modeling:** Implement and evaluate machine learning and deep learning models for toxicity classification.
* **Bias Analysis:** Assess unintended bias in model predictions.
* **Evaluation:** Use appropriate metrics to evaluate model performance and fairness.
* **API:** Implement an API that uses the model to evaluate text.
* **UI:** Implement a simple UI to demonstrate the usage of the API and visualize the model's response.


## Model Access

The final fine-tuned RoBERTa model is now hosted on **Hugging Face Hub**, making the project lighter and easier to deploy.
The API automatically downloads the model from this link during initialization:

👉 [https://huggingface.co/sofiasartori24/roberta-finetune-toxic](https://huggingface.co/sofiasartori24/roberta-finetune-toxic)

You no longer need to manually store or load large model files inside the repository.


## Datasets

* [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification)
* [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)


## Project Structure

```
.
├── api/                     # Encapsulates the API
   ├── app/                  # API implementation
   ├── Dockerfile   
   ├── requirements.txt

├── frontend/                # Encapsulates the UI
   ├── app/                  # UI implementation
   ├── Dockerfile   
   ├── requirements.txt

├── toxicity_model/          # Modeling, data processing, and experimentation
   ├── data/                 # Raw data from the Unbiased Challenge
      ├── wikipedia_data/    # Raw data from the Toxic Comment Challenge
      ├── merged_data/       # Processed data from both challenges 
   ├── notebooks/            # Jupyter notebooks for exploration, processing, and modeling
   ├── src/                  # Source code for data processing, modeling, and utilities
   ├── results/              # Model outputs, evaluation metrics, and visualizations
   ├── docker-compose.yml   
   ├── requirements.txt
   └── README.md     
```


## Getting Started

### Running the Full System (API + UI) with Docker Compose:

1. In the project root directory, run:

   ```bash
   docker-compose up --build
   ```
2. Access the system in your browser:

   * **API Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
   * **UI for Testing:** [http://localhost:8501](http://localhost:8501)


### Running Only the UI:

1. Navigate to the `frontend` directory:

   ```bash
   cd frontend
   ```
2. Build the Docker image:

   ```bash
   docker build -t toxicity-ui .
   ```
3. Run the container:

   ```bash
   docker run -p 8501:8501 toxicity-ui
   ```
4. Access the UI at: [http://localhost:8501](http://localhost:8501)


### Running Only the API:

1. Navigate to the `api` directory:

   ```bash
   cd api
   ```
2. Build the Docker image:

   ```bash
   docker build -t toxicity-api .
   ```
3. Run the container:

   ```bash
   docker run -p 8000:8000 toxicity-api
   ```
4. Access the API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)


### Exploring the `toxicity_model` Directory (Optional):

For additional exploration or experimentation:

1. Download the datasets from Kaggle and place them in:

   * `toxicity_model/data/`
   * `toxicity_model/data/wikipedia_data/`

2. Inside the `toxicity_model` directory, run:

   ```bash
   ./setup.sh
   ```

3. Explore notebooks, models, and scripts as needed.

4. To deactivate the environment:

   ```bash
   deactivate
   ```


## Future Work

* Continue improving model performance through advanced techniques and hyperparameter tuning.
* Utilize bias analysis results to implement targeted mitigation strategies.
* Explore lightweight model alternatives for faster inference if needed.
* Expand the system to detect additional categories or languages.
