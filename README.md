# CareerSync LLM Fine-tuning

> Fine-tuning Llama 3.1 8B for intelligent resume analysis and job matching

##  Project Overview

This project fine-tunes a Llama 3.1 8B model to replace OpenAI's GPT in the CareerSync Chrome extension. The custom model analyzes resumes against job descriptions, providing match scores, identifying missing skills, and suggesting improvements.

##  Features

- **Resume Analysis**: Intelligent parsing and analysis of PDF/DOCX resumes
- **Job Matching**: Accurate matching between candidate skills and job requirements
- **Skill Gap Analysis**: Identifies missing skills and suggests improvements
- **Cost Effective**: 10x cheaper than OpenAI API calls
- **Fast Inference**: Optimized for real-time Chrome extension usage

##  Project Structure

```
├── data/                    # Training datasets
├── notebooks/               # Jupyter notebooks for training
├── scripts/                 # Python utilities and training scripts
├── configs/                 # Training and model configurations
├── models/                  # Saved model checkpoints
└── docs/                    # Documentation and guides
```

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/careersync-llm-finetuning.git
cd careersync-llm-finetuning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Training Data
```bash
python scripts/data_preparation.py --input data/raw --output data/processed
```

### 4. Fine-tune the Model
Open `notebooks/fine_tuning_colab.ipynb` in Google Colab and follow the step-by-step guide.

### 5. Deploy the Model
```bash
python scripts/deploy.py --model models/careersync-llama-8b --platform huggingface
```

##  Training Data Format

The model expects training data in this format:

```json
{
  "instruction": "Analyze this resume against the job description...",
  "input": "Resume: [resume_text]\n\nJob Description: [job_text]",
  "output": "{\"match_score\": 78, \"missing_skills\": [...], \"suggestions\": [...]}"
}