#!/usr/bin/env python3
"""
Training Data Generation Script for CareerSync

This script helps generate training data by using your existing ChatGPT setup
to create examples for fine-tuning your own model.
"""

import json
import os
import time
from typing import List, Dict
import openai
from openai import OpenAI
import random

# Sample resume templates
RESUME_TEMPLATES = [
    {
        "name": "Software Engineer",
        "skills": ["JavaScript", "React", "Node.js", "MongoDB", "Git"],
        "experience": "3 years building web applications, REST APIs, responsive design",
        "education": "BS Computer Science"
    },
    {
        "name": "Data Scientist", 
        "skills": ["Python", "R", "SQL", "Pandas", "Scikit-learn", "TensorFlow"],
        "experience": "2 years, predictive modeling, data visualization, statistical analysis",
        "education": "PhD Statistics"
    },
    {
        "name": "DevOps Engineer",
        "skills": ["AWS", "Docker", "Kubernetes", "Terraform", "Jenkins", "Python"],
        "experience": "5 years, infrastructure automation, CI/CD pipelines, cloud migration",
        "education": "BS Computer Engineering"
    },
    {
        "name": "Mobile Developer",
        "skills": ["React Native", "JavaScript", "TypeScript", "Redux", "Firebase"],
        "experience": "4 years, built 8+ mobile apps, cross-platform development",
        "education": "BS Software Engineering"
    },
    {
        "name": "Full Stack Developer",
        "skills": ["Python", "Django", "PostgreSQL", "Docker", "AWS", "React"],
        "experience": "5 years, e-commerce platforms, API development, cloud deployment",
        "education": "MS Computer Science"
    }
]

# Sample job descriptions
JOB_TEMPLATES = [
    {
        "title": "Senior Frontend Developer",
        "requirements": ["React", "TypeScript", "Next.js", "GraphQL", "Testing", "Redux"],
        "experience": "4+ years experience",
        "responsibilities": "Building scalable web applications, mentoring junior developers"
    },
    {
        "title": "Backend Engineer", 
        "requirements": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes"],
        "experience": "3+ years backend development",
        "responsibilities": "Microservices architecture, high-traffic applications"
    },
    {
        "title": "Machine Learning Engineer",
        "requirements": ["Python", "PyTorch", "MLOps", "Docker", "AWS", "MLflow"],
        "experience": "3+ years ML experience",
        "responsibilities": "Model deployment, monitoring, pipeline automation"
    },
    {
        "title": "Senior Mobile Developer",
        "requirements": ["React Native", "TypeScript", "Native iOS/Android", "Testing", "CI/CD"],
        "experience": "5+ years experience",
        "responsibilities": "Lead mobile development, architecture decisions, mentoring"
    },
    {
        "title": "Cloud Engineer",
        "requirements": ["AWS", "Azure", "Kubernetes", "Terraform", "Monitoring", "Security"],
        "experience": "4+ years cloud experience",
        "responsibilities": "Infrastructure as Code, automated deployments, system reliability"
    }
]

class TrainingDataGenerator:
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=api_key)
        
    def generate_resume(self, template: Dict) -> str:
        """Generate a resume based on template with variations"""
        name = f"{random.choice(['John', 'Jane', 'Mike', 'Sarah', 'Alex', 'Emily'])} {random.choice(['Smith', 'Johnson', 'Chen', 'Rodriguez', 'Thompson', 'Davis'])}"
        
        # Add some variation to skills
        skills = template["skills"].copy()
        if random.random() > 0.3:  # 70% chance to add extra skills
            extra_skills = ["Git", "Linux", "Agile", "Scrum", "JIRA", "Figma", "Postman"]
            skills.extend(random.sample(extra_skills, random.randint(1, 3)))
        
        resume = f"""
{name}
{template["name"]}
Skills: {", ".join(skills)}
Experience: {template["experience"]}
Education: {template["education"]}
"""
        return resume.strip()
    
    def generate_job_description(self, template: Dict) -> str:
        """Generate a job description based on template"""
        job_desc = f"""
{template["title"]}
Requirements: {", ".join(template["requirements"])}
{template["experience"]}
Responsibilities: {template["responsibilities"]}
"""
        return job_desc.strip()
    
    def analyze_with_gpt(self, resume: str, job_desc: str) -> str:
        """Use GPT to analyze resume vs job description"""
        prompt = f"""
Analyze this resume against the job description and provide a detailed analysis in JSON format.

Resume:
{resume}

Job Description:
{job_desc}

Please provide your analysis in this exact JSON format:
{{
  "match_score": [0-100 integer],
  "missing_skills": [list of missing skills],
  "experience_gap": "description of experience gap or 'None' if adequate",
  "strengths": [list of candidate strengths],
  "suggestions": [list of specific improvement suggestions],
  "recommended_actions": [list of concrete action items]
}}

Be thorough and specific in your analysis.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" for cheaper option
                messages=[
                    {"role": "system", "content": "You are a resume analysis expert. Provide thorough, accurate analysis in the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None
    
    def generate_training_example(self, resume_template: Dict, job_template: Dict) -> Dict:
        """Generate a complete training example"""
        resume = self.generate_resume(resume_template)
        job_desc = self.generate_job_description(job_template)
        
        print(f"Generating analysis for {resume_template['name']} -> {job_template['title']}")
        
        analysis = self.analyze_with_gpt(resume, job_desc)
        
        if analysis:
            return {
                "instruction": "Analyze this resume against the job description. Provide a match score (0-100), identify missing skills, and suggest improvements.",
                "input": f"Resume:\n{resume}\n\nJob Description:\n{job_desc}",
                "output": analysis
            }
        return None
    
    def generate_dataset(self, num_examples: int = 100) -> List[Dict]:
        """Generate a complete training dataset"""
        training_data = []
        
        for i in range(num_examples):
            # Randomly select templates
            resume_template = random.choice(RESUME_TEMPLATES)
            job_template = random.choice(JOB_TEMPLATES)
            
            example = self.generate_training_example(resume_template, job_template)
            
            if example:
                training_data.append(example)
                print(f"Generated example {i+1}/{num_examples}")
            else:
                print(f"Failed to generate example {i+1}/{num_examples}")
            
            # Add delay to respect API rate limits
            time.sleep(1)
        
        return training_data

def main():
    """Main function to generate training data"""
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        print("You can get one from: https://platform.openai.com/api-keys")
        return
    
    # Initialize generator
    generator = TrainingDataGenerator(api_key)
    
    # Generate training data
    num_examples = int(input("How many training examples to generate? (recommended: 100-1000): ") or "100")
    
    print(f"Generating {num_examples} training examples...")
    print("This may take several minutes...")
    
    training_data = generator.generate_dataset(num_examples)
    
    # Save to file
    output_file = "../data/processed/training_data.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Generated {len(training_data)} examples")
    print(f"Saved to: {output_file}")
    print(f"Estimated cost: ${len(training_data) * 0.02:.2f}")
    
    # Display sample
    if training_data:
        print("\nSample generated example:")
        print(json.dumps(training_data[0], indent=2)[:500] + "...")

if __name__ == "__main__":
    main()