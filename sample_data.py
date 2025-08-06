"""
CAREERSYNC TRAINING - OPTIMIZED FOR RTX 3060 MOBILE
RTX 3060 Mobile: 6GB VRAM, Lower TDP, Mobile Cooling
"""

import torch
import gc
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json
from datetime import datetime
import psutil
import time

def clear_memory():
    """Aggressive memory clearing for 6GB GPU"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def check_rtx3060_mobile():
    """Check if we're on RTX 3060 Mobile and get specs"""
    clear_memory()
    
    print("="*60)
    print("RTX 3060 MOBILE - CAREERSYNC TRAINING SETUP")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_name = gpu_props.name
    total_memory = gpu_props.total_memory / 1e9
    free_memory = (gpu_props.total_memory - torch.cuda.memory_allocated(0)) / 1e9
    
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 Total VRAM: {total_memory:.1f} GB")
    print(f"💾 Free VRAM: {free_memory:.1f} GB")
    print(f"🖥️  RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    
    # RTX 3060 Mobile detection
    is_3060_mobile = "3060" in gpu_name and ("Mobile" in gpu_name or "Laptop" in gpu_name)
    
    if is_3060_mobile:
        print("✅ RTX 3060 Mobile detected - Using optimized config")
    elif "3060" in gpu_name:
        print("✅ RTX 3060 detected - Using mobile-optimized config anyway")
    else:
        print(f"⚠️  Different GPU detected. Optimizing for 6GB VRAM")
    
    if total_memory < 5.5:
        print("❌ Insufficient VRAM (need at least 5.5GB free)")
        return False
    
    return True

def load_optimized_model():
    """Load model optimized for RTX 3060 Mobile"""
    print("\n" + "="*50)
    print("LOADING MODEL - RTX 3060 MOBILE CONFIG")
    print("="*50)
    
    clear_memory()
    
    # Conservative settings for 6GB VRAM
    model_config = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_seq_length": 768,  # Reduced from 1024
        "dtype": None,
        "load_in_4bit": True,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    print(f"📁 Model: {model_config['model_name']}")
    print(f"📏 Max sequence length: {model_config['max_seq_length']}")
    print(f"🔢 Quantization: 4-bit")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**model_config)
        
        memory_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"✅ Model loaded successfully")
        print(f"💾 VRAM used: {memory_used:.2f} GB")
        
        if memory_used > 4.0:
            print("⚠️  High memory usage. Consider using smaller model if issues occur.")
        
        return model, tokenizer, model_config["max_seq_length"]
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ Out of memory loading model!")
            print("💡 Try closing other applications using GPU")
            print("💡 Or restart Python session")
        raise e

def setup_mobile_lora(model):
    """Conservative LoRA setup for mobile GPU"""
    print("\n" + "="*50)
    print("CONFIGURING LORA - MOBILE OPTIMIZATION")
    print("="*50)
    
    # Reduced LoRA parameters for 6GB VRAM
    lora_config = {
        "r": 8,  # Reduced from 16
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        "lora_alpha": 8,  # Reduced from 16
        "lora_dropout": 0.1,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": False,
    }
    
    model = FastLanguageModel.get_peft_model(model, **lora_config)
    
    memory_used = torch.cuda.memory_allocated(0) / 1e9
    print(f"✅ LoRA configured (r={lora_config['r']}, alpha={lora_config['lora_alpha']})")
    print(f"💾 VRAM used: {memory_used:.2f} GB")
    
    return model

def create_mobile_training_data():
    """Optimized dataset for faster training on mobile GPU"""
    print("\n" + "="*50)
    print("CREATING TRAINING DATASET - MOBILE OPTIMIZED")
    print("="*50)
    
    # Focused, high-quality examples for efficient training
    training_examples = [
        {
            "instruction": "Extract skills from resume and job description, provide match score.",
            "input": """Job Description: Python Developer
Required: Python, Django, PostgreSQL, Docker, Git

Resume: Alice
- Python (3 years)
- Django framework
- PostgreSQL database
- Docker containers
- Git version control""",
            "output": """SKILLS ANALYSIS:

Resume: Python, Django, PostgreSQL, Docker, Git
Required: Python, Django, PostgreSQL, Docker, Git

MATCHES (5/5): ✅ Python ✅ Django ✅ PostgreSQL ✅ Docker ✅ Git
MISSING (0/5): None

SCORE: 95/100 - Perfect match"""
        },
        {
            "instruction": "Extract skills from resume and job description, provide match score.",
            "input": """Job Description: Frontend Developer  
Required: React, JavaScript, CSS, HTML, Git

Resume: Bob
- React development (2 years)
- JavaScript ES6+
- CSS3 styling
- HTML5 markup
- Git workflow""",
            "output": """SKILLS ANALYSIS:

Resume: React, JavaScript, CSS3, HTML5, Git
Required: React, JavaScript, CSS, HTML, Git

MATCHES (5/5): ✅ React ✅ JavaScript ✅ CSS ✅ HTML ✅ Git
MISSING (0/5): None

SCORE: 92/100 - Excellent match"""
        },
        {
            "instruction": "Extract skills from resume and job description, provide match score.",
            "input": """Job Description: Data Analyst
Required: Python, SQL, Excel, Tableau, Statistics

Resume: Carol
- Python data analysis
- SQL queries
- Advanced Excel
- Basic Tableau
- Statistics background""",
            "output": """SKILLS ANALYSIS:

Resume: Python, SQL, Excel, Tableau, Statistics
Required: Python, SQL, Excel, Tableau, Statistics

MATCHES (5/5): ✅ Python ✅ SQL ✅ Excel ✅ Tableau ✅ Statistics
MISSING (0/5): None

SCORE: 88/100 - Strong match"""
        },
        {
            "instruction": "Extract skills from resume and job description, provide match score.",
            "input": """Job Description: DevOps Engineer
Required: AWS, Docker, Kubernetes, Python, Linux

Resume: Dave
- AWS cloud (2 years)
- Docker containers
- Basic Kubernetes
- Python scripting
- Linux administration""",
            "output": """SKILLS ANALYSIS:

Resume: AWS, Docker, Kubernetes, Python, Linux
Required: AWS, Docker, Kubernetes, Python, Linux

MATCHES (5/5): ✅ AWS ✅ Docker ✅ Kubernetes ✅ Python ✅ Linux
MISSING (0/5): None

SCORE: 85/100 - Good match"""
        },
        {
            "instruction": "Extract skills from resume and job description, provide match score.",
            "input": """Job Description: Full Stack Developer
Required: React, Node.js, MongoDB, JavaScript, Git

Resume: Eve
- React frontend (1.5 years)
- Node.js backend
- MySQL database
- JavaScript proficiency
- Git version control""",
            "output": """SKILLS ANALYSIS:

Resume: React, Node.js, MySQL, JavaScript, Git
Required: React, Node.js, MongoDB, JavaScript, Git

MATCHES (4/5): ✅ React ✅ Node.js ✅ JavaScript ✅ Git
MISSING (1/5): ❌ MongoDB (has MySQL)

SCORE: 78/100 - Good match, different database"""
        },
        {
            "instruction": "Extract skills from resume and job description, provide match score.",
            "input": """Job Description: Machine Learning Engineer
Required: Python, TensorFlow, PyTorch, AWS, Docker

Resume: Frank
- Python ML (2 years)
- TensorFlow projects
- Basic PyTorch
- Local development only
- No containerization""",
            "output": """SKILLS ANALYSIS:

Resume: Python, TensorFlow, PyTorch
Required: Python, TensorFlow, PyTorch, AWS, Docker

MATCHES (3/5): ✅ Python ✅ TensorFlow ✅ PyTorch
MISSING (2/5): ❌ AWS ❌ Docker

SCORE: 65/100 - Core ML skills present, needs cloud/deployment"""
        },
        {
            "instruction": "Extract skills from resume and job description, provide match score.",
            "input": """Job Description: iOS Developer
Required: Swift, Xcode, UIKit, Core Data, Git

Resume: Grace
- Swift programming (3 years)
- Xcode development
- UIKit frameworks
- Core Data storage
- Git version control""",
            "output": """SKILLS ANALYSIS:

Resume: Swift, Xcode, UIKit, Core Data, Git
Required: Swift, Xcode, UIKit, Core Data, Git

MATCHES (5/5): ✅ Swift ✅ Xcode ✅ UIKit ✅ Core Data ✅ Git
MISSING (0/5): None

SCORE: 93/100 - Comprehensive iOS skills"""
        },
        {
            "instruction": "Extract skills from resume and job description, provide match score.",
            "input": """Job Description: Backend API Developer
Required: Node.js, Express, MongoDB, REST, Git

Resume: Henry
- Node.js development
- Express framework
- PostgreSQL database
- REST API design
- Git workflow""",
            "output": """SKILLS ANALYSIS:

Resume: Node.js, Express, PostgreSQL, REST, Git
Required: Node.js, Express, MongoDB, REST, Git

MATCHES (4/5): ✅ Node.js ✅ Express ✅ REST ✅ Git
MISSING (1/5): ❌ MongoDB (has PostgreSQL)

SCORE: 82/100 - Strong backend skills, different database"""
        }
    ]
    
    print(f"✅ Created {len(training_examples)} optimized training examples")
    print("💡 Focused on clear, consistent skill extraction patterns")
    
    return training_examples

def get_mobile_training_config():
    """Training configuration optimized for RTX 3060 Mobile"""
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION - RTX 3060 MOBILE")
    print("="*50)
    
    # Very conservative settings for 6GB mobile GPU
    config = TrainingArguments(
        per_device_train_batch_size=1,        # Smallest possible
        gradient_accumulation_steps=4,         # Moderate accumulation
        warmup_steps=10,                       # Quick warmup
        max_steps=80,                         # Fewer steps for mobile
        learning_rate=2e-4,                   # Standard learning rate
        fp16=True,                            # Essential for memory
        logging_steps=5,                       # Frequent logging
        optim="adamw_8bit",                   # Memory efficient
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=f"careersync_rtx3060_{datetime.now().strftime('%m%d_%H%M')}",
        dataloader_pin_memory=False,          # Reduce memory pressure
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=20,                        # Save frequently
        evaluation_strategy="no",
        report_to=None,
        dataloader_num_workers=0,             # Single threaded for stability
        greater_is_better=False,
        load_best_model_at_end=False,
    )
    
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    
    print(f"📊 Batch size: {config.per_device_train_batch_size}")
    print(f"📊 Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"📊 Effective batch size: {effective_batch}")
    print(f"📊 Max steps: {config.max_steps}")
    print(f"📊 Learning rate: {config.learning_rate}")
    print(f"⚡ Estimated training time: 15-25 minutes")
    
    return config

def monitor_mobile_training(trainer, training_args):
    """Monitor training with mobile-specific warnings"""
    print("\n" + "="*50)
    print("STARTING TRAINING - RTX 3060 MOBILE")
    print("="*50)
    
    print("💡 Mobile GPU Training Tips:")
    print("   - Keep laptop plugged in and well-ventilated")
    print("   - Close unnecessary applications")
    print("   - Monitor temperatures if possible")
    print("   - Training will be automatically conservative")
    
    start_time = time.time()
    memory_before = torch.cuda.memory_allocated(0) / 1e9
    
    print(f"\n🚀 Starting training...")
    print(f"💾 VRAM before training: {memory_before:.2f} GB")
    
    try:
        trainer.train()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Training completed successfully!")
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
        print(f"💾 Peak VRAM: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ Out of memory after {(time.time()-start_time)/60:.1f} minutes")
            print("💡 Solutions for RTX 3060 Mobile:")
            print("   - Restart Python and try again")
            print("   - Reduce max_seq_length to 512")
            print("   - Use gradient_accumulation_steps=8")
            print("   - Close all other applications")
            return False
        else:
            raise e

def test_mobile_model(model, tokenizer):
    """Quick test optimized for mobile GPU"""
    print("\n" + "="*50)
    print("TESTING MODEL - MOBILE QUICK TEST")
    print("="*50)
    
    FastLanguageModel.for_inference(model)
    
    test_prompt = """<s>[INST] Extract skills from resume and job description, provide match score.

Job Description: Python Developer - Python, Django, PostgreSQL, Git
Resume: Test Candidate - Python programming, Django web framework, MySQL database, Git version control [/INST] """
    
    inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
    
    print("🧪 Running inference test...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Shorter for mobile
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output = result.split("[/INST] ")[-1]
    
    print("📋 Test Result:")
    print("-" * 40)
    print(output)
    print("-" * 40)
    
    return "SKILLS ANALYSIS" in output and "SCORE:" in output

# MAIN FUNCTION FOR RTX 3060 MOBILE
def train_on_rtx3060_mobile():
    """Complete training process optimized for RTX 3060 Mobile"""
    
    # Step 1: Check hardware
    if not check_rtx3060_mobile():
        return None
    
    # Step 2: Load model
    clear_memory()
    model, tokenizer, max_seq_length = load_optimized_model()
    
    # Step 3: Setup LoRA
    model = setup_mobile_lora(model)
    
    # Step 4: Prepare data
    training_examples = create_mobile_training_data()
    
    def format_chat(example):
        text = f"""<s>[INST] {example["instruction"]}\n\n{example["input"]} [/INST] {example["output"]}</s>"""
        return {"text": text}
    
    dataset = Dataset.from_list(training_examples)
    dataset = dataset.map(format_chat)
    
    # Step 5: Configure training
    training_args = get_mobile_training_config()
    
    # Step 6: Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        packing=False,
    )
    
    # Step 7: Train
    success = monitor_mobile_training(trainer, training_args)
    
    if success:
        # Step 8: Save model
        print(f"\n💾 Saving model to: {training_args.output_dir}")
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Save mobile-specific info
        mobile_info = {
            "gpu": "RTX 3060 Mobile",
            "optimization": "mobile_6gb",
            "training_time_minutes": "15-25",
            "max_seq_length": max_seq_length,
            "batch_size": training_args.per_device_train_batch_size,
            "date": datetime.now().isoformat()
        }
        
        with open(f"{training_args.output_dir}/mobile_config.json", "w") as f:
            json.dump(mobile_info, f, indent=2)
        
        # Step 9: Test
        if test_mobile_model(model, tokenizer):
            print("\n🎉 SUCCESS! Your CareerSync model is ready!")
            print(f"📁 Model location: {training_args.output_dir}")
            print("🔧 Ready for Chrome extension integration")
            return training_args.output_dir
        else:
            print("⚠️  Model saved but test failed. May need more training.")
            return training_args.output_dir
    else:
        print("\n❌ Training failed. Check solutions above.")
        return None

if __name__ == "__main__":
    print("🎮 RTX 3060 Mobile - CareerSync Training")
    print("📱 Optimized for 6GB VRAM mobile gaming laptop")
    
    model_path = train_on_rtx3060_mobile()
    
    if model_path:
        print(f"\n✅ Training complete! Model saved to: {model_path}")
    else:
        print("\n❌ Training failed. Check error messages above.")