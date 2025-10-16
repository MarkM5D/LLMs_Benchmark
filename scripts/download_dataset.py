#!/usr/bin/env python3
"""
Dataset Download and Preparation Script
Downloads the ShareGPT dataset for LLM benchmarking and converts it to the required format.
"""

import json
import random
import sys
import os
import requests
from pathlib import Path
import time

def download_with_progress(url, filename):
    """Download file with progress indicator."""
    print(f"📡 Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='', flush=True)
        
        print(f"\n✅ Download completed: {filename}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def create_sample_dataset():
    """Create a sample dataset if download fails."""
    print("🔧 Creating sample dataset...")
    
    sample_prompts = [
        "Explain the concept of artificial intelligence in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the main differences between machine learning and deep learning?",
        "Describe the process of how neural networks learn from data.",
        "Create a Python function that calculates the Fibonacci sequence.",
        "Explain quantum computing to someone who has never heard of it.",
        "Write a poem about the beauty of mathematics.",
        "What are the ethical implications of AI in healthcare?",
        "Design a simple algorithm to sort a list of numbers.",
        "Explain how natural language processing works.",
        "Write a dialogue between two AI systems discussing consciousness.",
        "What are the potential benefits and risks of autonomous vehicles?",
        "Create a step-by-step guide for training a neural network.",
        "Explain the concept of overfitting in machine learning.",
        "Write a creative story where AI helps solve climate change.",
        "What is the difference between supervised and unsupervised learning?",
        "Design a chatbot that can help students with math homework.",
        "Explain how computer vision enables machines to see.",
        "Write about the future of AI in 50 years.",
        "What are the main components of a recommendation system?"
    ]
    
    # Duplicate and vary prompts to reach 1000
    prompts = []
    for i in range(50):  # 20 base prompts * 50 = 1000 prompts
        base_prompt = sample_prompts[i % len(sample_prompts)]
        
        # Add variations
        variations = [
            base_prompt,
            f"Please {base_prompt.lower()}",
            f"Could you help me understand: {base_prompt.lower()}",
            f"I need assistance with: {base_prompt.lower()}",
            f"Can you elaborate on: {base_prompt.lower()}"
        ]
        
        for j, variation in enumerate(variations):
            if len(prompts) >= 1000:
                break
                
            prompts.append({
                'prompt': variation,
                'source': 'sample_dataset',
                'id': f"{i}_{j}"
            })
        
        if len(prompts) >= 1000:
            break
    
    return prompts[:1000]

def main():
    """Main dataset preparation function."""
    print("==========================================")
    print("LLM Benchmark Dataset Download")
    print("==========================================")
    
    # Create datasets directory
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    output_file = datasets_dir / "sharegpt_prompts.jsonl"
    
    # Check if dataset already exists
    if output_file.exists():
        print(f"✅ Dataset already exists: {output_file}")
        
        # Show statistics
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"📊 Found {len(lines)} prompts in existing dataset")
                
                if len(lines) > 0:
                    sample = json.loads(lines[0])
                    print(f"📋 Sample prompt: {sample.get('prompt', 'N/A')[:100]}...")
                
            choice = input("\n🔄 Regenerate dataset? (y/N): ").strip().lower()
            if choice not in ['y', 'yes']:
                print("✅ Using existing dataset")
                return True
        except Exception as e:
            print(f"⚠️  Error reading existing dataset: {e}")
            print("🔄 Will regenerate dataset...")
    
    # Try to download ShareGPT dataset
    print("\n📡 Attempting to download ShareGPT dataset...")
    
    urls_to_try = [
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/playground/data/dummy_conversation.json"
    ]
    
    prompts = None
    temp_file = datasets_dir / "sharegpt_raw.json"
    
    for url in urls_to_try:
        print(f"\n🔄 Trying: {url}")
        
        if download_with_progress(url, temp_file):
            try:
                print("🔄 Processing downloaded data...")
                
                with open(temp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"📊 Loaded {len(data)} conversations")
                
                # Extract prompts
                extracted_prompts = []
                for i, conversation in enumerate(data):
                    if isinstance(conversation, dict):
                        # Handle different data formats
                        conversations = conversation.get('conversations', [conversation])
                        
                        for turn in conversations:
                            if isinstance(turn, dict):
                                # Look for human messages
                                if (turn.get('from') == 'human' or 
                                    turn.get('role') == 'user' or 
                                    'human' in str(turn.get('speaker', '')).lower()):
                                    
                                    prompt_text = turn.get('value') or turn.get('content') or turn.get('text', '')
                                    
                                    if isinstance(prompt_text, str):
                                        prompt_text = prompt_text.strip()
                                        if 20 <= len(prompt_text) <= 2000:  # Filter reasonable lengths
                                            extracted_prompts.append({
                                                'prompt': prompt_text,
                                                'source': 'sharegpt',
                                                'id': f"sharegpt_{i}"
                                            })
                                            
                                            if len(extracted_prompts) >= 1000:
                                                break
                    
                    if len(extracted_prompts) >= 1000:
                        break
                
                if extracted_prompts:
                    prompts = extracted_prompts
                    print(f"✅ Extracted {len(prompts)} prompts from ShareGPT")
                    break
                else:
                    print("⚠️  No suitable prompts found in this dataset")
                    
            except Exception as e:
                print(f"❌ Error processing downloaded data: {e}")
                continue
            finally:
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()
    
    # Fallback to sample dataset if download failed
    if not prompts:
        print("\n🔧 Download failed, creating sample dataset...")
        prompts = create_sample_dataset()
        print(f"✅ Created {len(prompts)} sample prompts")
    
    # Shuffle for randomness
    random.shuffle(prompts)
    
    # Limit to 1000 prompts
    prompts = prompts[:1000]
    
    # Save as JSONL
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved to {output_file}")
        
        # Generate statistics
        total_chars = sum(len(p['prompt']) for p in prompts)
        avg_chars = total_chars / len(prompts) if prompts else 0
        max_chars = max((len(p['prompt']) for p in prompts), default=0)
        min_chars = min((len(p['prompt']) for p in prompts), default=0)
        
        print(f"\n� Dataset Statistics:")
        print(f"   Total prompts: {len(prompts)}")
        print(f"   Average length: {avg_chars:.1f} characters")
        print(f"   Min length: {min_chars} characters")
        print(f"   Max length: {max_chars} characters")
        
        # Show sample
        if prompts:
            print(f"\n📋 Sample prompt:")
            print(f"   {prompts[0]['prompt'][:200]}{'...' if len(prompts[0]['prompt']) > 200 else ''}")
        
        print(f"\n🎉 Dataset preparation completed!")
        print(f"📁 Dataset location: {output_file}")
        print(f"🚀 Ready for benchmarking!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Dataset preparation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Dataset preparation failed: {e}")
        sys.exit(1)