#!/usr/bin/env python3
"""
Create sample dataset for immediate testing
Creates datasets/sharegpt_prompts.jsonl with sample prompts if download fails
"""

import json
import os
from pathlib import Path

def create_sample_dataset():
    """Create a comprehensive sample dataset for testing."""
    
    # Create datasets directory
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    output_file = datasets_dir / "sharegpt_prompts.jsonl"
    
    print("🔧 Creating sample dataset for immediate testing...")
    
    # Diverse sample prompts for different test scenarios
    base_prompts = [
        # Technical prompts (good for throughput tests)
        "Explain the differences between machine learning and deep learning in detail.",
        "Write a Python function that implements the quicksort algorithm with comments.",
        "Describe how a neural network processes information from input to output.",
        "What are the key components of a transformer architecture in natural language processing?",
        "Explain the concept of backpropagation in neural network training.",
        
        # Creative prompts (good for diverse generation)
        "Write a short story about an AI that discovers it has emotions.",
        "Create a poem about the relationship between technology and nature.",
        "Describe a futuristic city where humans and AI coexist peacefully.",
        "Write a dialogue between two robots discussing the meaning of consciousness.",
        "Compose a song lyrics about the journey of learning artificial intelligence.",
        
        # Analytical prompts (good for structured responses)
        "Compare and contrast supervised vs unsupervised learning approaches.",
        "Analyze the pros and cons of using cloud computing for machine learning.",
        "Evaluate the ethical implications of AI in healthcare decision making.",
        "Discuss the advantages and disadvantages of different programming languages for AI.",
        "Examine the potential impact of quantum computing on artificial intelligence.",
        
        # Instructional prompts (good for JSON structure tests)
        "Provide a step-by-step guide for training your first neural network.",
        "List the essential tools and libraries for machine learning in Python.",
        "Describe the process of data preprocessing for machine learning projects.",
        "Explain how to evaluate the performance of a classification model.",
        "Outline the stages of a typical machine learning project lifecycle.",
        
        # Problem-solving prompts (good for latency tests)
        "How would you optimize a slow database query?",
        "What's the best approach to handle missing data in a dataset?",
        "How can you prevent overfitting in a machine learning model?",
        "What strategies would you use to improve model accuracy?",
        "How do you choose the right algorithm for a specific problem?",
        
        # Conversational prompts
        "Tell me about your favorite programming language and why you like it.",
        "What advice would you give to someone starting their AI journey?",
        "How has artificial intelligence changed the world in the past decade?",
        "What do you think the future of AI will look like in 10 years?",
        "Explain a complex AI concept using simple analogies.",
        
        # Technical deep-dive prompts
        "Implement a basic attention mechanism in PyTorch with detailed explanations.",
        "Design a recommendation system architecture for an e-commerce platform.",
        "Explain the mathematics behind gradient descent optimization.",
        "Create a complete data pipeline for real-time machine learning inference.",
        "Describe how to implement distributed training for large language models.",
        
        # Business and application prompts
        "How can AI be applied to improve customer service in retail?",
        "What are the challenges of implementing AI in autonomous vehicles?",
        "Discuss the role of AI in modern cybersecurity systems.",
        "How is artificial intelligence transforming the healthcare industry?",
        "What are the key considerations for AI governance in enterprise settings?"
    ]
    
    # Generate variations to reach 1000 prompts
    prompts = []
    variation_patterns = [
        "{}",
        "Please help me understand: {}",
        "Can you explain {}",
        "I need detailed information about: {}",
        "Could you elaborate on: {}",
        "Provide insights into: {}",
        "Give me a comprehensive overview of: {}",
        "Break down the concept of: {}",
        "Walk me through: {}",
        "Help me learn about: {}"
    ]
    
    # Create 1000 prompts with variations
    prompt_id = 0
    for base_prompt in base_prompts:
        for pattern in variation_patterns:
            if len(prompts) >= 1000:
                break
            
            # Apply variation pattern
            if '{}' in pattern:
                varied_prompt = pattern.format(base_prompt.lower() if 'help' in pattern or 'explain' in pattern else base_prompt)
            else:
                varied_prompt = base_prompt
            
            prompts.append({
                'prompt': varied_prompt,
                'source': 'sample_dataset',
                'id': f"sample_{prompt_id:04d}",
                'category': 'general',
                'length_category': 'medium' if len(varied_prompt) < 100 else 'long'
            })
            
            prompt_id += 1
        
        if len(prompts) >= 1000:
            break
    
    # Ensure we have exactly 1000 prompts
    while len(prompts) < 1000:
        # Add more variations of existing prompts
        base_idx = len(prompts) % len(base_prompts)
        base_prompt = base_prompts[base_idx]
        
        additional_variations = [
            f"What are your thoughts on: {base_prompt.lower()}",
            f"Share your knowledge about: {base_prompt.lower()}",
            f"Discuss in detail: {base_prompt.lower()}",
            f"Provide examples for: {base_prompt.lower()}"
        ]
        
        variation = additional_variations[len(prompts) % len(additional_variations)]
        prompts.append({
            'prompt': variation,
            'source': 'sample_dataset_extended',
            'id': f"sample_{prompt_id:04d}",
            'category': 'general',
            'length_category': 'medium'
        })
        prompt_id += 1
    
    # Limit to exactly 1000 and add some metadata
    prompts = prompts[:1000]
    
    # Save as JSONL
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
        
        # Calculate statistics
        total_chars = sum(len(p['prompt']) for p in prompts)
        avg_chars = total_chars / len(prompts)
        max_chars = max(len(p['prompt']) for p in prompts)
        min_chars = min(len(p['prompt']) for p in prompts)
        
        print(f"✅ Sample dataset created: {output_file}")
        print(f"\n📈 Dataset Statistics:")
        print(f"   Total prompts: {len(prompts)}")
        print(f"   Average length: {avg_chars:.1f} characters")
        print(f"   Min length: {min_chars} characters")
        print(f"   Max length: {max_chars} characters")
        
        # Show a sample
        print(f"\n📋 Sample prompts:")
        for i in [0, 250, 500, 750]:
            if i < len(prompts):
                prompt_text = prompts[i]['prompt']
                preview = prompt_text[:80] + "..." if len(prompt_text) > 80 else prompt_text
                print(f"   {i+1:3d}: {preview}")
        
        print(f"\n🎉 Ready for benchmarking!")
        print(f"📁 Dataset location: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample dataset: {e}")
        return False

if __name__ == "__main__":
    create_sample_dataset()