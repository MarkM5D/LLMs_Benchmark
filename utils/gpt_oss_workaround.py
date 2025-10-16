#!/usr/bin/env python3
"""
GPT-OSS Model Workaround for vLLM
Provides compatibility layer for gpt-oss-20b model with regular vLLM
"""

import json
import os
import tempfile
import shutil
from pathlib import Path

def create_gpt_oss_workaround(model_name_or_path):
    """
    Create a workaround for gpt-oss models to work with regular vLLM
    by creating a temporary directory with modified config
    """
    print(f"🔧 Creating gpt-oss workaround for: {model_name_or_path}")
    
    try:
        from huggingface_hub import snapshot_download
        import tempfile
        
        # Download the original model
        print("📥 Downloading original model...")
        original_path = snapshot_download(model_name_or_path, cache_dir='/workspace/.cache/huggingface')
        
        # Create temporary directory for modified model
        temp_dir = tempfile.mkdtemp(prefix="gpt_oss_workaround_")
        print(f"📁 Created workaround directory: {temp_dir}")
        
        # Copy all files from original to temp
        print("📋 Copying model files...")
        for item in os.listdir(original_path):
            src = os.path.join(original_path, item)
            dst = os.path.join(temp_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        # Modify config.json to be compatible with vLLM
        config_path = os.path.join(temp_dir, "config.json")
        if os.path.exists(config_path):
            print("⚙️ Modifying config.json for vLLM compatibility...")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create vLLM-compatible config
            # Map gpt_oss to mixtral (similar MoE architecture)
            config_backup = config.copy()
            
            # Try different compatibility approaches
            workaround_configs = [
                # Approach 1: Use Mixtral as base
                {
                    "model_type": "mixtral",
                    "architectures": ["MixtralForCausalLM"],
                },
                # Approach 2: Use Llama as base
                {
                    "model_type": "llama", 
                    "architectures": ["LlamaForCausalLM"],
                },
                # Approach 3: Use Qwen2 as base
                {
                    "model_type": "qwen2",
                    "architectures": ["Qwen2ForCausalLM"],
                }
            ]
            
            for i, workaround in enumerate(workaround_configs):
                workaround_dir = f"{temp_dir}_v{i+1}"
                shutil.copytree(temp_dir, workaround_dir)
                
                workaround_config_path = os.path.join(workaround_dir, "config.json")
                
                # Update config with workaround
                updated_config = config_backup.copy()
                updated_config.update(workaround)
                
                # Save modified config
                with open(workaround_config_path, 'w') as f:
                    json.dump(updated_config, f, indent=2)
                
                print(f"✅ Created workaround v{i+1}: {workaround['model_type']} compatibility")
            
            return [f"{temp_dir}_v{i+1}" for i in range(len(workaround_configs))]
        
        return [temp_dir]
        
    except Exception as e:
        print(f"❌ Workaround creation failed: {e}")
        return None

def test_vllm_with_workaround(workaround_paths):
    """Test vLLM with different workaround approaches"""
    print("🧪 Testing vLLM with workarounds...")
    
    for i, path in enumerate(workaround_paths):
        try:
            print(f"Testing workaround v{i+1}: {path}")
            
            # Try to initialize vLLM with this workaround
            from vllm import LLM
            
            llm = LLM(
                model=path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                max_model_len=2048,
                max_num_seqs=10,
                enforce_eager=True,
                trust_remote_code=True
            )
            
            print(f"✅ Workaround v{i+1} successful!")
            return llm, path
            
        except Exception as e:
            print(f"❌ Workaround v{i+1} failed: {e}")
            continue
    
    print("❌ All workarounds failed")
    return None, None

if __name__ == "__main__":
    # Test the workaround
    workaround_paths = create_gpt_oss_workaround("openai/gpt-oss-20b")
    if workaround_paths:
        llm, successful_path = test_vllm_with_workaround(workaround_paths)
        if llm:
            print(f"🎉 Success! Use path: {successful_path}")
        else:
            print("💥 All attempts failed")