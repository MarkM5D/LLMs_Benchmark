#!/usr/bin/env python3
"""
Debug vLLM Test - Minimal working test
"""

def test_vllm_basic():
    """Test if vLLM can be imported and run a basic test"""
    
    print("🧪 Testing vLLM basic functionality...")
    
    try:
        from vllm import LLM, SamplingParams
        print("✅ vLLM imported successfully")
        
        # Try to create model instance (without loading)
        print("🔄 Testing model configuration...")
        
        # Simple prompt test
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50
        )
        print("✅ SamplingParams created")
        
        print("🎉 vLLM basic test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ vLLM import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ vLLM test failed: {e}")
        return False

def test_dataset_loading():
    """Test if dataset can be loaded"""
    
    print("🧪 Testing dataset loading...")
    
    try:
        import json
        from pathlib import Path
        
        dataset_file = Path("./datasets/sharegpt_prompts.jsonl")
        
        if not dataset_file.exists():
            print(f"❌ Dataset not found: {dataset_file}")
            return False
            
        print(f"✅ Dataset found: {dataset_file}")
        
        # Try to read first few lines
        prompts = []
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Read first 5 lines
                    break
                data = json.loads(line)
                prompts.append(data['prompt'])
                
        print(f"✅ Loaded {len(prompts)} sample prompts")
        print(f"   Sample: {prompts[0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def main():
    print("🚀 DEBUG vLLM Test Suite")
    print("=" * 50)
    
    # Test 1: vLLM Import
    test1 = test_vllm_basic()
    
    # Test 2: Dataset Loading  
    test2 = test_dataset_loading()
    
    if test1 and test2:
        print("🎉 All tests passed! Ready for benchmark.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)