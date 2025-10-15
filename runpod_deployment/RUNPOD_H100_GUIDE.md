# 🚀 RunPod H100 Benchmark Çalıştırma Rehberi

## 📋 Adım Adım RunPod H100 Kullanımı

### 1️⃣ **RunPod Instance Başlatma**

1. **RunPod'a Login Olun**: https://runpod.io
2. **GPU Cloud** → **Deploy** 
3. **Template Seçin**: 
   - `PyTorch` veya `CUDA` template seçin
   - Veya `RunPod Pytorch` template'ini kullanın
4. **GPU Seçin**: 
   - **H100 80GB** kartını seçin (en güçlü seçenek)
   - Fiyat: ~$4-5/saat
5. **Deploy** butonuna basın

### 2️⃣ **Pod'a Bağlanma**

Pod hazır olunca:
1. **"Connect"** → **"Start Web Terminal"** 
2. Terminal açılacak, şu komutla başlayın:

```bash
cd /workspace
pwd  # Şu anda /workspace'de olmalısınız
```

### 3️⃣ **Kod Upload Etme Yöntemleri**

#### **Yöntem A: Git Clone (Önerilen)**
```bash
# Eğer GitHub'a push ettiyseniz:
git clone https://github.com/[kullanıcı-adınız]/[repo-adınız].git
cd [repo-adınız]
```

#### **Yöntem B: Manual Upload**
1. RunPod File Manager kullanın
2. `runpod_deployment/` klasörünü `/workspace/` altına upload edin
3. Terminal'de:
```bash
cd /workspace/runpod_deployment
ls -la  # Dosyaları kontrol edin
```

#### **Yöntem C: Wget (Eğer zip dosyası varsa)**
```bash
cd /workspace
wget [zip-dosyası-linki]
unzip [dosya-adı].zip
cd [klasör-adı]
```

### 4️⃣ **H100 Environment Setup**

```bash
# Çalıştırma izinleri verin
chmod +x setup_env.sh
chmod +x runpod_startup.sh

# Otomatik H100 setup (TEK KOMUT!)
./runpod_startup.sh
```

Bu komut şunları yapacak:
- H100 Hopper mimarisi ayarları
- PyTorch, vLLM, SGLang, TensorRT-LLM kurulumu
- Dataset download
- GPU optimizasyonları

### 5️⃣ **H100 Verification**

```bash
# H100 kartını kontrol edin
nvidia-smi

# H100 optimizasyonlarını doğrulayın
python3 h100_optimize.py
```

**Beklenen Çıktı:**
```
🎮 GPU: NVIDIA H100 80GB HBM3
🔢 Compute Capability: 9.0
✅ H100 GPU detected!
✅ Hopper architecture (9.0+) confirmed
```

### 6️⃣ **Benchmark Çalıştırma**

```bash
# Ana benchmark scripti (3 engine'i test eder)
python3 run_all.py

# Veya tek tek çalıştırmak için:
python3 benchmarks/vllm_benchmark.py
python3 benchmarks/sglang_benchmark.py  
python3 benchmarks/tensorrtllm_benchmark.py

# Sonuçları aggregate etmek için:
python3 aggregate_results.py
```

### 7️⃣ **Sonuçları Görüntüleme**

#### **Real-time Monitoring**
```bash
# Ayrı terminal açın, GPU kullanımını izleyin
watch -n 1 nvidia-smi

# Memory ve CPU monitoring
htop
```

#### **Benchmark Sonuçları**
```bash
# Sonuç dosyalarını listeleyin
ls -la benchmarks/results/

# JSON sonuçlarını pretty print ile görün
python3 -m json.tool benchmarks/results/vllm_results.json

# Hızlı özet görüntüleme
cat benchmarks/results/benchmark_summary.json | jq '.'
```

#### **Grafik Sonuçlar**
```bash
# Eğer matplotlib kuruluysa grafik oluştur
python3 -c "
import json
import matplotlib.pyplot as plt

# Throughput comparison grafiği
with open('benchmarks/results/aggregate_results.json', 'r') as f:
    data = json.load(f)

engines = []
throughputs = []
for engine, results in data.get('engine_comparison', {}).items():
    engines.append(engine)
    throughputs.append(results.get('throughput_tokens_per_second', 0))

plt.figure(figsize=(10, 6))
plt.bar(engines, throughputs, color=['blue', 'green', 'red'])
plt.title('H100 LLM Engine Throughput Comparison')
plt.ylabel('Tokens per Second')
plt.savefig('benchmarks/results/throughput_comparison.png', dpi=150, bbox_inches='tight')
print('📊 Grafik kaydedildi: benchmarks/results/throughput_comparison.png')
"
```

### 8️⃣ **Sonuçları Download Etme**

```bash
# Tüm sonuçları zip'leyin
cd /workspace
zip -r benchmark_results.zip runpod_deployment/benchmarks/results/

# RunPod File Manager'dan indirebilirsiniz
ls -la benchmark_results.zip
```

## 🔧 **Troubleshooting Komutları**

### **GPU Problemleri**
```bash
# GPU detection
nvidia-smi
lspci | grep -i nvidia

# CUDA version
nvcc --version
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

### **Memory Problemleri**
```bash
# GPU memory temizle
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')"

# System memory kontrol
free -h
df -h
```

### **Package Problemleri**
```bash
# Inference engine kontrolü
python3 -c "
try:
    import vllm; print(f'✅ vLLM {vllm.__version__}')
except: print('❌ vLLM not installed')

try:
    import sglang; print(f'✅ SGLang {sglang.__version__}')
except: print('❌ SGLang not installed')

try:
    import tensorrt_llm; print(f'✅ TensorRT-LLM {tensorrt_llm.__version__}')
except: print('❌ TensorRT-LLM not installed')
"

# Manuel reinstall
pip install --force-reinstall --no-cache-dir vllm
```

## 📊 **Beklenen H100 Sonuçları**

### **Tipik Performans Metrikleri:**
- **vLLM**: 8,000-12,000 tokens/sec
- **SGLang**: 7,000-10,000 tokens/sec  
- **TensorRT-LLM**: 10,000-15,000 tokens/sec
- **Latency P95**: <200ms
- **GPU Utilization**: 85-95%
- **Memory Usage**: 40-70GB (80GB H100'den)

### **Başarı İndikatörleri:**
```bash
# Bu çıktıları görmelisiniz:
✅ H100 GPU detected!
✅ All inference engines installed
✅ Benchmarks completed successfully
✅ Results saved to benchmarks/results/
```

## 🎯 **Pro Tips**

### **Performance Optimization**
```bash
# Maximum performance için
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
ulimit -n 65536

# Model caching
export TRANSFORMERS_CACHE=/workspace/cache
export HF_DATASETS_CACHE=/workspace/cache
mkdir -p /workspace/cache
```

### **Cost Optimization**
```bash
# İşlem bitince pod'u durdurun (ücretlendirme durur)
# Önemli dosyalar /workspace'de kalır

# Log monitoring
tail -f benchmarks/logs/*.log
```

## 🚨 **Acil Durumlar**

### **Pod Donması**
1. RunPod dashboard'dan "Restart" yapın
2. `/workspace` dosyalarınız korunur
3. `cd /workspace/runpod_deployment && python3 run_all.py` ile devam edin

### **Out of Memory**
```bash
# Batch size küçültün
export BENCHMARK_BATCH_SIZE=16  # Default 32 yerine
python3 run_all.py
```

Bu rehberle H100'de tam performans alacaksınız! 🚀⚡