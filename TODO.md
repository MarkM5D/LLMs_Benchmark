# TODO.md — DETAILED

**Proje:** LLM Inference Engine Benchmark (vLLM, SGLang, TensorRT-LLM)
**Hedef Ortam:** RunPod — 1x NVIDIA H100 80GB
**Model:** `gpt-oss-20b`

---

## Amaç

Her bir inference motorunu (vLLM, SGLang, TensorRT-LLM) **temiz kurulum** ile tek tek yükleyip test etmek; her motor için 3 test (Throughput / JSON-struct / Low-latency) çalıştırmak; sonuçları zaman damgalı klasörlere kaydetmek; her motor sonrası tam uninstall yapıp bir sonraki motora geçmek.

---

## 1. Klasör & Dosya Yapısı (kesin)

```
benchmark_project/
├── TODO.md
├── README.md
├── datasets/
│   └── sharegpt_prompts.jsonl
├── scripts/
│   ├── install_vllm.sh
│   ├── uninstall_vllm.sh
│   ├── install_sglang.sh
│   ├── uninstall_sglang.sh
│   ├── install_tensorrt.sh
│   ├── uninstall_tensorrt.sh
│   ├── run_benchmark.py
│   ├── run_engine.py
│   ├── collect_metrics.sh
│   ├── save_results.sh
│   └── env_info.sh
├── results/
│   └── (timestamped folders)
├── logs/
│   └── install_logs/
└── docs/
    └── benchmark_plan.md
```

---

## 2. Ortam & Versiyon Politikası (kritik)

* **Her motor için ayrı sanal ortam (venv)** kullan. `llmbench_<engine>_venv`.
* PyTorch/CUDA, cuDNN, TensorRT gibi altyapı farkları için **tam uninstall + venv silme** politikası uygulanacak.
* Sistem genel paketleri değiştirilmeyecek; yalnızca Python venv içinde paketler yönetilecek.
* Kernel seviyesinde çakışma olursa RunPod instance yeniden başlatılacak.

---

## 3. Kurulum / Kaldırma Prensipleri

Her motor için şu adımlar: `prepare-clean -> create-venv -> install-deps -> validate -> run-small-test -> run-full-bench -> save-results -> uninstall-clean`.

### 3.1 `prepare-clean`

* (a) Active venv varsa deactivate et.
* (b) Eğer önceki venv varsa `rm -rf` ile kaldır.
* (c) Pip cache temizle: `pip cache purge`.
* (d) GPU driver/jar değişiklik gerekirse RunPod restart.

### 3.2 venv oluşturma örneği

```bash
python3 -m venv /opt/llmbench/venvs/llmbench_vllm
source /opt/llmbench/venvs/llmbench_vllm/bin/activate
pip install --upgrade pip setuptools wheel
```

### 3.3 install_X.sh / uninstall_X.sh

Her `install_*.sh` şu bölümleri içerir:

1. venv oluştur (veya reuse istenir ise bunu parametreleştir)
2. PyTorch uygun versiyonunu kur
3. Motorun resmi kurulumu (pip install veya git clone + pip install -e .)
4. Model indirme scripti veya HuggingFace cache yönergesi
5. Küçük doğrulama çağrısı (ör. bir prompt ile single token generation)
6. Kurulum logunu `logs/install_logs/{engine}_install_$(date...).log` içinde sakla

Her `uninstall_*.sh` şu adımları uygular:

1. deactivate venv
2. pip uninstall -y <engine packages>
3. rm -rf venv dizini
4. pip cache purge
5. opsiyonel: docker image/containers sil (TensorRT için)
6. log kaydı

---

## 4. Motor-spesifik Öneriler (kurulum sırasında dikkat)

### vLLM

* **Hedef PyTorch:** `torch==2.3.0+cu121` (H100 için CUDA 12.1). Eğer RunPod'un template'i CUDA 12.2 veya farklıysa, PyTorch wheel'leri uyumlu olacak şekilde seç.
* **Kurulum adımları:**

  * `pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.3.0+cu121 torchvision ...`
  * `pip install git+https://github.com/vllm/vllm.git@main` veya `pip install vllm`
* **Model:** Hugging Face hub'dan `gpt-oss-20b` indirme. Eğer quantize gerekiyorsa bunun için `bitsandbytes` veya vLLM destekli quantize yolları izlenecek.
* **Örnek küçük test:** tek prompt ile `llm.generate()` çağırıp ilk token sürelerini kontrol et.

### SGLang

* **Versiyon/pyTorch:** SGLang dokümanına göre PyTorch versiyonu değişebilir; genelde `torch==2.2+cu121` veya `2.3` desteklenir. Kurulum sırasında requirements.txt dikkatle takip edilecek.
* **RadixAttention vs PagedAttention:** Kurulum sonrası config parametreleri ile `RadixAttention` etkinleştirme opsiyonları denenecek.
* **Kurulum:** resmi repo veya pip package (doküman referansına göre). `pip install sglang` veya `git clone` şeklinde.

### TensorRT-LLM

* **Önemli:** Genelde Docker tabanlı veya NVIDIA tarafından sağlanan extension paketleri ile kurulur. Root erişimi ve uygun TensorRT/CUDA sürümü gerekir.
* **Pipeline:** HF model -> ONNX export -> TensorRT compile/optimize -> TRT engine
* **Kurulum yolu (özet):**

  1. Install correct NVIDIA drivers + CUDA + TensorRT (RunPod image ile geliyor genelde)
  2. Install `tensorrt` Python package (veya NVIDIA NGC container)
  3. Kullanılan araç: `trt-llm` veya `TensorRT-LLM` dokümanındaki converter
* **Not:** Eğer RunPod'un image'ı TensorRT ile geliyorsa, Docker kullanmadan host ortamda çalıştırma yapılabilir; değilse Docker container tercih et.

---

## 5. Benchmark Sürüş Planı — Tek Satır Komutlar

Tüm testler Python driver `scripts/run_benchmark.py` ile çalıştırılacak.

**Örnek:**

```bash
# vllm throughput testi
source /opt/llmbench/venvs/llmbench_vllm/bin/activate
python scripts/run_benchmark.py \
  --engine vllm \
  --model gpt-oss-20b \
  --test throughput \
  --concurrency 100 \
  --batch_size 8 \
  --max_num_seqs 200 \
  --runs 3 \
  --outdir results/$(date -u +%F_%H.%M)_test
```

`run_benchmark.py` sorumlulukları:

* Dataset'ten prompt oku
* engine-specific adapter (vllm/sge/tensorrt) ile `.generate()` veya muadil çağrıyı yap
* zaman ölçümlerini (first token, total tokens produced, per-run throughput) topla
* GPU metriklerini per-second `nvidia-smi` snapshot ile yakala (collect_metrics.sh kullan)
* Çıktıyı `{engine}.json` içine yaz

---

## 6. `run_engine.py` — Motor-Adaptör Mantığı

Bu script motorlara göre:

* `load_model(engine, model_name, config)`
* `generate(engine_handle, prompt, sampling_params)`
* Ortak sampling parametreleri: `max_tokens`, `temperature`, `top_p`, `batch_size`, `max_new_tokens`.

Motorlar arasında interface standardize edilecek; örn dönüş şöyle olmalı:

```py
{
  "tokens_generated": 123,
  "first_token_latency_ms": 12.3,
  "total_latency_ms": 345.6,
  "raw_text": "..."
}
```

---

## 7. Data Set & Workload

* **Dataset:** `datasets/sharegpt_prompts.jsonl` — tipik diyalog promptları.
* **Throughput workload:** concurrency yüksek (örn 100), batch_size moderate (8), hedef GPU util ~ %80-95.
* **JSON-struct test:** özel şema verilecek; her motorun native methodu kullanılacaksa ona göre parametre seti.
* **Low-latency test:** concurrency=1, batch_size=1, 500+ tekrar.

---

## 8. Metrics — JSON Şeması (motor sonuç dosyası)

`{engine}.json` örnek anahtarlar:

```json
{
  "engine": "vllm",
  "timestamp": "2025-10-16T14:07:00Z",
  "config": {"torch_version":"2.3.0+cu121", "cuda":"12.1"},
  "benchmarks": {
    "throughput": {"runs":[...], "avg_tokens_per_sec": 12345.6, "p95_latency_ms":234.5},
    "json_struct": {...},
    "low_latency": {...}
  },
  "system": {"gpu_util_avg": 92.3, "gpu_mem_peak_gb": 68.2},
  "logs": "path/to/logs.log"
}
```

---

## 9. Monitoring & Collection

* `collect_metrics.sh` çalıştır: `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1` ile per-sec snapshot
* `pidstat -r -u 1` veya `dstat` ile CPU/RAM
* `env_info.sh` ile `python -c 'import torch; print(torch.__version__, torch.version.cuda)'` gibi

---

## 10. Analiz & Summary

* `scripts/analyze_results.py` üç `{engine}.json`'i okuyup `summary.json` üretir.
* Rapor markdown `docs/report_template.md` doldurulur.
* Grafikleri Jupyter `notebooks/analysis.ipynb` içinde üret.

---

## 11. Hata Yönetimi ve Recovery

* Kurulum başarısızsa `logs/install_logs/{engine}_install_*.log` kontrol et.
* Kütüphane çakışması varsa: `deactivate && rm -rf /opt/llmbench/venvs/llmbench_<engine>` sonra yeniden başlat.
* Eğer GPU driver hata veriyorsa RunPod UI üzerinden restart.

---

## 12. Checklist — Her Motor İçin (adım adım)

* [ ] RunPod instance başlat (H100 80GB)
* [ ] `env_info.sh` çalıştır (kaydet)
* [ ] `prepare-clean`
* [ ] `install_<engine>.sh` çalıştır
* [ ] küçük doğrulama testi geçti
* [ ] `run_benchmark.py --test throughput --runs 3` çalıştır
* [ ] `run_benchmark.py --test json_struct` çalıştır (1000 iter)
* [ ] `run_benchmark.py --test low_latency` çalıştır (500+ iter)
* [ ] `{engine}.json` kaydedildi
* [ ] `uninstall_<engine>.sh` çalıştır
* [ ] instance restart (gerekirse)

---

## 13. Otomasyon İpuçları

* `install_*.sh` içinde `set -euo pipefail` kullan
* `run_benchmark.py` için `--dry-run` flag'i ve `--limit` parametreleri ekle (debug kolaylığı)
* Her benchmark run sonunda `sync` komutu ile disk flush

---

## 14. İlk Adımlar (Hemen Çalıştırılacak Komutlar)

1. Repo'yu clone et

```bash
git clone <repo>
cd benchmark_project
```

2. Edit: `scripts/install_vllm.sh` doldurulduktan sonra çalıştır

```bash
bash scripts/install_vllm.sh | tee logs/install_logs/vllm_install_$(date -u +%F_%H.%M).log
```

3. Sonra benchmark testini çalıştır

```bash
bash scripts/run_benchmark.sh --engine vllm --test throughput
```

---

## 15. Notlar (RunPod Özel)

* TensorRT için Docker container kullanılması tavsiye edilir; Docker imajı belleği, CUDA ve TRT sürümlerini izole eder.
* RunPod image'ın CUDA/TensorRT versiyonunu önceden kontrol et: `nvidia-smi` ve `/usr/local/tensorrt/version.txt`.

---