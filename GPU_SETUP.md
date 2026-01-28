# GPU対応PyTorchのインストール手順

## 現在の状態
- PyTorch 2.10.0+cpu（CPU版）がインストール済み
- GPUサポートなし

## GPU対応にする方法

### ステップ1: NVIDIA GPUとドライバーの確認

まず、お使いのPCにNVIDIA GPUが搭載されているか確認してください：

```powershell
# デバイスマネージャーで確認
devmgmt.msc

# またはPowerShellで確認
Get-WmiObject Win32_VideoController | Select-Object Name, DriverVersion
```

### ステップ2: CUDA対応PyTorchのインストール

NVIDIA GPUがある場合、CUDA対応版をインストールします：

```powershell
# 現在のPyTorchをアンインストール
pip uninstall torch torchvision torchaudio

# CUDA 12.1対応版をインストール（推奨）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# または CUDA 11.8対応版（古いGPU向け）
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ステップ3: インストール確認

```powershell
python check_gpu.py
```

成功すると以下のように表示されます：
```
CUDA available: True
CUDA version: 12.1
Number of GPUs: 1
  GPU 0: NVIDIA GeForce RTX 3060
```

### ステップ4: 学習を実行

GPU対応になったら、そのまま学習を実行するだけです：

```powershell
# コードは自動的にGPUを検出して使用します
python train.py
```

## 現在のままCPUで学習する場合

CPUでも学習は可能ですが、GPUより遅くなります：

```powershell
# そのまま実行可能（現在の環境）
python train.py
```

**学習時間の目安：**
- CPU: 1000エピソード ≈ 2-4時間
- GPU (RTX 3060クラス): 1000エピソード ≈ 30-60分

## マルチGPU対応（複数GPUがある場合）

複数のGPUで並列学習を行う場合は、train.pyを修正する必要があります。
詳細はお知らせください。

## よくある問題

### Q: CUDA out of memory エラーが出る
A: batch_sizeを小さくしてください（dqn_agent.py内）：
```python
self.batch_size = 32  # 64から32に変更
```

### Q: GPUが検出されない
A: 
1. NVIDIAドライバーが最新か確認
2. CUDA Toolkitのバージョンとの互換性を確認
3. PyTorchを再インストール

### Q: 複数のPyTorchバージョンが競合する
A: 仮想環境を作り直すのが確実：
```powershell
# 新しい仮想環境を作成
python -m venv .venv_gpu
.venv_gpu\Scripts\activate
pip install -r requirements.txt
# CUDA版PyTorchをインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
