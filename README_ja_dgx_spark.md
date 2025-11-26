# SAM 3 - DGX Spark セットアップガイド

このガイドでは、NVIDIA DGX Spark（aarch64 / CUDA 12.x）環境でSAM 3を動かす方法を説明します。

## 前提条件

- NVIDIA DGX Spark（aarch64アーキテクチャ）
- CUDA 12.x
- Python 3.10以上
- [uv](https://docs.astral.sh/uv/)（Python パッケージマネージャー）

## セットアップ

### 1. uvのインストール（未インストールの場合）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. リポジトリのクローン

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
```

### 3. 依存関係のインストール

```bash
uv sync
```

これだけで仮想環境の作成とすべての依存関係のインストールが完了します。

### 4. Hugging Face認証

SAM 3のチェックポイントをダウンロードするには、Hugging Faceへの認証が必要です。

1. [SAM 3 Hugging Face リポジトリ](https://huggingface.co/facebook/sam3)でアクセスをリクエスト
2. アクセスが承認されたら、ログイン：

```bash
source .venv/bin/activate
hf auth login
```

## 使い方

### 仮想環境の有効化

```bash
source .venv/bin/activate
```

### 基本的な使用例

```python
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# モデルの読み込み
model = build_sam3_image_model()
processor = Sam3Processor(model)

# 画像の読み込み
image = Image.open("your_image.jpg")
inference_state = processor.set_image(image)

# テキストプロンプトでセグメンテーション
output = processor.set_text_prompt(state=inference_state, prompt="dog")

# 結果の取得
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

### デモの実行

```bash
python demo.py
```

結果は `demo_output.jpg` に保存されます。

### Jupyter Notebookの実行

```bash
uv sync --extra notebooks
jupyter notebook examples/sam3_image_predictor_example.ipynb
```

## DGX Spark固有の注意事項

### decordについて

DGX Spark（aarch64）では、`decord`ライブラリのホイールが提供されていません。
このリポジトリでは、代わりに`PyAV`を使用する互換レイヤー（`sam3/utils/decord_compat.py`）を実装しています。

### GPUの警告について

以下の警告が表示されることがありますが、動作には問題ありません：

```
UserWarning: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

これはNVIDIA GB10がPyTorchの公式サポート範囲を少し超えているためですが、実際の動作には影響しません。

## トラブルシューティング

### `ModuleNotFoundError`が発生する場合

依存関係が正しくインストールされていない可能性があります：

```bash
uv sync
```

### Hugging Faceからのダウンロードに失敗する場合

認証が正しく設定されているか確認してください：

```bash
source .venv/bin/activate
hf auth login
```

### CUDAエラーが発生する場合

CUDAドライバとPyTorchのバージョンが互換性があるか確認してください：

```bash
source .venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 変更点（オリジナルからの差分）

DGX Spark対応のため、以下の変更を行っています：

1. **pyproject.toml**
   - `requires-python`を`>=3.10`に変更
   - PyTorch CUDA 12.9用のインデックス設定を追加
   - `decord`を`av`（PyAV）と`opencv-python`に置き換え
   - 不足していた依存関係を追加（`einops`, `pycocotools`, `psutil`）

2. **sam3/utils/decord_compat.py**（新規作成）
   - PyAVを使用したdecord互換レイヤー

3. **sam3/train/data/sam3_image_dataset.py**
   - decord互換レイヤーを使用するように変更

4. **sam3/model/utils/sam2_utils.py**
   - decord互換レイヤーを使用するように変更

## ライセンス

SAM License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。
