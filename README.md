# Sumikko Food Analysis GUI

## 機能

- **食器設定エディタ**: `plate_setting.json`をGUIで編集・管理
- **パイプライン実行**: 分析パイプラインの各ステップをWeb UIから実行
- **結果表示**: 分析結果の確認と可視化

## セットアップ

1. 依存パッケージのインストール:
```bash
pip install -r requirements.txt
```

**重要**: このプロジェクトでは以下の機械学習・画像処理ライブラリが必要です:
- opencv-python
- ultralytics (YOLO)
- ai-edge-litert (TensorFlow Lite)
- scikit-learn
- matplotlib
- seaborn

これらのライブラリが正しくインストールされていることを確認してください。

2. データディレクトリの確認:
   - `sumikko_analysis/` ディレクトリがプロジェクトの親ディレクトリにあることを確認してください
   - 必要なファイル:
     - `config.json`
     - `models/` (YOLOモデルとTFLiteモデル)
     - `captured_images/` (トレイ画像データ)

## 実行方法

```bash
python main.py
```

または:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

ブラウザで `http://localhost:8000` にアクセスしてください。

## パイプラインのステップ

1. **基準データ作成**: 空の食器画像から基準データを作成
2. **食器検出と分類**: YOLOで食器を検出し、特徴量マッチングで分類
   - ドーナツ状範囲（外周80%、内周30%）の色特徴を使用
   - 食品の影響を抑制
3. **食事量推定**: 深度情報と食器マスクを使用して食事量を推定
4. **結果可視化**: 混同行列などの結果を可視化

## ファイル名形式

### 旧形式（ファイル名埋め込み）
- アンダーバーあり: `A_0002060606_20260310_163444.png`
- アンダーバーなし: `A_0002060606_20260310163444.png`
- 比率を2桁または3桁でエンコード（0.2刻みまたは0.001刻み）

### 新形式（JSON参照）
- アンダーバーあり: `A_P_20260907_123111.png`
- アンダーバーなし: `A_P_20260907123111.png`
- 正解データは `sumikko_analysis/ground_truth.json` で管理

### 正解データJSON
```json
{
  "A_P_20260907_123111.png": {
    "shushoku": 0.0,
    "shusai": 0.24,
    "fukusai1": 0.40,
    "fukusai2": 0.60,
    "fukusai3": 0.80
  }
}
```

## モード切り替え

パイプライン実行時にファイル名モードを選択できます：
- **自動判定**: ファイル名形式を自動判定
- **旧形式**: ファイル名から比率を抽出
- **新形式**: JSONから正解データを参照

## APIエンドポイント

- `GET /` - ホームページ
- `GET /settings` - 食器設定ページ
- `GET /pipeline` - パイプライン実行ページ
- `GET /results` - 結果表示ページ
- `GET /api/status` - システムステータス確認
- `GET /api/plate-settings` - 食器設定取得
- `POST /api/plate-settings/{tray_id}` - 食器設定更新
- `POST /api/step1/create-baseline` - 基準データ作成
- `POST /api/step2/detect-plates` - 食器検出
- `POST /api/step2/classify-plates` - 食器分類
- `POST /api/step3/estimate-amount` - 食事量推定
- `POST /api/step4/generate-results` - 結果可視化

## 注意事項

- このアプリケーションは `sumikko_analysis` ディレクトリにあるデータを使用します
- モデルファイルと画像データが正しく配置されていることを確認してください
- 大きなデータセットを処理する場合、時間がかかることがあります
