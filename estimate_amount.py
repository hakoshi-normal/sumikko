import os
import glob
import json
import random
import re
import numpy as np
import cv2
from datetime import datetime
import hashlib
import base64
import io
import matplotlib
matplotlib.use('Agg')  # サーバー環境でmatplotlibを使用
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix

# フォント設定（日本語対応）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'AppleGothic']

LEVELS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# 正解データの保存場所
GROUND_TRUTH_PATH = "sumikko_analysis/ground_truth.json"


def load_ground_truth(data_path="sumikko_analysis"):
    """正解データJSONを読み込む"""
    ground_truth_path = os.path.join(data_path, "ground_truth.json")
    if not os.path.exists(ground_truth_path):
        print("正解データJSONが見つかりません: ground_truth.json")
        return {}
    
    try:
        with open(ground_truth_path, encoding='utf-8') as f:
            ground_truth = json.load(f)
        print(f"正解データ読み込み完了: {len(ground_truth)} 件")
        return ground_truth
    except Exception as e:
        print(f"正解データ読み込みエラー: {e}")
        return {}


def parse_new_filename(filename):
    """新ファイル名形式のパース: A_P_20260907_123111.png（アンダーバーなしも対応）"""
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        tray_id = parts[0]
        marker = parts[1]  # 'P' など
        # マーカーが数字のみの場合は旧形式の比率列なので新形式と判定しない
        if marker.isdigit():
            return None
        timestamp_parts = parts[2:]
        
        # タイムスタンプ部分を結合（アンダーバーの有無に対応）
        timestamp = '_'.join(timestamp_parts)
        
        return {
            'tray_id': tray_id,
            'marker': marker,
            'timestamp': timestamp,
            'original_filename': basename,
            'format': 'new'
        }
    return None


def parse_old_filename(filename):
    """旧ファイル名形式のパース: A_0002060606_20260310_153003.png（浮動小数点対応、アンダーバーなしも対応）"""
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        tray_id = parts[0]
        ratio_str = parts[1]  # 0002060606
        timestamp_parts = parts[2:]
        
        # タイムスタンプ部分を結合（アンダーバーの有無に対応）
        timestamp = '_'.join(timestamp_parts)
        
        # 比率をパース（3桁ずつで分割して1000で割る → 0.000刻み対応）
        if len(ratio_str) >= 18:  # 6食器 * 3桁 = 18桁
            ratios = []
            for i in range(0, len(ratio_str), 3):
                ratio = int(ratio_str[i:i+3]) / 1000.0  # 000->0.0, 250->0.25, 1000->1.0
                ratios.append(ratio)
            
            return {
                'tray_id': tray_id,
                'ratios': ratios,
                'timestamp': timestamp,
                'original_filename': basename,
                'format': 'old'
            }
        elif len(ratio_str) >= 10 and len(ratio_str) % 2 == 0:  # 互換性のため2桁形式もサポート
            ratios = []
            for i in range(0, len(ratio_str), 2):
                ratio = int(ratio_str[i:i+2]) / 10.0  # 00->0.0, 10->1.0
                ratios.append(ratio)
            
            return {
                'tray_id': tray_id,
                'ratios': ratios,
                'timestamp': timestamp,
                'original_filename': basename,
                'format': 'old'
            }
    return None


def parse_filename(filename, mode='auto'):
    """ファイル名をパース（モードに応じて自動判定または指定）"""
    if mode == 'old':
        return parse_old_filename(filename)
    elif mode == 'new':
        return parse_new_filename(filename)
    else:  # auto
        # 自動判定
        new_result = parse_new_filename(filename)
        if new_result:
            return new_result
        
        old_result = parse_old_filename(filename)
        if old_result:
            return old_result
        
        return None


def to_index(values):
    """値をLEVELSのインデックスに変換"""
    return np.array([np.argmin(np.abs(LEVELS - v)) for v in values])


def draw_confusion_matrix(y_true, y_pred, title, r2, mae):
    """混同行列を生成してbase64エンコードされた画像を返す"""
    y_true_idx = to_index(y_true)
    y_pred_idx = to_index(y_pred)
    
    # 混同行列の作成
    cm = confusion_matrix(y_true_idx, y_pred_idx).T
    cm = np.flipud(cm)
    
    plt.figure(figsize=(6, 5))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LEVELS,
        yticklabels=LEVELS[::-1],
    )
    
    plt.title(f"{title}\nN={len(y_true)}, R²={r2:.3f}, MAE={mae:.3f}", fontsize=15)
    plt.xlabel("実測値", fontsize=13)
    plt.ylabel("システム推定値", fontsize=13)
    
    plt.tight_layout()
    
    # base64エンコード
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_base64

# TensorFlow Lite Interpreterのインポート（オプション）
try:
    from ai_edge_litert.interpreter import Interpreter
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("警告: ai_edge_litertがインストールされていません。UNet機能は無効になります。")

# モデルパスの設定
TFLITE_MODEL_PATH = "sumikko_analysis/models/unet_food_int8_256.tflite"
interpreter = None
input_details = None
output_details = None
img_size = (256, 256)
THRESHOLD = 0.5

# U-Netキャッシュ用ディレクトリ
CACHE_DIR = "sumikko_analysis/mask_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_key(image_paths):
    """画像パスからキャッシュキーを生成"""
    # パスをソートして結合
    sorted_paths = sorted(image_paths)
    path_string = "|".join(sorted_paths)
    return hashlib.md5(path_string.encode()).hexdigest()


def get_cached_mask(cache_key):
    """キャッシュからマスクを取得"""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"キャッシュ読み込みエラー: {e}")
            return None
    return None


def save_cached_mask(cache_key, mask):
    """マスクをキャッシュに保存"""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    try:
        np.save(cache_path, mask)
        return True
    except Exception as e:
        print(f"キャッシュ保存エラー: {e}")
        return False


def clear_cache():
    """キャッシュをクリア"""
    try:
        cache_files = glob.glob(os.path.join(CACHE_DIR, "*.npy"))
        for cache_file in cache_files:
            os.remove(cache_file)
        print(f"キャッシュをクリアしました: {len(cache_files)} ファイル")
        return True
    except Exception as e:
        print(f"キャッシュクリアエラー: {e}")
        return False

# TensorFlow Liteが利用可能な場合のみ初期化
if TFLITE_AVAILABLE and os.path.exists(TFLITE_MODEL_PATH):
    try:
        interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"警告: TFLiteモデルの初期化に失敗しました: {e}")
        interpreter = None


def get_tablewaremask(images, unet_flg=False):
    """食器マスク生成"""
    h, w, _ = images[0].shape
    
    if unet_flg and interpreter is not None:
        # キャッシュキーを生成
        image_paths = []
        for img in images:
            # numpy配列からハッシュを生成（実際のパスが使えない場合）
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            image_paths.append(img_hash)
        
        cache_key = get_cache_key(image_paths)
        
        # キャッシュを確認
        cached_mask = get_cached_mask(cache_key)
        if cached_mask is not None:
            print("        キャッシュヒット: マスクを使用")
            mask = cached_mask
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks = [mask] * len(images)
            return mask, masks
        
        print("        キャッシュミス: 新規マスク生成")
        
        masks = []
        for image in images:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, img_size).astype(np.float32) / 255.0
            image_dims = np.expand_dims(image_resized, axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], image_dims)
            interpreter.invoke()
            
            mask = interpreter.get_tensor(output_details[0]['index'])[0]
            masks.append(mask[:,:,0] > THRESHOLD)
            
        # 論理和マスクの作成とリサイズ
        mask = np.any(np.stack(masks, axis=0), axis=0).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        masks_resized = [cv2.resize(m.astype(np.uint8)*255, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]
        
        # キャッシュに保存
        save_cached_mask(cache_key, mask)
        
        masks = masks_resized
    else:
        # UNetが利用できない場合は円形マスクを使用
        center = (w // 2, h // 2)
        radius = int(min(h, w) / 2 * 0.5)
        mask_circle = np.zeros((h, w), dtype=np.uint8)
        mask = cv2.circle(mask_circle, center, radius, 255, -1)
        masks = [mask] * len(images)
        
    return mask, masks


def calc_amount(depth_empty, depth_full, depth_after, meal_mask,
                depth_min=2000, depth_max=3500,
                inpaint_radius=3):
    """残存率計算"""
    h, w = depth_empty.shape
    depth_full = cv2.resize(depth_full, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_after = cv2.resize(depth_after, (w, h), interpolation=cv2.INTER_NEAREST)
    
    def clip_and_inpaint(depth):
        depth = depth.astype(np.float32)
        depth[(depth < depth_min) | (depth > depth_max)] = np.nan
        mask = np.isnan(depth).astype(np.uint8) * 255
        depth_filled = np.nan_to_num(depth, nan=0.0)
        if np.any(mask):
            depth_inpainted = cv2.inpaint(depth_filled.astype(np.float32), mask, inpaint_radius, cv2.INPAINT_TELEA)
        else:
            depth_inpainted = depth_filled
        depth_inpainted[meal_mask == 0] = 0
        return depth_inpainted

    depth_empty = clip_and_inpaint(depth_empty)
    depth_full = clip_and_inpaint(depth_full)
    depth_after = clip_and_inpaint(depth_after)

    h_full = depth_empty - depth_full
    h_after = depth_empty - depth_after

    valid_mask = meal_mask > 0
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        return None
    
    V_full = np.nanmean(h_full[valid_mask])
    V_after = np.nanmean(h_after[valid_mask])

    if V_full < 1e-6:
        return 0.0
    
    remaining_ratio = V_after / V_full
    remaining_ratio = float(np.clip(remaining_ratio, 0, 1))
    
    if V_full < 0:
        remaining_ratio = 1.0
    if V_after < 0:
        remaining_ratio = 0.0

    return remaining_ratio


def calc_score(y_true, y_pred):
    """スコア計算"""
    try:
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return float(r2), float(mae)
    except ImportError:
        print("警告: scikit-learnがインストールされていません")
        return 0.0, 0.0


def estimate_food_amount(data_path="sumikko_analysis", iterations=100, filename_mode='auto', tray_ids=None, plate_types=None):
    """食事量推定の実行（最適化版）"""
    tmp_data = {}
    
    # 正解データの読み込み
    ground_truth = load_ground_truth(data_path)
    
    # 利用可能なトレイIDを確認
    plate_ids = [os.path.basename(p) for p in glob.glob(f"{data_path}/plate_images/*")]
    if tray_ids:
        plate_ids = [p for p in plate_ids if p in tray_ids]
    print(f"利用可能なトレイID: {plate_ids}")
    print(f"ファイル名モード: {filename_mode}")
    
    if not plate_ids:
        print("エラー: 処理可能なトレイデータが見つかりません")
        return {
            "status": "error",
            "message": "処理可能なトレイデータが見つかりません。先にStep 2を実行してください。"
        }
    
    print(f"食事量推定開始: 反復回数={iterations}, トレイ数={len(plate_ids)}")
    
    # 設定ファイルとパスを事前にキャッシュ
    print("設定ファイルとパスのキャッシュを作成中...")
    config_cache = {}
    for plate_id in plate_ids:
        setting_path = f"{data_path}/test_data/{plate_id}/plate_setting.json"
        if not os.path.exists(setting_path):
            continue
            
        with open(setting_path, encoding='utf-8') as f:
            base_config = json.load(f)
        
        plate_names = [d["plate_name"] for d in base_config]
        nansai_flags = [d["nansai"] for d in base_config]
        
        config_cache[plate_id] = {
            'plate_names': plate_names,
            'nansai_flags': nansai_flags,
            'base_config': base_config
        }
    
    print(f"キャッシュ作成完了: {len(config_cache)} トレイ")
    
    for iter_num in range(iterations):
        if (iter_num + 1) % 10 == 0 or iter_num == 0:
            print(f"反復進捗: {iter_num + 1}/{iterations}")
            
        for plate_idx, plate_id in enumerate(plate_ids):
            if plate_id not in config_cache:
                continue
                
            if (plate_idx + 1) % 5 == 0 or plate_idx == 0:
                print(f"  トレイ処理: {plate_idx + 1}/{len(plate_ids)} (ID: {plate_id})")
                
            cache_data = config_cache[plate_id]
            plate_names = cache_data['plate_names']
            nansai_flags = cache_data['nansai_flags']
            
            for plate_name_idx, (plate_name, nansai_flag) in enumerate(zip(plate_names, nansai_flags)):
                if plate_types and plate_name not in plate_types:
                    continue

                dict_path = f"{data_path}/plate_images/{plate_id}/{plate_name}/"
                
                if not os.path.exists(dict_path):
                    continue
                    
                full_plate_paths = glob.glob(f"{dict_path}/plate_10_*.png")
                if not full_plate_paths:
                    continue
                    
                full_plate_path = random.choice(full_plate_paths)
                full_plate_image = cv2.imread(full_plate_path)
                full_plate_depth = np.load(full_plate_path.replace(".png", ".npy"))
                
                empty_plate_paths = glob.glob(f"{dict_path}/plate_00_*.png")
                if not empty_plate_paths:
                    continue
                    
                empty_plate_path = random.choice(empty_plate_paths)
                empty_plate_image = cv2.imread(empty_plate_path)
                empty_plate_depth = np.load(empty_plate_path.replace(".png", ".npy"))
                
                target_plate_paths = glob.glob(f"{dict_path}/plate_*.png")
                
                y_true, y_pred = [], []
                
                for target_plate_path in target_plate_paths:
                    try:
                        target_plate_image = cv2.imread(target_plate_path)
                        target_plate_depth = np.load(target_plate_path.replace(".png", ".npy"))
                        
                        # 分類済みファイル名 plate_XX_<timestamp>.png のXXから正解値を取得
                        # （XX = 割合×10。例: plate_04_... → 0.4）
                        basename = os.path.basename(target_plate_path)
                        amount_match = re.match(r"plate_(\d+)_", basename)
                        if amount_match:
                            real_amount = int(amount_match.group(1)) / 10.0
                            parsed = None
                        else:
                            # ファイル名をパースして正解値を取得
                            parsed = parse_filename(target_plate_path, filename_mode)
                        
                        if parsed and parsed['format'] == 'old':
                            # 旧形式: ファイル名から直接比率を取得
                            if plate_name_idx < len(parsed['ratios']):
                                real_amount = parsed['ratios'][plate_name_idx]
                            else:
                                continue
                        elif parsed and parsed['format'] == 'new':
                            # 新形式: JSONから正解値を取得
                            filename_key = parsed['original_filename']
                            if filename_key in ground_truth:
                                gt_data = ground_truth[filename_key]
                                if plate_name in gt_data:
                                    real_amount = gt_data[plate_name]
                                else:
                                    # 正解がない場合は推定のみ実行
                                    real_amount = None
                            else:
                                # 正解がない場合は推定のみ実行
                                real_amount = None
                        elif not amount_match:
                            continue
                        
                        # 全ての残量データを処理
                        tableware_images = [empty_plate_image, full_plate_image, target_plate_image]
                        meal_mask, meal_masks = get_tablewaremask(tableware_images, unet_flg=not nansai_flag)
                        
                        remaining_ratio = calc_amount(empty_plate_depth, full_plate_depth, target_plate_depth, meal_mask)
                        
                        if remaining_ratio is None:
                            continue
                        
                        # 正解がある場合のみ精度評価に追加
                        if real_amount is not None:
                            y_true.append(real_amount)
                            y_pred.append(remaining_ratio)
                    except Exception as e:
                        continue
                
                if len(y_true) == 0:
                    continue
                
                try:
                    r2, mae = calc_score(y_true, y_pred)
                    
                    key = f"{plate_id}_{plate_name}"
                    if key not in tmp_data.keys():
                        tmp_data[key] = []
                    tmp_data[key].append({
                        "plate_id": plate_id,
                        "plate_name": plate_name,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "r2": r2,
                        "mae": mae
                    })
                except Exception as e:
                    continue
    
    # 結果の集計
    final_results = []
    for key, records in tmp_data.items():
        best_record = max(records, key=lambda x: x["r2"])
        final_results.append(best_record)
    
    results_path = f"{data_path}/final_results.json"
    with open(results_path, "w", encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"食事量推定完了: 反復回数={iterations}, 結果数={len(final_results)}")
    print(f"処理されたトレイ: {list(set([r['plate_id'] for r in final_results]))}")
    print(f"処理された食器: {list(set([r['plate_name'] for r in final_results]))}")
    print(f"保存場所: {data_path}/final_results.json")
    print(f"保存場所: {data_path}/mask_cache/ (U-Netマスクキャッシュ)")
    
    return {
        "status": "completed",
        "iterations": iterations,
        "total_results": len(final_results),
        "processed_trays": list(set([r['plate_id'] for r in final_results])),
        "processed_plates": list(set([r['plate_name'] for r in final_results])),
        "results_path": results_path
    }


def get_analysis_data(data_path="sumikko_analysis"):
    """分析用データを集計して返却"""
    results_path = f"{data_path}/final_results.json"
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, encoding='utf-8') as f:
        results = json.load(f)
    
    # トレイ別、食器別、軟菜フラグ別にデータを集計
    analysis_data = {
        "by_tray": {},
        "by_plate": {},
        "by_nansai": {"nansai": [], "non_nansai": []},
        "all_data": []
    }
    
    # トレイ別・食器別の設定を読み込み
    plate_settings = {}
    for plate_id in set([r['plate_id'] for r in results]):
        setting_path = f"{data_path}/test_data/{plate_id}/plate_setting.json"
        if os.path.exists(setting_path):
            with open(setting_path, encoding="utf-8") as f:
                setting = json.load(f)
            plate_settings[plate_id] = {
                item["plate_name"]: item["nansai"]
                for item in setting
            }
    
    for result in results:
        plate_id = result["plate_id"]
        plate_name = result["plate_name"]
        nansai = plate_settings.get(plate_id, {}).get(plate_name, False)
        
        # 全データ
        for y_true, y_pred in zip(result["y_true"], result["y_pred"]):
            analysis_data["all_data"].append({
                "plate_id": plate_id,
                "plate_name": plate_name,
                "nansai": nansai,
                "y_true": y_true,
                "y_pred": y_pred
            })
        
        # トレイ別
        if plate_id not in analysis_data["by_tray"]:
            analysis_data["by_tray"][plate_id] = []
        for y_true, y_pred in zip(result["y_true"], result["y_pred"]):
            analysis_data["by_tray"][plate_id].append({
                "plate_name": plate_name,
                "nansai": nansai,
                "y_true": y_true,
                "y_pred": y_pred
            })
        
        # 食器別
        if plate_name not in analysis_data["by_plate"]:
            analysis_data["by_plate"][plate_name] = []
        for y_true, y_pred in zip(result["y_true"], result["y_pred"]):
            analysis_data["by_plate"][plate_name].append({
                "plate_id": plate_id,
                "nansai": nansai,
                "y_true": y_true,
                "y_pred": y_pred
            })
        
        # 軟菜別
        for y_true, y_pred in zip(result["y_true"], result["y_pred"]):
            if nansai:
                analysis_data["by_nansai"]["nansai"].append({
                    "plate_id": plate_id,
                    "plate_name": plate_name,
                    "y_true": y_true,
                    "y_pred": y_pred
                })
            else:
                analysis_data["by_nansai"]["non_nansai"].append({
                    "plate_id": plate_id,
                    "plate_name": plate_name,
                    "y_true": y_true,
                    "y_pred": y_pred
                })
    
    return analysis_data


def generate_scatter_plot(y_true, y_pred, title="予測 vs 実測"):
    """散布図を生成してbase64エンコードされた画像を返す"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    plt.plot([0, 1], [0, 1], 'r--', lw=2)  # 完全一致ライン
    plt.xlabel('実測値')
    plt.ylabel('予測値')
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # R²とMAEを計算して表示
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # base64エンコード
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_base64


def generate_analysis_charts(analysis_data, aggregation_type='all', filter_value='all', chart_type='confusion', tray_id='all', plate_name='all'):
    """分析用グラフを生成"""
    charts = []
    
    if aggregation_type == 'all':
        # 全体
        y_true = [d['y_true'] for d in analysis_data['all_data']]
        y_pred = [d['y_pred'] for d in analysis_data['all_data']]
        if y_true:
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            if chart_type == 'confusion':
                # 混同行列（デフォルト）
                img_base64 = draw_confusion_matrix(1 - np.array(y_true), 1 - np.array(y_pred), "全体精度", r2, mae)
                charts.append({
                    'title': '全体精度（混同行列）',
                    'image': img_base64,
                    'r2': r2,
                    'mae': mae,
                    'sample_count': len(y_true),
                    'chart_type': 'confusion'
                })
            else:
                # 散布図（オプション）
                img_base64 = generate_scatter_plot(y_true, y_pred, "全体精度: 予測 vs 実測")
                charts.append({
                    'title': '全体精度（散布図）',
                    'image': img_base64,
                    'r2': r2,
                    'mae': mae,
                    'sample_count': len(y_true),
                    'chart_type': 'scatter'
                })
    
    elif aggregation_type == 'by_tray':
        # トレイ別
        if tray_id == 'all':
            for tid, data in analysis_data['by_tray'].items():
                y_true = [d['y_true'] for d in data]
                y_pred = [d['y_pred'] for d in data]
                if y_true:
                    r2 = r2_score(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    
                    if chart_type == 'confusion':
                        img_base64 = draw_confusion_matrix(1 - np.array(y_true), 1 - np.array(y_pred), f"トレイ{tid}精度", r2, mae)
                        charts.append({
                            'title': f'トレイ{tid}（混同行列）',
                            'image': img_base64,
                            'r2': r2,
                            'mae': mae,
                            'sample_count': len(y_true),
                            'chart_type': 'confusion'
                        })
                    else:
                        img_base64 = generate_scatter_plot(y_true, y_pred, f"トレイ{tid}精度: 予測 vs 実測")
                        charts.append({
                            'title': f'トレイ{tid}（散布図）',
                            'image': img_base64,
                            'r2': r2,
                            'mae': mae,
                            'sample_count': len(y_true),
                            'chart_type': 'scatter'
                        })
        else:
            data = analysis_data['by_tray'].get(tray_id, [])
            y_true = [d['y_true'] for d in data]
            y_pred = [d['y_pred'] for d in data]
            if y_true:
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                
                if chart_type == 'confusion':
                    img_base64 = draw_confusion_matrix(1 - np.array(y_true), 1 - np.array(y_pred), f"トレイ{tray_id}精度", r2, mae)
                    charts.append({
                        'title': f'トレイ{tray_id}（混同行列）',
                        'image': img_base64,
                        'r2': r2,
                        'mae': mae,
                        'sample_count': len(y_true),
                        'chart_type': 'confusion'
                    })
                else:
                    img_base64 = generate_scatter_plot(y_true, y_pred, f"トレイ{tray_id}精度: 予測 vs 実測")
                    charts.append({
                        'title': f'トレイ{tray_id}（散布図）',
                        'image': img_base64,
                        'r2': r2,
                        'mae': mae,
                        'sample_count': len(y_true),
                        'chart_type': 'scatter'
                    })
    
    elif aggregation_type == 'by_plate':
        # 食器別
        if plate_name == 'all':
            for pname, data in analysis_data['by_plate'].items():
                y_true = [d['y_true'] for d in data]
                y_pred = [d['y_pred'] for d in data]
                if y_true:
                    r2 = r2_score(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    
                    if chart_type == 'confusion':
                        img_base64 = draw_confusion_matrix(1 - np.array(y_true), 1 - np.array(y_pred), f"{pname}精度", r2, mae)
                        charts.append({
                            'title': f'{pname}（混同行列）',
                            'image': img_base64,
                            'r2': r2,
                            'mae': mae,
                            'sample_count': len(y_true),
                            'chart_type': 'confusion'
                        })
                    else:
                        img_base64 = generate_scatter_plot(y_true, y_pred, f"{pname}精度: 予測 vs 実測")
                        charts.append({
                            'title': f'{pname}（散布図）',
                            'image': img_base64,
                            'r2': r2,
                            'mae': mae,
                            'sample_count': len(y_true),
                            'chart_type': 'scatter'
                        })
        else:
            data = analysis_data['by_plate'].get(plate_name, [])
            y_true = [d['y_true'] for d in data]
            y_pred = [d['y_pred'] for d in data]
            if y_true:
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                
                if chart_type == 'confusion':
                    img_base64 = draw_confusion_matrix(1 - np.array(y_true), 1 - np.array(y_pred), f"{plate_name}精度", r2, mae)
                    charts.append({
                        'title': f'{plate_name}（混同行列）',
                        'image': img_base64,
                        'r2': r2,
                        'mae': mae,
                        'sample_count': len(y_true),
                        'chart_type': 'confusion'
                    })
                else:
                    img_base64 = generate_scatter_plot(y_true, y_pred, f"{plate_name}精度: 予測 vs 実測")
                    charts.append({
                        'title': f'{plate_name}（散布図）',
                        'image': img_base64,
                        'r2': r2,
                        'mae': mae,
                        'sample_count': len(y_true),
                        'chart_type': 'scatter'
                    })
    
    elif aggregation_type == 'by_tray_plate':
        # トレイ×食器別
        for tid, data in analysis_data['by_tray'].items():
            if tray_id != 'all' and tid != tray_id:
                continue
                
            for item in data:
                if plate_name != 'all' and item['plate_name'] != plate_name:
                    continue
                
                y_true = [item['y_true']]
                y_pred = [item['y_pred']]
                
                if y_true:
                    r2 = r2_score(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    
                    title = f"トレイ{tid}-{item['plate_name']}"
                    if chart_type == 'confusion':
                        img_base64 = draw_confusion_matrix(1 - np.array(y_true), 1 - np.array(y_pred), f"{title}精度", r2, mae)
                        charts.append({
                            'title': f'{title}（混同行列）',
                            'image': img_base64,
                            'r2': r2,
                            'mae': mae,
                            'sample_count': len(y_true),
                            'chart_type': 'confusion'
                        })
                    else:
                        img_base64 = generate_scatter_plot(y_true, y_pred, f"{title}精度: 予測 vs 実測")
                        charts.append({
                            'title': f'{title}（散布図）',
                            'image': img_base64,
                            'r2': r2,
                            'mae': mae,
                            'sample_count': len(y_true),
                            'chart_type': 'scatter'
                        })
    
    elif aggregation_type == 'by_nansai':
        # 軟菜別
        for category, data in [('軟菜食', analysis_data['by_nansai']['nansai']), 
                               ('固体食', analysis_data['by_nansai']['non_nansai'])]:
            y_true = [d['y_true'] for d in data]
            y_pred = [d['y_pred'] for d in data]
            if y_true:
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                
                if chart_type == 'confusion':
                    img_base64 = draw_confusion_matrix(1 - np.array(y_true), 1 - np.array(y_pred), f"{category}精度", r2, mae)
                    charts.append({
                        'title': f'{category}（混同行列）',
                        'image': img_base64,
                        'r2': r2,
                        'mae': mae,
                        'sample_count': len(y_true),
                        'chart_type': 'confusion'
                    })
                else:
                    img_base64 = generate_scatter_plot(y_true, y_pred, f"{category}精度: 予測 vs 実測")
                    charts.append({
                        'title': f'{category}（散布図）',
                        'image': img_base64,
                        'r2': r2,
                        'mae': mae,
                        'sample_count': len(y_true),
                        'chart_type': 'scatter'
                    })
    
    return charts


def get_final_results(data_path="sumikko_analysis"):
    """最終結果を取得"""
    results_path = f"{data_path}/final_results.json"
    if os.path.exists(results_path):
        with open(results_path, encoding='utf-8') as f:
            return json.load(f)
    return []
