import random
import glob
import json
import os
import shutil
from pathlib import Path

# 必要なライブラリのインポート（オプション）
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: opencv-pythonがインストールされていません。画像処理機能は無効になります。")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("警告: numpyがインストールされていません。数値計算機能は無効になります。")

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: scipyがインストールされていません。マッチング機能は無効になります。")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    yolo_model = YOLO('sumikko_analysis/models/tableware_best_ncnn_model', task='detect', verbose=False)
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralyticsがインストールされていません。YOLO機能は無効になります。")
    yolo_model = None


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
        # アンダーバーがある場合：["20260310", "163444"] → "20260310_163444"
        # アンダーバーがない場合：["20260310163444"] → "20260310163444"
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


def calc_depth_var(depth):
    """深度層分散計算関数"""
    if not NUMPY_AVAILABLE or not CV2_AVAILABLE:
        return 0.0
        
    h, w = depth.shape
    center = (w // 2, h // 2)
    radius = min(h, w) // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.circle(mask, center, radius, 1, -1)
    depth_masked = depth * mask
    data = depth_masked.flatten()
    data = np.delete(data, data==0)
    BIN_COUNT = 50
    DATA_MIN = np.min(data)
    DATA_MAX = np.max(data)
    bins = np.linspace(DATA_MIN, DATA_MAX, BIN_COUNT + 1)
    hist, _ = np.histogram(data, bins=bins)
    nums = [d/data.size for d in hist.tolist()]
    return np.var(nums)


def create_baseline_data(data_path="sumikko_analysis", plate_type_mode="both"):
    """テスト用基準データ作成"""
    if not YOLO_AVAILABLE:
        raise ImportError("YOLOモデルが利用できません。ultralyticsをインストールしてください。")
    if not CV2_AVAILABLE:
        raise ImportError("OpenCVが利用できません。opencv-pythonをインストールしてください。")
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPyが利用できません。numpyをインストールしてください。")
        
    config_path = os.path.join(data_path, "config.json")
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    # captured_imagesディレクトリから実際のtray_idsを取得
    captured_images_path = f"{data_path}/captured_images"
    tray_ids = []
    if os.path.exists(captured_images_path):
        for dir_name in os.listdir(captured_images_path):
            if dir_name.startswith("captured_images_"):
                tray_id = dir_name.replace("captured_images_", "")
                if tray_id not in tray_ids:
                    tray_ids.append(tray_id)
    
    if not tray_ids:
        raise ValueError("captured_imagesディレクトリに有効なトレイデータが見つかりません")
    
    print(f"検出されたトレイID: {tray_ids}")
    
    plate_names = list(config.keys())
    if 'shokushu' in plate_names:
        plate_names.remove('shokushu')

    results = {}
    
    for tray_id in tray_ids:
        # 空の食器（0割）を基準とする
        plate_ids = "00" * len(plate_names)
        plate_images = glob.glob(f"{data_path}/captured_images/captured_images_{tray_id}/{tray_id}_{plate_ids}*.png")
        if not plate_images:
            print(f"警告: トレイ {tray_id} の基準画像が見つかりません")
            continue

        image_path = random.choice(plate_images)
        os.makedirs(f"{data_path}/test_data/{tray_id}", exist_ok=True)

        try:
            results_list = yolo_model(image_path, verbose=False)
            result = results_list[0]
            boxes = result.boxes.xyxy.cpu().numpy()
        except Exception as e:
            print(f"エラー: トレイ {tray_id} のYOLO検出に失敗: {e}")
            continue

        depth_var_list = []
        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = map(int, box)
                dish_image = cv2.imread(image_path)[y1:y2, x1:x2]
                # プレビュー用の画像を保存
                cv2.imwrite(f"{data_path}/test_data/{tray_id}/plate_{i}.png", dish_image)
                
                depth_path = image_path.replace(".png", ".npy")
                if os.path.exists(depth_path):
                    dish_depth = np.load(depth_path)[y1:y2, x1:x2]
                    depth_var = calc_depth_var(dish_depth)
                    depth_var_list.append(depth_var)
                else:
                    print(f"警告: トレイ {tray_id} の深度データが見つかりません: {depth_path}")
                    depth_var_list.append(0)  # デフォルト値
            except Exception as e:
                print(f"エラー: トレイ {tray_id} の食器 {i} の処理に失敗: {e}")
                depth_var_list.append(0)  # デフォルト値

        if plate_type_mode == "nansai":
            binary = [True] * len(depth_var_list)
        elif plate_type_mode == "kokei":
            binary = [False] * len(depth_var_list)
        elif plate_type_mode == "both":
            try:
                depth_var_list = np.array(depth_var_list)
                if len(depth_var_list) > 1:
                    sorted_idx = np.argsort(depth_var_list)
                    sorted_arr = depth_var_list[sorted_idx]
                    diff = np.diff(sorted_arr)
                    max_gap_idx = np.argmax(diff)
                    threshold = (sorted_arr[max_gap_idx] + sorted_arr[max_gap_idx + 1]) / 2
                    binary = np.array(depth_var_list) > threshold
                    binary = binary.astype(bool).tolist()
                else:
                    binary = [False]  # デフォルト値
            except Exception as e:
                print(f"エラー: トレイ {tray_id} の閾値計算に失敗: {e}")
                binary = [False] * len(depth_var_list)

        plate_setting = [{"plate_name": plate_names[i] if i < len(plate_names) else f"plate_{i}", "plate_number": i, "nansai": nansai_flg} for i, nansai_flg in enumerate(binary)]
        
        setting_path = f"{data_path}/test_data/{tray_id}/plate_setting.json"
        with open(setting_path, "w", encoding='utf-8') as f:
            json.dump(plate_setting, f, indent=4, ensure_ascii=False)
        
        results[tray_id] = {
            "status": "success",
            "plate_count": len(plate_setting),
            "setting_path": setting_path
        }
        
        print(f"トレイ {tray_id} の基準データ作成完了: {len(plate_setting)} 食器")
    
    print(f"基準データ作成完了: 処理成功数={len(results)}/{len(tray_ids)}")
    print(f"保存場所: {data_path}/test_data/")
    print(f"  - 各トレイディレクトリ: {data_path}/test_data/<tray_id>/")
    print(f"  - 設定ファイル: {data_path}/test_data/<tray_id>/plate_setting.json")
    print(f"  - プレビュー画像: {data_path}/test_data/<tray_id>/plate_*.png")
    
    return results


def detect_and_crop_plates(data_path="sumikko_analysis", use_yolo_cache=False, tray_ids=None):
    """食器の検出と切り出し（最適化版）"""
    if not YOLO_AVAILABLE:
        raise ImportError("YOLOモデルが利用できません。ultralyticsをインストールしてください。")
    if not CV2_AVAILABLE:
        raise ImportError("OpenCVが利用できません。opencv-pythonをインストールしてください。")
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPyが利用できません。numpyをインストールしてください。")
        
    captured_path = f"{data_path}/captured_images"
    plate_images = sorted(glob.glob(f"{captured_path}/*/*.png"))

    if tray_ids:
        plate_images = [p for p in plate_images if os.path.basename(p).split("_")[0] in tray_ids]
        if use_yolo_cache:
            print("警告: トレイIDを指定した場合はYOLOキャッシュを使用できません。再検出を実行します。")
            use_yolo_cache = False

    print(f"処理対象の画像数: {len(plate_images)}")
    
    if use_yolo_cache and os.path.exists(f"{data_path}/boxes_cache.npy"):
        boxes_cache = np.load(f"{data_path}/boxes_cache.npy", allow_pickle=True)
        if len(boxes_cache) != len(plate_images):
            print("警告: キャッシュの食器数が現在の画像数と異なります。キャッシュを使用せずに再実行します。")
            boxes_cache = []
    else:
        boxes_cache = []

    all_plate_N = 0
    detect_plate_N = 0
    error_count = 0
    
    # 基準データのキャッシュ
    gt_cache = {}
    for plate_image in plate_images:
        plate_id = os.path.basename(plate_image).split("_")[0]
        if plate_id not in gt_cache:
            gt_images = glob.glob(f"{data_path}/test_data/{plate_id}/*.png")
            gt_count = len(gt_images)
            gt_cache[plate_id] = gt_count if gt_count > 0 else 1
    
    print(f"基準データキャッシュ作成完了: {len(gt_cache)} トレイ")

    for image_idx, plate_image in enumerate(plate_images):
        if (image_idx + 1) % 10 == 0:
            print(f"処理進捗: {image_idx + 1}/{len(plate_images)}")
            
        plate_name = os.path.splitext(os.path.basename(plate_image))[0]
        dir_path = f"{data_path}/plate_images_tmp/{plate_name}"
        os.makedirs(dir_path, exist_ok=True)
        
        try:
            if use_yolo_cache:
                boxes = boxes_cache[image_idx]
                if np.any(np.isnan(boxes)):
                    error_count += 1
                    print(f"警告: {plate_name} のキャッシュデータが無効です")
                    continue
            else:
                # YOLO検出
                try:
                    results = yolo_model(plate_image, verbose=False)
                    boxes = results[0].boxes
                    sorted_boxes = boxes[boxes.conf.argsort(descending=True)]
                    N = 5
                    boxes = sorted_boxes[:N].xyxy.cpu().numpy()
                except Exception as e:
                    print(f"エラー: {plate_name} のYOLO検出に失敗: {e}")
                    error_count += 1
                    continue
                
                plate_id = plate_name.split("_")[0]
                gt_count = gt_cache.get(plate_id, 1)
                
                all_plate_N += gt_count
                detect_plate_N += len(boxes)
                
                if len(boxes) != gt_count:
                    error_count += 1
                    print(f"警告: {plate_name} の食器数が不一致: 検出={len(boxes)}, 基準={gt_count}")
                    # ダミーボックスを追加してキャッシュに保存
                    while len(boxes) < gt_count:
                        boxes = np.append(boxes, np.full((1, 4), np.nan), axis=0)
                    boxes_cache.append(boxes)
                    continue
                else:
                    boxes_cache.append(boxes)

            # 画像読み込み
            img = cv2.imread(plate_image)
            if img is None:
                print(f"エラー: {plate_name} の画像読み込みに失敗")
                error_count += 1
                continue
                
            H, W = img.shape[:2]
            
            # 深度データの存在確認
            depth_path = plate_image.replace(".png", ".npy")
            if not os.path.exists(depth_path):
                print(f"警告: {plate_name} の深度データが見つかりません")
                error_count += 1
                continue
            
            for i, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 座標の境界チェック
                    x1 = max(0, min(x1, W))
                    y1 = max(0, min(y1, H))
                    x2 = max(0, min(x2, W))
                    y2 = max(0, min(y2, H))
                    
                    if x2 <= x1 or y2 <= y1:
                        print(f"警告: {plate_name} の食器 {i} の座標が無効です")
                        continue
                    
                    dish_image = img[y1:y2, x1:x2]
                    dish_depth = np.load(depth_path)[y1:y2, x1:x2]

                    def rotate_by_region(img, depth, cx, cy, W, H):
                        if cx < W/2 and cy < H/2:
                            k = 0
                        elif cx >= W/2 and cy < H/2:
                            k = 1
                        elif cx >= W/2 and cy >= H/2:
                            k = 2
                        else:
                            k = 3
                        img_rot = np.rot90(img, k=k)
                        depth_rot = np.rot90(depth, k=k)
                        return img_rot, depth_rot, k

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    dish_image, dish_depth, region = rotate_by_region(dish_image, dish_depth, cx, cy, W, H)
                    
                    cv2.imwrite(f"{data_path}/plate_images_tmp/{plate_name}/plate_{i}_tmp.png", dish_image)
                    np.save(f"{data_path}/plate_images_tmp/{plate_name}/plate_{i}_tmp.npy", dish_depth)
                except Exception as e:
                    print(f"エラー: {plate_name} の食器 {i} の処理に失敗: {e}")
                    continue
                    
        except Exception as e:
            print(f"エラー: {plate_name} の処理中に例外が発生: {e}")
            error_count += 1
            continue

    if not use_yolo_cache:
        np.save(f"{data_path}/boxes_cache.npy", np.array(boxes_cache))
    
    print(f"処理完了: 総数={len(plate_images)}, エラー={error_count}, エラー率={error_count / len(plate_images) * 100 if plate_images else 0:.2f}%")
    print(f"保存場所: {data_path}/plate_images_tmp/")
    print(f"  - 切り出し画像: {data_path}/plate_images_tmp/<plate_name>/plate_*_tmp.png")
    print(f"  - 切り出し深度: {data_path}/plate_images_tmp/<plate_name>/plate_*_tmp.npy")
    print(f"  - YOLOキャッシュ: {data_path}/boxes_cache.npy")
    
    return {
        "total_plates": len(plate_images),
        "error_count": error_count,
        "error_rate": error_count / len(plate_images) * 100 if plate_images else 0,
        "total_dishes": all_plate_N,
        "detected_dishes": detect_plate_N
    }


def calc_feature(img):
    """食器の特徴量計算（ドーナツ状範囲のみ）"""
    if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
        return np.array([0, 0, 0, 0], dtype=np.float32)
        
    h, w = img.shape[:2]
    area = h * w
    
    # ドーナツ状マスクの作成
    center_x, center_y = w // 2, h // 2
    max_radius = min(w, h) // 2
    outer_radius = int(max_radius * 1)  # 外周：最大半径の100%
    inner_radius = int(max_radius * 0.5)  # 内周：最大半径の50%
    
    # ドーナツ状マスクの生成
    y_indices, x_indices = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    donut_mask = (distance_from_center >= inner_radius) & (distance_from_center <= outer_radius)
    
    # ドーナツ状範囲の画像抽出
    donut_img = img.copy()
    donut_img[~donut_mask] = 0  # ドーナツ範囲外を黒にする
    
    # ドーナツ状範囲の特徴量計算
    white_mask = (
        (donut_img[:, :, 0] > 180)
        & (donut_img[:, :, 1] > 180)
        & (donut_img[:, :, 2] > 180)
    )
    white_ratio = white_mask[donut_mask].mean() if donut_mask.any() else 0
    
    hsv = cv2.cvtColor(donut_img, cv2.COLOR_BGR2HSV)
    mean_s = hsv[:, :, 1][donut_mask].mean() if donut_mask.any() else 0
    
    gray = cv2.cvtColor(donut_img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edge[donut_mask]) / donut_mask.sum() if donut_mask.sum() > 0 else 0

    return np.array([area, white_ratio, mean_s, edge_ratio], dtype=np.float32)


def get_matched_index(features1, features2):
    """ハンガリアンマッチングで食器の種別を判定"""
    if not NUMPY_AVAILABLE or not SCIPY_AVAILABLE:
        return list(range(len(features1)))
        
    features1 = np.array(features1, dtype=np.float32)
    features2 = np.array(features2, dtype=np.float32)

    all_features = np.vstack([features1, features2])
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    std[std < 1e-6] = 1.0

    features1 = (features1 - mean) / std
    features2 = (features2 - mean) / std

    weights = np.array([1.0, 0, 1.0, 0], dtype=np.float32)

    cost_matrix = np.zeros((len(features2), len(features1)), dtype=np.float32)
    for i, f2 in enumerate(features2):
        for j, f1 in enumerate(features1):
            diff = (f1 - f2) * weights
            cost_matrix[i, j] = np.linalg.norm(diff)

    _, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind.tolist()


def classify_plates(data_path="sumikko_analysis", filename_mode='auto', tray_ids=None, plate_types=None):
    """食器の分類と整理（新旧ファイル名対応）"""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCVが利用できません。opencv-pythonをインストールしてください。")
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPyが利用できません。numpyをインストールしてください。")
        
    plate_names = [os.path.basename(d) for d in glob.glob(f"{data_path}/plate_images_tmp/*")]

    if tray_ids:
        plate_names = [p for p in plate_names if p.split("_")[0] in tray_ids]

    print(f"分類対象のプレート数: {len(plate_names)}")
    print(f"ファイル名モード: {filename_mode}")

    results = []
    
    for plate_idx, plate_name in enumerate(plate_names):
        if (plate_idx + 1) % 10 == 0:
            print(f"分類進捗: {plate_idx + 1}/{len(plate_names)}")
            
        try:
            features = []
            plate_images_tmp = sorted(glob.glob(f"{data_path}/plate_images_tmp/{plate_name}/*_tmp.png"))
            if len(plate_images_tmp) == 0:
                print(f"警告: {plate_name} の一時画像が見つかりません")
                continue

            for plate_image_tmp in plate_images_tmp:
                img = cv2.imread(plate_image_tmp)
                if img is None:
                    print(f"警告: {plate_image_tmp} の読み込みに失敗")
                    continue
                feature = calc_feature(img)
                features.append(feature)

            if len(features) == 0:
                print(f"警告: {plate_name} の特徴量が計算できませんでした")
                continue

            plate_id = plate_name.split("_")[0]
            base_images = sorted(glob.glob(f"{data_path}/test_data/{plate_id}/*.png"))
            base_features = []

            for plate_image in base_images:
                img = cv2.imread(plate_image)
                if img is None:
                    continue
                feature = calc_feature(img)
                base_features.append(feature)

            if len(base_features) == 0:
                print(f"警告: {plate_id} の基準特徴量が見つかりません")
                continue

            matched_indices = get_matched_index(base_features, features)

            setting_path = f"{data_path}/test_data/{plate_id}/plate_setting.json"
            if not os.path.exists(setting_path):
                print(f"警告: {plate_id} の設定ファイルが見つかりません")
                continue
                
            with open(setting_path, encoding="utf-8") as f:
                base_config = json.load(f)
            idx_to_plate_type = {d["plate_number"]: d["plate_name"] for d in base_config}

            # ファイル名をパース
            parsed = parse_filename(plate_name, filename_mode)
            
            for i, idx in enumerate(matched_indices):
                try:
                    if idx >= len(matched_indices) or i >= len(plate_images_tmp):
                        print(f"警告: {plate_name} のインデックスが範囲外です")
                        continue
                        
                    meal_type = idx_to_plate_type.get(idx, "unknown")

                    if plate_types and meal_type not in plate_types:
                        continue
                    
                    # ファイル名から割合を取得
                    if parsed and parsed['format'] == 'old':
                        # 旧形式: ファイル名から直接比率を取得
                        if i < len(parsed['ratios']):
                            amount = int(parsed['ratios'][i] * 10)  # 0.5 -> 5
                        else:
                            amount = 0
                        timestamp = parsed['timestamp']
                    elif parsed and parsed['format'] == 'new':
                        # 新形式: タイムスタンプのみ使用
                        timestamp = parsed['timestamp']
                        # 未知割合なのでデフォルト値を使用
                        amount = 0  # 0割として扱う（推定のみ実行）
                    else:
                        # 自動判定失敗
                        timestamp = "_".join(plate_name.split("_")[-2:])
                        amount = 0

                    os.makedirs(f"{data_path}/plate_images/{plate_id}/{meal_type}", exist_ok=True)

                    src_png = plate_images_tmp[i]
                    dst_png = f"{data_path}/plate_images/{plate_id}/{meal_type}/plate_{amount:02d}_{timestamp}.png"
                    src_npy = src_png.replace(".png", ".npy")
                    dst_npy = dst_png.replace(".png", ".npy")

                    if os.path.exists(src_png):
                        os.rename(src_png, dst_png)
                    if os.path.exists(src_npy):
                        os.rename(src_npy, dst_npy)
                except Exception as e:
                    print(f"エラー: {plate_name} の食器 {i} の移動に失敗: {e}")
                    continue
            
            results.append({
                "plate_name": plate_name,
                "status": "classified"
            })
            
        except Exception as e:
            print(f"エラー: {plate_name} の分類中に例外が発生: {e}")
            continue

    if os.path.exists(f"{data_path}/plate_images_tmp"):
        total_size = sum(f.stat().st_size for f in Path(f"{data_path}/plate_images_tmp").rglob("*") if f.is_file())
        if total_size == 0:
            shutil.rmtree(f"{data_path}/plate_images_tmp")

    print(f"分類完了: 処理数={len(results)}, 総数={len(plate_names)}")
    print(f"保存場所: {data_path}/plate_images/")
    print(f"  - 分類済み画像: {data_path}/plate_images/<tray_id>/<meal_type>/plate_<amount>_<timestamp>.png")
    print(f"  - 分類済み深度: {data_path}/plate_images/<tray_id>/<meal_type>/plate_<amount>_<timestamp>.npy")

    return {
        "total_processed": len(results),
        "results": results
    }


def get_plate_settings(data_path="sumikko_analysis"):
    """全ての食器設定を取得"""
    config_path = os.path.join(data_path, "config.json")
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    
    tray_ids = [d["id"] for d in config["shokushu"]]
    settings = {}
    
    for tray_id in tray_ids:
        setting_path = f"{data_path}/test_data/{tray_id}/plate_setting.json"
        if os.path.exists(setting_path):
            with open(setting_path, encoding="utf-8") as f:
                settings[tray_id] = json.load(f)
        else:
            settings[tray_id] = []
    
    return settings


def update_plate_setting(tray_id, plate_number, plate_name, nansai, data_path="sumikko_analysis"):
    """食器設定を更新"""
    print(f"Updating plate setting: {tray_id}, {plate_number}, {plate_name}, {nansai}")
    setting_path = f"{data_path}/test_data/{tray_id}/plate_setting.json"
    
    if os.path.exists(setting_path):
        with open(setting_path, encoding="utf-8") as f:
            settings = json.load(f)
    else:
        settings = []
    
    # 該当するplate_numberを探して更新
    updated = False
    for i, setting in enumerate(settings):
        if setting["plate_number"] == plate_number:
            settings[i]["plate_name"] = plate_name
            settings[i]["nansai"] = nansai
            updated = True
            break
    
    if not updated:
        settings.append({
            "plate_number": plate_number,
            "plate_name": plate_name,
            "nansai": nansai
        })
    
    # plate_numberでソート
    settings.sort(key=lambda x: x["plate_number"])
    
    with open(setting_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)
    
    return settings
