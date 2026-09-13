import os
import json
import numpy as np

# matplotlibとseabornのインポート（オプション）
try:
    import matplotlib
    matplotlib.use('Agg')  # 非インタラクティブモード
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("警告: matplotlib/seaborn/scikit-learnがインストールされていません。可視化機能は無効になります。")

from datetime import datetime

if PLOTTING_AVAILABLE:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'AppleGothic']

LEVELS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def to_index(values):
    return np.array([np.argmin(np.abs(LEVELS - v)) for v in values])


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


def draw_confusion_matrix(y_true, y_pred, title, r2, mae, save_path):
    """混同行列の描画と保存"""
    if not PLOTTING_AVAILABLE:
        print("警告: プロット機能が利用できません")
        return None
        
    y_true_idx = to_index(y_true)
    y_pred_idx = to_index(y_pred)

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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path


def generate_results(data_path="sumikko_analysis"):
    """結果の集計と可視化"""
    results_path = f"{data_path}/final_results.json"
    if not os.path.exists(results_path):
        return {
            "status": "error",
            "message": "final_results.jsonが見つかりません。先にStep 3を実行してください。"
        }
    
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    plate_settings = {}

    for plate_id in ["A", "B", "C", "D"]:
        setting_path = f"{data_path}/test_data/{plate_id}/plate_setting.json"
        if os.path.exists(setting_path):
            with open(setting_path, encoding="utf-8") as f:
                setting = json.load(f)
            plate_settings[plate_id] = {
                item["plate_name"]: item["nansai"]
                for item in setting
            }
        else:
            plate_settings[plate_id] = {}

    true_nansai = []
    pred_nansai = []
    true_other = []
    pred_other = []

    for result in results:
        plate_id = result["plate_id"]
        plate_name = result["plate_name"]
        
        nansai = plate_settings.get(plate_id, {}).get(plate_name, False)

        if nansai:
            true_nansai.extend(result["y_true"])
            pred_nansai.extend(result["y_pred"])
        else:
            true_other.extend(result["y_true"])
            pred_other.extend(result["y_pred"])

    os.makedirs(f"{data_path}/result_analyze", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_files = []

    if len(true_nansai) > 0:
        r2_nansai, mae_nansai = calc_score(true_nansai, pred_nansai)
        nansai_path = f"{data_path}/result_analyze/cm_nansai_{timestamp}.png"
        
        if PLOTTING_AVAILABLE:
            draw_confusion_matrix(
                1 - np.array(true_nansai),
                1 - np.array(pred_nansai),
                "液体食品推定結果",
                r2_nansai,
                mae_nansai,
                nansai_path,
            )
        
        output_files.append({
            "type": "nansai",
            "path": nansai_path,
            "r2": r2_nansai,
            "mae": mae_nansai
        })

    if len(true_other) > 0:
        r2_other, mae_other = calc_score(true_other, pred_other)
        other_path = f"{data_path}/result_analyze/cm_not_nansai_{timestamp}.png"
        
        if PLOTTING_AVAILABLE:
            draw_confusion_matrix(
                1 - np.array(true_other),
                1 - np.array(pred_other),
                "固体食品推定結果",
                r2_other,
                mae_other,
                other_path,
            )
        
        output_files.append({
            "type": "other",
            "path": other_path,
            "r2": r2_other,
            "mae": mae_other
        })

    return {
        "status": "completed",
        "output_files": output_files,
        "nansai_count": len(true_nansai),
        "other_count": len(true_other),
        "plotting_available": PLOTTING_AVAILABLE
    }
