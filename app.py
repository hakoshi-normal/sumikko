# -*- coding: utf-8 -*-
"""
食後食器に対する深度データによる食事摂取量推定システム (GUI版)

データ構造:
    data/[MMDD]/[タイプ1]/[トレーID]_[タイプ2]_[YYYYMMDD]_[連番].png / .npy
      - タイプ1: "0"=空皿, "10"=満杯, 任意名=食後(P)
      - タイプ2: "0", "10", "P"
      - 連番  : 1-5 (深度は平均、画像は連番最大のものを使用)

出力構造:
    work/test_data/{MMDD}_{トレーID}/            基準食器画像 + plate_setting.json
    work/plate_images/{MMDD}_{トレーID}/{食種}/  分類済み食器画像 + メタ情報 + 平均深度
    work/cache/                                  YOLO/マスクのキャッシュ
    work/result_analyze/                         混同行列画像
    final_results.json                           推定結果
    measured_results.json                        実測値 (final_results.jsonと同形式)
"""

import os
import re
import glob
import json
import random
import hashlib
import shutil
import threading

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WORK_DIR = os.path.join(BASE_DIR, "work")
TEST_DATA_DIR = os.path.join(WORK_DIR, "test_data")
PLATE_IMG_DIR = os.path.join(WORK_DIR, "plate_images")
CACHE_DIR = os.path.join(WORK_DIR, "cache")
RESULT_DIR = os.path.join(WORK_DIR, "result_analyze")
FINAL_RESULTS = os.path.join(BASE_DIR, "final_results.json")
MEASURED_RESULTS = os.path.join(BASE_DIR, "measured_results.json")

MEAN_DEPTH_DIR = os.path.join(WORK_DIR, "mean_depth")

for d in [DATA_DIR, WORK_DIR, TEST_DATA_DIR, PLATE_IMG_DIR, CACHE_DIR, RESULT_DIR, MEAN_DEPTH_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# ログ / 進捗
# ---------------------------------------------------------------------------
PROGRESS = {"running": False, "current": 0, "total": 0, "message": "idle"}


def log(msg):
    print(msg, flush=True)


def set_progress(current, total, message=""):
    PROGRESS.update({"running": total > 0 and current < total,
                     "current": current, "total": total, "message": message})
    if total:
        log(f"[{current}/{total}] {message}")

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "tableware_best_ncnn_model")
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "models", "unet_food_int8_256.tflite")

with open(os.path.join(BASE_DIR, "config.json"), encoding="utf-8") as f:
    CONFIG = json.load(f)
PLATE_NAMES = CONFIG.get("plate_names") or [k for k in CONFIG.keys() if k != "shokushu"]

# ---------------------------------------------------------------------------
# 遅延ロードされる重いモデル
# ---------------------------------------------------------------------------
_yolo = None
_interpreter = None
_lock = threading.Lock()


def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(YOLO_MODEL_PATH, task="detect", verbose=False)
    return _yolo


def get_interpreter():
    global _interpreter
    if _interpreter is None:
        from ai_edge_litert.interpreter import Interpreter
        _interpreter = Interpreter(model_path=TFLITE_MODEL_PATH,
                                   num_threads=os.cpu_count() or 4)
        _interpreter.allocate_tensors()
    return _interpreter


# ---------------------------------------------------------------------------
# データ走査ユーティリティ
# ---------------------------------------------------------------------------
FNAME_RE = re.compile(r"^([A-Z])_(0|10|P)_(\d+)_(\d+)\.(png|npy)$")


def tray_key(date, tray):
    return f"{date}_{tray}"


def scan_data():
    """data/ 以下を走査し {date: {tray: {type1: [...]}}} を返す"""
    out = {}
    if not os.path.isdir(DATA_DIR):
        return out
    for date in sorted(os.listdir(DATA_DIR)):
        ddir = os.path.join(DATA_DIR, date)
        if not os.path.isdir(ddir):
            continue
        for type1 in sorted(os.listdir(ddir)):
            tdir = os.path.join(ddir, type1)
            if not os.path.isdir(tdir):
                continue
            for fn in sorted(os.listdir(tdir)):
                m = FNAME_RE.match(fn)
                if not m:
                    continue
                tray, type2, ymd, seq, ext = m.groups()
                ent = out.setdefault(date, {}).setdefault(tray, {}).setdefault(type1, {})
                ent.setdefault("files", set()).add(fn.rsplit(".", 1)[0])
                ent["type2"] = type2
    # set -> sorted list
    for date in out:
        for tray in out[date]:
            for t1 in out[date][tray]:
                out[date][tray][t1]["files"] = sorted(out[date][tray][t1]["files"])
    return out


def capture_path(date, type1, stem, ext):
    return os.path.join(DATA_DIR, date, type1, f"{stem}.{ext}")


def seqs_of(type1_dir, stem_prefix):
    """stem_prefix = {tray}_{type2}_{ymd} に一致する連番一覧"""
    seqs = []
    for fn in os.listdir(type1_dir):
        m = FNAME_RE.match(fn)
        if m and fn.startswith(stem_prefix) and m.group(5) == "npy":
            seqs.append(int(m.group(4)))
    return sorted(seqs)


def _mean_depth_raw(date, type1, stem_prefix):
    """連番1-5の深度を平均して返す。stem_prefix={tray}_{type2}_{ymd}"""
    tdir = os.path.join(DATA_DIR, date, type1)
    seqs = seqs_of(tdir, stem_prefix)
    if not seqs:
        return None
    depths = []
    for s in seqs:
        p = os.path.join(tdir, f"{stem_prefix}_{s}.npy")
        if os.path.exists(p):
            depths.append(np.load(p).astype(np.float32))
    if not depths:
        return None
    return np.mean(depths, axis=0)


def mean_depth_path(date, type1, stem_prefix):
    return os.path.join(MEAN_DEPTH_DIR, date, type1, f"{stem_prefix}_mean.npy")


def mean_depth(date, type1, stem_prefix):
    """平均深度を返す。事前計算済み(work/mean_depth/)があればそれを使用。"""
    cp = mean_depth_path(date, type1, stem_prefix)
    if os.path.exists(cp):
        return np.load(cp)
    return _mean_depth_raw(date, type1, stem_prefix)


def precompute_mean_depth(date, tray):
    """対象トレーの全タイプ1・全撮影の平均深度を事前計算して保存（前処理）"""
    done, total = 0, 0
    targets = []
    for type1 in list_type1_dirs(date):
        t2 = type2_of(date, type1, tray)
        if t2 is None:
            continue
        for prefix in list_captures(date, type1, tray, t2):
            targets.append((type1, prefix))
    total = len(targets)
    set_progress(0, total, f"{date}_{tray}: 平均深度の事前計算")
    for type1, prefix in targets:
        cp = mean_depth_path(date, type1, prefix)
        if not os.path.exists(cp):
            d = _mean_depth_raw(date, type1, prefix)
            if d is not None:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                np.save(cp, d.astype(np.float32))
        done += 1
        set_progress(done, total, f"{date}_{tray}: 平均深度 {type1}/{prefix}")
    return total


def rep_image_path(date, type1, stem_prefix):
    """画像は連番最大(5番目)のものを採用"""
    tdir = os.path.join(DATA_DIR, date, type1)
    seqs = []
    for fn in os.listdir(tdir):
        m = FNAME_RE.match(fn)
        if m and fn.startswith(stem_prefix) and m.group(5) == "png":
            seqs.append(int(m.group(4)))
    if not seqs:
        return None
    return os.path.join(tdir, f"{stem_prefix}_{max(seqs)}.png")


def list_captures(date, type1, tray, type2):
    """指定条件の撮影セット (stem_prefixの一覧)"""
    tdir = os.path.join(DATA_DIR, date, type1)
    if not os.path.isdir(tdir):
        return []
    prefixes = set()
    for fn in os.listdir(tdir):
        m = FNAME_RE.match(fn)
        if m and m.group(1) == tray and m.group(2) == type2:
            prefixes.add(f"{tray}_{type2}_{m.group(3)}")
    return sorted(prefixes)


def list_type1_dirs(date):
    ddir = os.path.join(DATA_DIR, date)
    if not os.path.isdir(ddir):
        return []
    return sorted(t1 for t1 in os.listdir(ddir)
                  if os.path.isdir(os.path.join(ddir, t1)))


def type2_of(date, type1, tray):
    """そのtype1ディレクトリ内のトレーのタイプ2を返す"""
    tdir = os.path.join(DATA_DIR, date, type1)
    if not os.path.isdir(tdir):
        return None
    for fn in os.listdir(tdir):
        m = FNAME_RE.match(fn)
        if m and m.group(1) == tray:
            return m.group(2)
    return None


# ---------------------------------------------------------------------------
# 検出・切り出し関連 (1_detect_plate.ipynb 相当)
# ---------------------------------------------------------------------------
def calc_depth_var(depth):
    h, w = depth.shape
    center = (w // 2, h // 2)
    radius = min(h, w) // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.circle(mask, center, radius, 1, -1)
    depth_masked = depth * mask
    data = depth_masked.flatten()
    data = np.delete(data, data == 0)
    if data.size == 0:
        return 0.0
    bins = np.linspace(np.min(data), np.max(data), 51)
    hist, _ = np.histogram(data, bins=bins)
    nums = [d / data.size for d in hist.tolist()]
    return float(np.var(nums))


def rotate_by_region(img, depth, cx, cy, W, H):
    if cx < W / 2 and cy < H / 2:
        k = 0
    elif cx >= W / 2 and cy < H / 2:
        k = 1
    elif cx >= W / 2 and cy >= H / 2:
        k = 2
    else:
        k = 3
    return np.rot90(img, k=k), np.rot90(depth, k=k) if depth is not None else None, k


def yolo_boxes(image_path, use_cache=True):
    """YOLO検出(信頼度順上位5件)。キャッシュ対応。"""
    key = hashlib.md5(os.path.abspath(image_path).encode()).hexdigest()
    cpath = os.path.join(CACHE_DIR, "boxes", f"{key}.npy")
    if use_cache and os.path.exists(cpath):
        return np.load(cpath)
    results = get_yolo()(image_path, verbose=False)
    boxes = results[0].boxes
    sorted_boxes = boxes[boxes.conf.argsort(descending=True)]
    arr = sorted_boxes[:5].xyxy.cpu().numpy()
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    np.save(cpath, arr)
    return arr


def detect_reference(date, tray, use_cache=True):
    """10割画像から食器を検出し work/test_data/{date}_{tray}/ に保存"""
    prefixes = list_captures(date, "10", tray, "10")
    if not prefixes:
        raise HTTPException(404, f"{date}/{tray}: 10割データがありません")
    prefix = prefixes[-1]
    log(f"{date}_{tray}: 食器検出 ({prefix})")
    img_path = rep_image_path(date, "10", prefix)
    depth = mean_depth(date, "10", prefix)
    if img_path is None or depth is None:
        raise HTTPException(404, f"{date}/{tray}: 10割データが不完全です")

    boxes = yolo_boxes(img_path, use_cache)
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    outdir = os.path.join(TEST_DATA_DIR, tray_key(date, tray))
    os.makedirs(outdir, exist_ok=True)
    # 既存のplate_*.pngを削除
    for p in glob.glob(os.path.join(outdir, "plate_*.png")):
        os.remove(p)

    depth_var_list = []
    crops = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        dish_img = img[y1:y2, x1:x2]
        dish_depth = depth[y1:y2, x1:x2]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dish_img, dish_depth, k = rotate_by_region(dish_img, dish_depth, cx, cy, W, H)
        cv2.imwrite(os.path.join(outdir, f"plate_{i}.png"), dish_img)
        crops.append({"box": [int(v) for v in box], "k": int(k)})
        depth_var_list.append(calc_depth_var(dish_depth))

    # 軟菜食判定 (最大ギャップで2分割)
    n = len(depth_var_list)
    if n >= 2:
        arr = np.array(depth_var_list)
        si = np.argsort(arr)
        sa = arr[si]
        diff = np.diff(sa)
        gi = int(np.argmax(diff))
        th = (sa[gi] + sa[gi + 1]) / 2
        binary = (arr > th).tolist()
    else:
        binary = [False] * n

    # 既存設定があれば plate_number 単位で食種名・軟菜フラグを引き継ぐ
    prev = {}
    sp = os.path.join(outdir, "plate_setting.json")
    if os.path.exists(sp):
        try:
            for d in json.load(open(sp, encoding="utf-8")):
                prev[d["plate_number"]] = d
        except Exception:
            pass

    setting = [
        {
            "plate_name": prev.get(i, {}).get("plate_name")
                          or (PLATE_NAMES[i] if i < len(PLATE_NAMES) else f"plate_{i}"),
            "plate_number": i,
            "nansai": bool(prev.get(i, {}).get("nansai", binary[i])),
        }
        for i in range(n)
    ]
    with open(os.path.join(outdir, "plate_setting.json"), "w", encoding="utf-8") as f:
        json.dump(setting, f, indent=4, ensure_ascii=False)
    with open(os.path.join(outdir, "detect_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"source": img_path, "crops": crops}, f, indent=4)

    # ボックス描画プレビュー
    prev = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(prev, str(i), (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(outdir, "tray_boxes.png"), prev)
    return {"n_dishes": n, "depth_var": depth_var_list, "setting": setting}


# ---------------------------------------------------------------------------
# 特徴量マッチング (ハンガリアン)
# ---------------------------------------------------------------------------
def calc_feature(img):
    h, w = img.shape[:2]
    area = h * w
    white_mask = (img[:, :, 0] > 180) & (img[:, :, 1] > 180) & (img[:, :, 2] > 180)
    white_ratio = white_mask.mean()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_s = hsv[:, :, 1].mean()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edge) / edge.size
    return np.array([area, white_ratio, mean_s, edge_ratio], dtype=np.float32)


def get_matched_index(features1, features2):
    features1 = np.array(features1, dtype=np.float32)
    features2 = np.array(features2, dtype=np.float32)
    all_features = np.vstack([features1, features2])
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    std[std < 1e-6] = 1.0
    features1 = (features1 - mean) / std
    features2 = (features2 - mean) / std
    weights = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    cost = np.zeros((len(features2), len(features1)), dtype=np.float32)
    for i, f2 in enumerate(features2):
        for j, f1 in enumerate(features1):
            cost[i, j] = np.linalg.norm((f1 - f2) * weights)
    _, col_ind = linear_sum_assignment(cost)
    return col_ind.tolist()


def classify_tray(date, tray, type1s=None, use_cache=True):
    """対象トレーの食器を分類し plate_images/ に保存。
    type1s: 任意名タイプ1のリスト (None=全て)。0/10は常に対象。"""
    key = tray_key(date, tray)
    ref_dir = os.path.join(TEST_DATA_DIR, key)
    setting_path = os.path.join(ref_dir, "plate_setting.json")
    if not os.path.exists(setting_path):
        raise HTTPException(404, f"{key}: 基準データ(plate_setting.json)がありません。先に食器検出を実行してください。")
    base_config = json.load(open(setting_path, encoding="utf-8"))
    idx_to_plate = {d["plate_number"]: d["plate_name"] for d in base_config}

    base_images = sorted(glob.glob(os.path.join(ref_dir, "plate_*.png")))
    base_features = [calc_feature(cv2.imread(p)) for p in base_images]
    gt_count = len(base_images)
    if gt_count == 0:
        raise HTTPException(404, f"{key}: 基準食器が0件です。先に食器検出を実行してください。")

    # 前回の分類結果をクリア（マッチング結果が変わった場合の残存ファイルを防ぐ）
    out_base = os.path.join(PLATE_IMG_DIR, key)
    if os.path.isdir(out_base):
        shutil.rmtree(out_base)

    dirs = [t1 for t1 in list_type1_dirs(date)
            if t1 in ("0", "10") or type1s is None or t1 in type1s]
    total = sum(len(list_captures(date, t1, tray, type2_of(date, t1, tray) or ""))
                for t1 in dirs)
    done = 0
    set_progress(0, total, f"{key}: 食器分類")
    summary = []
    for type1 in dirs:
        t2 = type2_of(date, type1, tray)
        if t2 is None:
            continue
        for prefix in list_captures(date, type1, tray, t2):
            done += 1
            set_progress(done, total, f"{key}: {type1}/{prefix} を分類中")
            img_path = rep_image_path(date, type1, prefix)
            depth = mean_depth(date, type1, prefix)
            if img_path is None or depth is None:
                continue
            img = cv2.imread(img_path)
            H, W = img.shape[:2]
            boxes = yolo_boxes(img_path, use_cache)
            status = "ok"
            if len(boxes) != gt_count:
                status = f"warn:検出数{len(boxes)}≠基準{gt_count}"
            n = min(len(boxes), gt_count)

            features, crops = [], []
            for i in range(n):
                x1, y1, x2, y2 = map(int, boxes[i])
                d_img = img[y1:y2, x1:x2]
                d_dep = depth[y1:y2, x1:x2]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d_img, d_dep, k = rotate_by_region(d_img, d_dep, cx, cy, W, H)
                crops.append((d_img, d_dep, k, [int(v) for v in boxes[i]]))
                features.append(calc_feature(d_img))

            if features:
                matched = get_matched_index(base_features, features)
            else:
                matched = []

            mapping = []
            for i, midx in enumerate(matched):
                if i >= len(crops):
                    break
                meal = idx_to_plate.get(midx, f"plate_{midx}")
                d_img, d_dep, k, box = crops[i]
                odir = os.path.join(PLATE_IMG_DIR, key, meal)
                os.makedirs(odir, exist_ok=True)
                stem = f"{type1}_{prefix}"
                cv2.imwrite(os.path.join(odir, f"{stem}.png"), d_img)
                np.save(os.path.join(odir, f"{stem}_mean.npy"), d_dep.astype(np.float32))
                meta = {
                    "date": date, "tray": tray, "type1": type1, "type2": t2,
                    "prefix": prefix, "meal": meal,
                    "box": box, "rot_k": int(k),
                }
                with open(os.path.join(odir, f"{stem}.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=4, ensure_ascii=False)
                mapping.append({"crop": i, "meal": meal})
            summary.append({"type1": type1, "prefix": prefix, "status": status, "mapping": mapping})
    return summary


# ---------------------------------------------------------------------------
# 摂取量推定 (2_estimate_amount.ipynb 相当)
# ---------------------------------------------------------------------------
UNET_SIZE = (256, 256)
UNET_TH = 0.5


def _unet_mask_one(image):
    """U-Netで1枚分の食器マスク(256x256 bool)を推論"""
    interpreter = get_interpreter()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, UNET_SIZE).astype(np.float32) / 255.0
    interpreter.set_tensor(in_det[0]["index"], np.expand_dims(resized, 0))
    interpreter.invoke()
    return interpreter.get_tensor(out_det[0]["index"])[0][:, :, 0] > UNET_TH


def get_tablewaremask(images, unet_flg=False, cache_keys=None, use_cache=True):
    """images: [empty, full, target] の食器画像。
    cache_keys: 画像ごとのキャッシュキー(リスト)。画像単位でキャッシュし論理和を返す。"""
    h, w, _ = images[0].shape
    if unet_flg:
        masks = []
        for i, image in enumerate(images):
            m = None
            cpath = None
            if cache_keys:
                cpath = os.path.join(CACHE_DIR, "masks", f"{cache_keys[i]}.npy")
                if use_cache and os.path.exists(cpath):
                    m = np.load(cpath) > 0
            if m is None:
                m = _unet_mask_one(image)
                if cpath:
                    os.makedirs(os.path.dirname(cpath), exist_ok=True)
                    np.save(cpath, m.astype(np.uint8))
            masks.append(m)
        mask = np.any(np.stack(masks, axis=0), axis=0).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask, None
    else:
        center = (w // 2, h // 2)
        radius = int(min(h, w) / 2 * 0.5)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask = cv2.circle(mask, center, radius, 255, -1)
        return mask, None


def calc_amount(depth_empty, depth_full, depth_after, meal_mask,
                depth_min=2000, depth_max=3500, inpaint_radius=3):
    h, w = depth_empty.shape
    depth_full = cv2.resize(depth_full, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_after = cv2.resize(depth_after, (w, h), interpolation=cv2.INTER_NEAREST)

    def clip_and_inpaint(depth):
        depth = depth.astype(np.float32)
        depth[(depth < depth_min) | (depth > depth_max)] = np.nan
        mask = np.isnan(depth).astype(np.uint8) * 255
        depth_filled = np.nan_to_num(depth, nan=0.0)
        if np.any(mask):
            depth_inp = cv2.inpaint(depth_filled, mask, inpaint_radius, cv2.INPAINT_TELEA)
        else:
            depth_inp = depth_filled
        depth_inp[meal_mask == 0] = 0
        return depth_inp

    depth_empty = clip_and_inpaint(depth_empty)
    depth_full = clip_and_inpaint(depth_full)
    depth_after = clip_and_inpaint(depth_after)

    h_full = depth_empty - depth_full
    h_after = depth_empty - depth_after

    valid = meal_mask > 0
    if np.sum(valid) == 0:
        return None
    V_full = np.nanmean(h_full[valid])
    V_after = np.nanmean(h_after[valid])
    if V_full < 1e-6:
        return 0.0
    ratio = float(np.clip(V_after / V_full, 0, 1))
    if V_full < 0:
        ratio = 1.0
    if V_after < 0:
        ratio = 0.0
    return ratio


def load_crop(date, tray, meal, stem):
    p = os.path.join(PLATE_IMG_DIR, tray_key(date, tray), meal, stem)
    img = cv2.imread(p + ".png")
    meta = json.load(open(p + ".json", encoding="utf-8"))
    dep = np.load(p + "_mean.npy") if os.path.exists(p + "_mean.npy") else None
    return img, dep, meta


def depth_for_meta(meta, seq=None):
    """メタ情報から食器深度を取得。seq指定時はその連番のみ使用。"""
    tdir = os.path.join(DATA_DIR, meta["date"], meta["type1"])
    box = meta["box"]
    if seq is None:
        cp = mean_depth_path(meta["date"], meta["type1"], meta["prefix"])
        if os.path.exists(cp):
            x1, y1, x2, y2 = map(int, box)
            d = np.load(cp).astype(np.float32)[y1:y2, x1:x2]
            return np.rot90(d, k=meta["rot_k"])
    seqs = [seq] if seq else seqs_of(tdir, meta["prefix"])
    x1, y1, x2, y2 = map(int, box)
    ds = []
    for s in seqs:
        p = os.path.join(tdir, f"{meta['prefix']}_{s}.npy")
        if os.path.exists(p):
            d = np.load(p).astype(np.float32)[y1:y2, x1:x2]
            ds.append(d)
    if not ds:
        return None
    d = np.mean(ds, axis=0)
    return np.rot90(d, k=meta["rot_k"])


def estimate_tray(date, tray, use_unet=True, use_cache=True,
                  n_trials=1, type1s=None):
    """1トレー分の推定。レコードのリストを返す。
    基準深度は試行ごとに0割/10割の撮影セット・連番をランダム選択。
    type1s: 推定対象の任意名タイプ1のリスト (None=全て)。"""
    key = tray_key(date, tray)
    ref_dir = os.path.join(TEST_DATA_DIR, key)
    setting_path = os.path.join(ref_dir, "plate_setting.json")
    if not os.path.exists(setting_path):
        raise HTTPException(404, f"{key}: 基準データがありません")
    base_config = json.load(open(setting_path, encoding="utf-8"))
    nansai_map = {d["plate_name"]: d["nansai"] for d in base_config}

    records = []
    meals_dir = os.path.join(PLATE_IMG_DIR, key)
    if not os.path.isdir(meals_dir):
        raise HTTPException(404, f"{key}: 分類済み食器がありません。先に分類を実行してください。")

    meals = [m for m in sorted(os.listdir(meals_dir))
             if os.path.isdir(os.path.join(meals_dir, m))]
    # 進捗用に対象数を事前計算
    total = 0
    for meal in meals:
        for p in glob.glob(os.path.join(meals_dir, meal, "*.json")):
            s = os.path.basename(p)[:-5]
            if not s.startswith("0_") and not s.startswith("10_"):
                if type1s is None or any(s.startswith(t + "_") for t in type1s):
                    total += 1
    done = 0
    set_progress(0, total, f"{key}: 摂取量推定")

    for meal in meals:
        mdir = os.path.join(meals_dir, meal)
        stems = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(mdir, "*.json"))]
        empty = sorted(s for s in stems if s.startswith("0_"))
        full = sorted(s for s in stems if s.startswith("10_"))
        targets = [s for s in stems if not s.startswith("0_") and not s.startswith("10_")
                   and (type1s is None or any(s.startswith(t + "_") for t in type1s))]
        if not empty or not full:
            continue
        nansai = bool(nansai_map.get(meal, False))
        unet_flg = use_unet and not nansai

        # 食器画像・メタ情報のキャッシュ
        _img_cache, _meta_cache = {}, {}
        def get_crop(stem):
            if stem not in _img_cache:
                img, _, meta = load_crop(date, tray, meal, stem)
                _img_cache[stem] = img
                _meta_cache[stem] = meta
            return _img_cache[stem], _meta_cache[stem]

        for tgt in sorted(targets):
            done += 1
            set_progress(done, total, f"{key}: {meal}/{tgt} を推定中")
            t_img, t_meta = get_crop(tgt)
            type1 = t_meta["type1"]
            trials = max(1, n_trials)
            preds = []
            used_refs = []
            for t in range(trials):
                # 試行ごとに0割/10割の撮影セットと連番をランダムに割当て
                e_stem, f_stem = random.choice(empty), random.choice(full)
                e_img, e_meta = get_crop(e_stem)
                f_img, f_meta = get_crop(f_stem)
                seqs_e = seqs_of(os.path.join(DATA_DIR, date, e_meta["type1"]), e_meta["prefix"])
                seqs_f = seqs_of(os.path.join(DATA_DIR, date, f_meta["type1"]), f_meta["prefix"])
                d_e = depth_for_meta(e_meta, seq=random.choice(seqs_e))
                d_f = depth_for_meta(f_meta, seq=random.choice(seqs_f))
                # 食後側は連番平均深度を使用
                d_a = depth_for_meta(t_meta)
                if d_e is None or d_f is None or d_a is None:
                    continue
                # マスクは画像単位でキャッシュ（組合せが変わっても再利用される）
                mask, _ = get_tablewaremask(
                    [e_img, f_img, t_img],
                    unet_flg=unet_flg,
                    cache_keys=[hashlib.md5((key + meal + s).encode()).hexdigest()
                                for s in (e_stem, f_stem, tgt)],
                    use_cache=use_cache)
                r = calc_amount(d_e, d_f, d_a, mask)
                if r is not None:
                    preds.append(r)
                    used_refs.append({"empty": e_stem, "full": f_stem})
            if not preds:
                continue
            records.append({
                "plate_id": key,
                "plate_name": meal,
                "target": type1,
                "target_file": tgt,
                "n_trials": len(preds),
                "ref_sets": used_refs,        # 各試行で使用した基準セット
                "y_true": None,
                "y_pred": preds,              # 全試行の推定値
                "y_pred_median": float(np.median(preds)),  # 採用値(中央値)
                "r2": None,
                "mae": None,
            })
    return records


# ---------------------------------------------------------------------------
# 結果評価 (3_result_draw.ipynb 相当)
# ---------------------------------------------------------------------------
LEVELS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def to_index(values):
    return np.array([np.argmin(np.abs(LEVELS - v)) for v in values])


def calc_score(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_absolute_error
    return float(r2_score(y_true, y_pred)), float(mean_absolute_error(y_true, y_pred))


def draw_confusion_matrix(y_true, y_pred, title, r2, mae, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Meiryo", "Yu Gothic",                    # Windows
        "Hiragino Sans", "AppleGothic",           # macOS
        "Noto Sans CJK JP", "IPAexGothic", "TakaoGothic",  # Linux
    ]

    cm = confusion_matrix(to_index(y_true), to_index(y_pred)).T
    cm = np.flipud(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LEVELS, yticklabels=LEVELS[::-1])
    plt.title(f"{title}\nN={len(y_true)}, R²={r2:.3f}, MAE={mae:.3f}", fontsize=13)
    plt.xlabel("実測値", fontsize=12)
    plt.ylabel("システム推定値", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def load_json_list(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_results(selected_keys=None):
    """final_results.json と measured_results.json を突合し統計値を返す"""
    preds = load_json_list(FINAL_RESULTS)
    measured = load_json_list(MEASURED_RESULTS)

    # measured を索引化
    # y_true は「食事摂取量割合 (0-1)」。スカラー/リスト/{食種:値} を許容。
    # 突合キー: (target_file, plate_name) 優先、次に (plate_id, plate_name, target)
    def as_list(v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    m_index = {}
    m_file = {}
    for m in measured:
        yt = m.get("y_true")
        tf = m.get("target_file") or m.get("file")
        if tf and isinstance(yt, dict):
            for pn, v in yt.items():
                m_file[(tf, pn)] = v
        elif tf:
            m_file[(tf, m.get("plate_name"))] = yt
        else:
            k = (m.get("plate_id"), m.get("plate_name"), m.get("target"))
            m_index.setdefault(k, []).extend(as_list(yt))
            if k[2] is not None:
                m_index.setdefault((k[0], k[1], None), [])

    rows = []
    nansai_map = {}
    for rec in preds:
        pid, pname = rec["plate_id"], rec["plate_name"]
        if selected_keys and pid not in selected_keys:
            continue
        date, tray = pid.split("_", 1)
        sp = os.path.join(TEST_DATA_DIR, pid, "plate_setting.json")
        if os.path.exists(sp):
            cfg = json.load(open(sp, encoding="utf-8"))
            nansai_map[pid] = {d["plate_name"]: d["nansai"] for d in cfg}
        # 残存率 -> 摂取量割合に変換。複数試行は中央値を採用
        y_pred_all = as_list(rec.get("y_pred"))
        y_med = float(rec.get("y_pred_median")
                      if rec.get("y_pred_median") is not None
                      else np.median(y_pred_all)) if y_pred_all else None
        y_pred = [1 - y_med] if y_med is not None else []
        y_true = None
        fk = (rec.get("target_file"), pname)
        if fk in m_file:
            y_true = as_list(m_file[fk])[:1]
        else:
            for k in [(pid, pname, rec.get("target")), (pid, pname, None)]:
                if k in m_index:
                    y_true = m_index[k][:1]
                    break
        row = {
            "plate_id": pid, "plate_name": pname,
            "target": rec.get("target"), "target_file": rec.get("target_file"),
            "n_trials": rec.get("n_trials", len(y_pred_all)),
            "y_pred": y_pred,
            "y_pred_all": [1 - p for p in y_pred_all],
            "y_true": y_true,
            "nansai": bool(nansai_map.get(pid, {}).get(pname, False)),
            "n": len(y_pred),
        }
        if y_true and y_pred:
            row["r2"], row["mae"] = None, float(abs(y_true[0] - y_pred[0]))
        else:
            row["r2"], row["mae"] = None, None
        rows.append(row)

    # 軟菜/固形で集約して混同行列を生成
    imgs = {}
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for label, flag in [("nansai", True), ("solid", False)]:
        tt, pp = [], []
        for r in rows:
            if r["nansai"] == flag and r["y_true"] and len(r["y_true"]) == len(r["y_pred"]):
                tt.extend(r["y_true"])
                pp.extend(r["y_pred"])
        if len(tt) >= 2:
            r2, mae = calc_score(tt, pp)
            name = "軟菜食" if flag else "固形食"
            path = os.path.join(RESULT_DIR, f"cm_{label}_{ts}.png")
            draw_confusion_matrix(tt, pp, f"{name}推定結果", r2, mae, path)
            imgs[label] = {"path": os.path.relpath(path, BASE_DIR), "r2": r2, "mae": mae, "n": len(tt)}
    return {"rows": rows, "cm": imgs}


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="食事摂取量推定システム")


class DetectReq(BaseModel):
    targets: List[dict]  # [{date, tray}]
    type1s: List[str] = []  # 任意名タイプ1 (空=全て)
    use_cache: bool = True


class SettingReq(BaseModel):
    date: str
    tray: str
    setting: List[dict]


class EstimateReq(BaseModel):
    targets: List[dict]
    type1s: List[str] = []
    use_unet: bool = True
    use_cache: bool = True
    n_trials: int = 1


@app.get("/api/scan")
def api_scan():
    data = scan_data()
    trays = sorted({t for d in data.values() for t in d.keys()})
    names = sorted({t1 for d in data.values() for tr in d.values()
                    for t1 in tr.keys() if t1 not in ("0", "10")})
    return {"dates": [
        {"date": d, "trays": [
            {"tray": t, "type1s": sorted(data[d][t].keys())}
            for t in sorted(data[d])
        ]}
        for d in sorted(data)
    ], "all_trays": trays, "target_names": names}


@app.get("/api/progress")
def api_progress():
    return PROGRESS


@app.get("/api/config")
def api_config():
    return {"plate_names": PLATE_NAMES, "config": CONFIG}


@app.post("/api/detect")
def api_detect(req: DetectReq):
    out = []
    for t in req.targets:
        date, tray = t["date"], t["tray"]
        try:
            with _lock:
                res = detect_reference(date, tray, req.use_cache)
                # 前処理: 連番深度の平均を事前計算
                n_mean = precompute_mean_depth(date, tray)
            key = tray_key(date, tray)
            res["n_mean_depth"] = n_mean
            out.append({
                "key": key, "ok": True, **res,
                "tray_image": f"/api/file?path=work/test_data/{key}/tray_boxes.png",
                "crops": [f"/api/file?path=work/test_data/{key}/plate_{i}.png"
                          for i in range(res["n_dishes"])],
            })
        except HTTPException as e:
            out.append({"key": tray_key(date, tray), "ok": False, "error": e.detail})
        except Exception as e:
            out.append({"key": tray_key(date, tray), "ok": False, "error": str(e)})
    return {"results": out}


@app.get("/api/detect_result")
def api_detect_result(date: str, tray: str):
    key = tray_key(date, tray)
    d = os.path.join(TEST_DATA_DIR, key)
    sp = os.path.join(d, "plate_setting.json")
    if not os.path.exists(sp):
        return {"exists": False}
    setting = json.load(open(sp, encoding="utf-8"))
    return {
        "exists": True, "key": key, "setting": setting,
        "tray_image": f"/api/file?path=work/test_data/{key}/tray_boxes.png",
        "crops": [f"/api/file?path=work/test_data/{key}/plate_{i}.png" for i in range(len(setting))],
    }


@app.post("/api/plate_setting")
def api_plate_setting(req: SettingReq):
    key = tray_key(req.date, req.tray)
    d = os.path.join(TEST_DATA_DIR, key)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "plate_setting.json"), "w", encoding="utf-8") as f:
        json.dump(req.setting, f, indent=4, ensure_ascii=False)
    return {"ok": True}


@app.post("/api/classify")
def api_classify(req: DetectReq):
    out = []
    for t in req.targets:
        date, tray = t["date"], t["tray"]
        try:
            with _lock:
                summary = classify_tray(date, tray,
                                        type1s=req.type1s or None,
                                        use_cache=req.use_cache)
            out.append({"key": tray_key(date, tray), "ok": True, "summary": summary})
        except HTTPException as e:
            out.append({"key": tray_key(date, tray), "ok": False, "error": e.detail})
        except Exception as e:
            out.append({"key": tray_key(date, tray), "ok": False, "error": str(e)})
    return {"results": out}


@app.get("/api/classify_result")
def api_classify_result(date: str, tray: str):
    key = tray_key(date, tray)
    base = os.path.join(PLATE_IMG_DIR, key)
    if not os.path.isdir(base):
        return {"exists": False, "meals": []}
    meals = []
    for meal in sorted(os.listdir(base)):
        mdir = os.path.join(base, meal)
        if not os.path.isdir(mdir):
            continue
        crops = []
        for p in sorted(glob.glob(os.path.join(mdir, "*.json"))):
            meta = json.load(open(p, encoding="utf-8"))
            stem = os.path.basename(p)[:-5]
            crops.append({
                "type1": meta["type1"], "stem": stem,
                "image": f"/api/file?path=work/plate_images/{key}/{meal}/{stem}.png",
            })
        meals.append({"meal": meal, "crops": crops})
    return {"exists": True, "meals": meals}


@app.post("/api/estimate")
def api_estimate(req: EstimateReq):
    all_records = load_json_list(FINAL_RESULTS)
    out = []
    for t in req.targets:
        date, tray = t["date"], t["tray"]
        try:
            with _lock:
                recs = estimate_tray(date, tray, req.use_unet, req.use_cache,
                                     req.n_trials, type1s=req.type1s or None)
            key = tray_key(date, tray)
            # 同一キーの旧レコードを置き換え
            all_records = [r for r in all_records if r.get("plate_id") != key]
            all_records.extend(recs)
            out.append({"key": key, "ok": True, "n_records": len(recs)})
        except HTTPException as e:
            out.append({"key": tray_key(date, tray), "ok": False, "error": e.detail})
        except Exception as e:
            out.append({"key": tray_key(date, tray), "ok": False, "error": str(e)})
    with open(FINAL_RESULTS, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False)
    return {"results": out, "saved": "final_results.json"}


@app.get("/api/results")
def api_results(date: Optional[str] = None, tray: Optional[str] = None):
    keys = None
    if date and tray:
        keys = {tray_key(date, tray)}
    elif date:
        keys = {k for k in
                (tray_key(date, t) for t in scan_data().get(date, {}).keys())}
    res = evaluate_results(keys)
    for label, cm in res["cm"].items():
        cm["url"] = "/api/file?path=" + cm["path"].replace("\\", "/")
    res["has_measured"] = os.path.exists(MEASURED_RESULTS)
    res["has_final"] = os.path.exists(FINAL_RESULTS)
    return res


@app.get("/api/file")
def api_file(path: str):
    """work/ と data/ 配下のファイルのみ配信"""
    ap = os.path.abspath(os.path.join(BASE_DIR, path))
    allowed = [os.path.abspath(WORK_DIR), os.path.abspath(DATA_DIR)]
    if not any(ap.startswith(a + os.sep) or ap == a for a in allowed):
        raise HTTPException(403, "forbidden")
    if not os.path.exists(ap):
        raise HTTPException(404, "not found")
    return FileResponse(ap)


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import socket
    import uvicorn

    HOST, PORT = "0.0.0.0", 8080

    # LAN内IPを取得してアクセス用URLを表示
    def lan_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))  # 実際には送信しない
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            s.close()

    ip = lan_ip()
    print("=" * 50)
    print("  食事摂取量推定システム")
    print(f"  ローカル:   http://127.0.0.1:{PORT}")
    print(f"  LAN内他端末: http://{ip}:{PORT}")
    print("=" * 50, flush=True)

    uvicorn.run(app, host=HOST, port=PORT)
