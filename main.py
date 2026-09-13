from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import json
import shutil
from pydantic import BaseModel, Field
from fastapi import status
from detect_plate import (
    create_baseline_data,
    detect_and_crop_plates,
    classify_plates,
    get_plate_settings,
    update_plate_setting
)

# estimate_amountとresult_drawのインポート（オプション）
try:
    from estimate_amount import estimate_food_amount, get_final_results
    ESTIMATE_AVAILABLE = True
except ImportError:
    ESTIMATE_AVAILABLE = False
    print("警告: estimate_amount.pyが見つかりません。Step 3機能は無効になります。")

app = FastAPI(title="Sumikko Food Analysis Pipeline")

# 静的ファイルの設定
app.mount("/static", StaticFiles(directory="static"), name="static")

# 結果画像配信用のエンドポイント
@app.get("/result_images/{filename}")
async def get_result_image(filename: str):
    file_path = f"{DATA_PATH}/result_analyze/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")


# 食器プレビュー画像配信用のエンドポイント
@app.get("/plate_images/{tray_id}/{filename}")
async def get_plate_image(tray_id: str, filename: str):
    """食器画像を配信"""
    file_path = f"{DATA_PATH}/test_data/{tray_id}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

# データパスの設定
DATA_PATH = "sumikko_analysis"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """ホームページ"""
    # テンプレートファイルを直接読み込む
    template_path = "templates/index.html"
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>テンプレートが見つかりません</h1>", status_code=404)


@app.get("/api/status")
async def get_status():
    """システムステータス確認"""
    status = {
        "data_path_exists": os.path.exists(DATA_PATH),
        "config_exists": os.path.exists(f"{DATA_PATH}/config.json"),
        "models_exist": os.path.exists(f"{DATA_PATH}/models"),
        "captured_images_exist": os.path.exists(f"{DATA_PATH}/captured_images"),
    }
    return status


@app.get("/api/tray-ids")
async def get_tray_ids():
    """実際に存在するトレイIDを取得"""
    captured_images_path = f"{DATA_PATH}/captured_images"
    tray_ids = []
    
    if os.path.exists(captured_images_path):
        for dir_name in os.listdir(captured_images_path):
            if dir_name.startswith("captured_images_"):
                tray_id = dir_name.replace("captured_images_", "")
                if tray_id not in tray_ids:
                    tray_ids.append(tray_id)
    
    return {"tray_ids": sorted(tray_ids)}


@app.get("/api/config")
async def get_config():
    """config.jsonを取得"""
    config_path = f"{DATA_PATH}/config.json"
    if os.path.exists(config_path):
        with open(config_path, encoding='utf-8') as f:
            return json.load(f)
    return {}


# Step 1: 基準データ作成
@app.post("/api/step1/create-baseline")
async def step1_create_baseline(plate_type_mode: str = "both"):
    """基準データ作成"""
    try:
        result = create_baseline_data(data_path=DATA_PATH, plate_type_mode=plate_type_mode)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/step1/status")
async def step1_status():
    """Step 1のステータス確認"""
    # 実際に存在するトレイIDを取得
    captured_images_path = f"{DATA_PATH}/captured_images"
    tray_ids = []
    
    if os.path.exists(captured_images_path):
        for dir_name in os.listdir(captured_images_path):
            if dir_name.startswith("captured_images_"):
                tray_id = dir_name.replace("captured_images_", "")
                if tray_id not in tray_ids:
                    tray_ids.append(tray_id)
    
    status_result = {}
    for tray_id in tray_ids:
        setting_path = f"{DATA_PATH}/test_data/{tray_id}/plate_setting.json"
        status_result[tray_id] = os.path.exists(setting_path)
    return status_result


# Step 2: 食器検出と切り出し
@app.post("/api/step2/detect-plates")
async def step2_detect_plates(request: Request):
    """食器検出と切り出し"""
    try:
        body = await request.json()
        use_yolo_cache = body.get("use_yolo_cache", False)
        tray_ids = body.get("tray_ids") or None
        result = detect_and_crop_plates(data_path=DATA_PATH, use_yolo_cache=use_yolo_cache, tray_ids=tray_ids)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/step2/classify-plates")
async def step2_classify_plates(request: Request):
    """食器分類"""
    try:
        body = await request.json()
        filename_mode = body.get("filename_mode", "auto")  # auto, old, new
        tray_ids = body.get("tray_ids") or None
        plate_types = body.get("plate_types") or None
        result = classify_plates(data_path=DATA_PATH, filename_mode=filename_mode, tray_ids=tray_ids, plate_types=plate_types)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Step 3: 食事量推定
@app.post("/api/step3/estimate-amount")
async def step3_estimate_amount(request: Request):
    """食事量推定"""
    if not ESTIMATE_AVAILABLE:
        return {"status": "error", "message": "estimate_amountモジュールが利用できません。必要なライブラリをインストールしてください。"}
    try:
        body = await request.json()
        iterations = body.get("iterations", 100)
        filename_mode = body.get("filename_mode", "auto")  # auto, old, new
        tray_ids = body.get("tray_ids") or None
        plate_types = body.get("plate_types") or None
        print(f"受け取った反復回数: {iterations}, ファイル名モード: {filename_mode}, トレイID: {tray_ids}, 食器: {plate_types}")
        result = estimate_food_amount(data_path=DATA_PATH, iterations=iterations, filename_mode=filename_mode, tray_ids=tray_ids, plate_types=plate_types)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/step3/results")
async def step3_get_results():
    """食事量推定結果の取得"""
    if not ESTIMATE_AVAILABLE:
        return {"status": "error", "message": "estimate_amountモジュールが利用できません。必要なライブラリをインストールしてください。", "results": []}
    try:
        results = get_final_results(data_path=DATA_PATH)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/step3/clear-cache")
async def step3_clear_cache():
    """キャッシュをクリア"""
    if not ESTIMATE_AVAILABLE:
        return {"status": "error", "message": "estimate_amountモジュールが利用できません。"}
    try:
        from estimate_amount import clear_cache
        result = clear_cache()
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/step3/analysis-data")
async def step3_get_analysis_data():
    """分析用データを取得"""
    if not ESTIMATE_AVAILABLE:
        return {"status": "error", "message": "estimate_amountモジュールが利用できません。"}
    try:
        from estimate_amount import get_analysis_data
        data = get_analysis_data(data_path=DATA_PATH)
        if data is None:
            return {"status": "error", "message": "分析データが見つかりません。先にStep 3を実行してください。"}
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/step3/generate-charts")
async def step3_generate_charts(aggregation_type: str = "all", filter_value: str = "all", chart_type: str = "confusion", tray_id: str = "all", plate_name: str = "all"):
    """分析用グラフを生成"""
    if not ESTIMATE_AVAILABLE:
        return {"status": "error", "message": "estimate_amountモジュールが利用できません。"}
    try:
        from estimate_amount import get_analysis_data, generate_analysis_charts
        data = get_analysis_data(data_path=DATA_PATH)
        if data is None:
            return {"status": "error", "message": "分析データが見つかりません。先にStep 3を実行してください。"}
        
        charts = generate_analysis_charts(data, aggregation_type, filter_value, chart_type, tray_id, plate_name)
        return {"status": "success", "charts": charts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 食器設定管理
@app.get("/api/plate-settings")
async def get_plate_settings_api():
    """全ての食器設定を取得"""
    try:
        settings = get_plate_settings(data_path=DATA_PATH)
        return {"status": "success", "settings": settings}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


class PlateSettingUpdateRequest(BaseModel):
    plate_number: int = Field(..., ge=0, description="皿の番号")
    plate_name: str = Field(..., min_length=1, description="皿の名前")
    nansai: bool = Field(..., description="軟菜フラグ")

@app.post("/api/plate-settings/{tray_id}")
async def update_plate_setting_api(tray_id: str, body: PlateSettingUpdateRequest):
    """食器設定を更新"""
    try:
        settings = update_plate_setting(
            tray_id=tray_id,
            plate_number=body.plate_number,
            plate_name=body.plate_name,
            nansai=body.nansai,
            data_path=DATA_PATH
        )
        return {"status": "success", "settings": settings}
    except Exception as e:
        # 500エラー発生時に、サーバー側のログにエラー詳細を出力する
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="サーバー側でエラーが発生しました。"
        )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """食器設定ページ"""
    template_path = "templates/settings.html"
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>テンプレートが見つかりません</h1>", status_code=404)


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request):
    """パイプライン実行ページ"""
    template_path = "templates/pipeline.html"
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>テンプレートが見つかりません</h1>", status_code=404)


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    """結果表示ページ"""
    template_path = "templates/results.html"
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>テンプレートが見つかりません</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)