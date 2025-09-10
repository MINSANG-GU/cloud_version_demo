import os, glob, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from pdf2image import convert_from_bytes
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from collections import defaultdict
import shutil
import subprocess
import json
import re   # 이 줄을 추가합니다.
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import concurrent.futures
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Dict
import io
from typing import List, Dict, Any, Tuple
import re
from pathlib import Path
from io import BytesIO
from openpyxl.styles import Border, Side, Alignment
import warnings
import io
import os
import sys, os


# ───────────── 설정 & 모델 로딩 ─────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_base_dir, get_ocr_results_folder

BASE_DIR = get_base_dir()
SURYA_RESULTS_FOLDER =get_ocr_results_folder()



Slab_raw_data  = os.path.join(BASE_DIR,"slab", "raw_data")
Slab_table     = os.path.join(BASE_DIR,"slab", "Slab_table")
Slab_anchor    = os.path.join(BASE_DIR,"slab", "Slab_anchor_img")
Slab_anchor_ocr    = os.path.join(BASE_DIR,"slab", "Slab_anchor_ocr")
Slab_elements    = os.path.join(BASE_DIR,"slab", "Slab_elements")
table_ocr_folder  = os.path.join(BASE_DIR,"slab", "Slab_table_ocr")
table_ocr_excel = os.path.join(BASE_DIR,"slab", "Slab_table_ocr_results")




Slab_raw_data_SDD  = os.path.join(BASE_DIR,"slab", "raw_data_SDD")
Slab_column_SDD = os.path.join(BASE_DIR,"slab", "raw_data_column_label_SDD")
Slab_OCR_SDD =os.path.join(BASE_DIR,"slab", "raw_data_OCR_SDD")
Slab_OCR_image_line =os.path.join(BASE_DIR,"slab", "raw_data_line")
Slab_cell = os.path.join(BASE_DIR,"slab", "Slab_cell")
Slab_table_region = os.path.join(BASE_DIR,"slab", "Slab_table_region")
Slab_table_crop_OCR = os.path.join(BASE_DIR,"slab", "Slab_table_crop_OCR")
Slab_same_line = os.path.join(BASE_DIR,"slab", "Slab_same_line")
Slab_table_excel = os.path.join(Slab_table_region,"slab","Slab_excel")
Slab_text_clean = os.path.join(Slab_table_region,"slab","Slab_text_clean")




os.makedirs(Slab_anchor_ocr, exist_ok=True)
os.makedirs(Slab_elements, exist_ok=True)
os.makedirs(table_ocr_folder, exist_ok=True)
os.makedirs(table_ocr_excel, exist_ok=True)
os.makedirs(Slab_raw_data_SDD, exist_ok=True)
os.makedirs(Slab_column_SDD, exist_ok=True)
os.makedirs(Slab_OCR_SDD, exist_ok=True)
os.makedirs(Slab_OCR_image_line, exist_ok=True)
os.makedirs(Slab_cell, exist_ok=True)
os.makedirs(Slab_table_region, exist_ok=True)
os.makedirs(Slab_table_crop_OCR, exist_ok=True)
os.makedirs(Slab_same_line, exist_ok=True)
os.makedirs(Slab_table_excel, exist_ok=True)
os.makedirs(Slab_text_clean, exist_ok=True)





for d in (Slab_raw_data, Slab_table, Slab_anchor):
    os.makedirs(d, exist_ok=True)

CONF_THRESH = 0.3
IOU_THRESH  = 0.3
IMAGE_SIZE  = 1024

model_file = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
yolo_model = YOLOv10(model_file)


# ───────────── 함수 정의부 ─────────────
def compute_iou(b1, b2):
    x1,y1,x2,y2 = max(b1[0],b2[0]), max(b1[1],b2[1]), min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def convert_pdf_to_images(uploaded):
    paths=[]
    for idx, img in enumerate(convert_from_bytes(uploaded.read(), dpi=300),1):
        p=os.path.join(Slab_raw_data, f"page_{idx}.png")
        img.save(p,"PNG"); paths.append(p)
    return paths

def analyze_first_page():
    files = sorted(f for f in os.listdir(Slab_raw_data) if f.endswith(".png"))
    if not files: return None, []
    path = os.path.join(Slab_raw_data, files[0])
    img = Image.open(path).convert("RGB")
    res = yolo_model.predict(path, imgsz=IMAGE_SIZE, conf=CONF_THRESH, iou=IOU_THRESH)[0]
    boxes = [list(map(int,b.xyxy[0].tolist())) for b in res.boxes]
    return img, boxes

def extract_with_offset(anchors, margin_right=150):
    top_idx = min(range(len(anchors)), key=lambda i: anchors[i][1])

    img_paths = sorted(glob.glob(os.path.join(Slab_raw_data, "*.png")))
    progress_bar = st.progress(0)
    status_text  = st.empty()
    shown = 0

    for idx, p in enumerate(img_paths, start=1):
        status_text.text(f"🔄 처리 중: {os.path.basename(p)} ({idx}/{len(img_paths)})")
        page_im = Image.open(p).convert("RGB")
        page_np = np.array(page_im)
        page_gray = cv2.cvtColor(page_np, cv2.COLOR_RGB2GRAY)

        # layout 분석
        res = yolo_model.predict(p, imgsz=IMAGE_SIZE, conf=CONF_THRESH, iou=IOU_THRESH)[0]

        # 테이블 저장: 페이지별 카운터 사용
        page_name = os.path.splitext(os.path.basename(p))[0]
        table_idx = 1
        for b in res.boxes:
            if res.names[int(b.cls.item())] == "table":
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                crop = page_im.crop((x1, y1, x2, y2))
                fn = f"Salb_{page_name}_table_{table_idx}.png"
                crop.save(os.path.join(Slab_table, fn))
                table_idx += 1

        # IoU 매칭 → 중심 이동량 수집
        shifts = []
        page_boxes = [list(map(int,b.xyxy[0].tolist())) for b in res.boxes]
        for anc in anchors:
            best_iou, best_box = 0, None
            for pb in page_boxes:
                iou = compute_iou(anc, pb)
                if iou > best_iou:
                    best_iou, best_box = iou, pb
            if best_box and best_iou >= 0.1:
                cx_a = (anc[0]+anc[2])/2; cy_a = (anc[1]+anc[3])/2
                cx_b = (best_box[0]+best_box[2])/2; cy_b = (best_box[1]+best_box[3])/2
                shifts.append((cx_b-cx_a, cy_b-cy_a))

        # 평균 오프셋 계산
        if shifts:
            dx = sum(s[0] for s in shifts)/len(shifts)
            dy = sum(s[1] for s in shifts)/len(shifts)
        else:
            dx = dy = 0

        # 앵커 크롭 & 저장
        for i, (x1,y1,x2,y2) in enumerate(anchors):
            nx1, ny1 = int(x1+dx), int(y1+dy)
            nx2, ny2 = int(x2+dx), int(y2+dy)
            if i == top_idx:
                nx2 += margin_right
            # clamp
            nx1,ny1 = max(0,nx1), max(0,ny1)
            nx2 = min(page_im.width, nx2); ny2 = min(page_im.height, ny2)
            crop = page_im.crop((nx1,ny1,nx2,ny2))
            fn = f"Slab_{page_name}_box{i}.png"
            crop.save(os.path.join(Slab_anchor, fn))

            if shown < 2:
                st.image(crop, caption=f"예시 Crop {i} on {page_name}", width=200)
                shown += 1

        progress_bar.progress(idx/len(img_paths))

    status_text.text("✅ 모든 페이지 처리 완료")
    st.success(f"테이블: '{Slab_table}', 앵커 크롭: '{Slab_anchor}' 저장 완료")






def apply_surya_ocr_to_anchors():
    anchor_folder       = Slab_anchor
    surya_output_folder = Slab_anchor_ocr

    os.makedirs(surya_output_folder, exist_ok=True)

    # 기존 OCR 결과가 있으면 생략
    existing = [f for f in os.listdir(surya_output_folder) if f.endswith(".json")]
    if existing:
        st.info(f"ℹ️ 이미 {len(existing)}개의 OCR 결과가 존재합니다. Surya OCR 생략.")
        return

    # 입력 이미지 확인
    images = [f for f in os.listdir(anchor_folder) if f.lower().endswith((".png", ".jpg"))]
    if not images:
        st.error("❌ 크롭 이미지가 존재하지 않습니다.")
        return

    progress_bar = st.progress(0)
    status_text  = st.empty()
    st.write("🔍 OCR 실행 중...")

    for idx, img_file in enumerate(images):
        in_path = os.path.join(anchor_folder, img_file)
        # surya_ocr 커맨드 실행
        cmd = ["surya_ocr", in_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        stderr = result.stderr.strip()
        if stderr and all(x not in stderr for x in ["Detecting bboxes", "Recognizing Text"]):
            st.warning(f"⚠️ 오류: {img_file} → {stderr}")

        pct = int((idx + 1) / len(images) * 100)
        progress_bar.progress(pct)
        status_text.write(f"🖼️ OCR 중: {img_file} ({pct}%)")

    progress_bar.empty()
    status_text.write("✅ OCR 완료. 결과 이동 중...")

    # 결과 JSON 이동
    moved = skipped = 0
    for folder in os.listdir(SURYA_RESULTS_FOLDER):
        src = os.path.join(SURYA_RESULTS_FOLDER, folder, "results.json")
        if os.path.exists(src):
            dst = os.path.join(surya_output_folder, f"{folder}.json")
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as e:
                st.error(f"❌ 이동 오류: {folder} → {e}")
        else:
            skipped += 1

    st.success(f"📁 OCR 결과 {moved}개 이동 완료 ({skipped}개 누락)")



#------------------------------------------------------------------------------------

def extract_field_from_json(ocr_data: dict, key: str, y_tol: int = 10) -> str | None:
    records = next(iter(ocr_data.values()))
    page_rec = records[0]
    lines = page_rec["text_lines"]

    # 음수 부호와 숫자·문자 조합을 모두 매칭
    token_pat = r"-?[A-Za-z0-9\.×/²³μμ%]+"

    for line in lines:
        txt = line["text"]
        m = re.search(fr"\b{key}\b", txt, re.IGNORECASE)
        if not m:
            continue

        tail = txt[m.end():]
        # 부호는 남기고, 그 밖의 선행 문자를 제거
        tail = re.sub(r'^[^A-Za-z0-9\-]+', '', tail)

        parts = tail.split()
        if parts:
            first = parts[0]
            # 1) "-" 토큰만 나올 때는 다음 토큰과 합친다
            if first in ("-", "–") and len(parts) > 1:
                tok = "-" + parts[1]
            # 2) 순수 숫자 다음 문자 조합 합치기 (예: "1" + "S5A")
            elif len(parts) > 1 and re.fullmatch(r"-?\d+", first) and re.fullmatch(r"[A-Za-z0-9]+", parts[1]):
                tok = first + parts[1]
            else:
                tok = first

            # 최종 토큰에서 음수·문자 조합 패턴 추출
            vm = re.match(token_pat, tok)
            return vm.group(0) if vm else tok

        # 분리된 박스 케이스에서도 동일 로직 적용
        kb = line["bbox"]
        ky = (kb[1] + kb[3]) / 2
        candidates = []
        for o in lines:
            ob = o["bbox"]
            oy = (ob[1] + ob[3]) / 2
            if abs(oy - ky) <= y_tol and ob[0] > kb[2]:
                text2 = re.sub(r'^[^A-Za-z0-9\-]+', '', o["text"])
                p2 = text2.split()
                if not p2:
                    continue
                first2 = p2[0]
                if first2 in ("-", "–") and len(p2) > 1:
                    tok2 = "-" + p2[1]
                elif len(p2) > 1 and re.fullmatch(r"-?\d+", first2) and re.fullmatch(r"[A-Za-z0-9]+", p2[1]):
                    tok2 = first2 + p2[1]
                else:
                    tok2 = first2

                vm2 = re.match(token_pat, tok2)
                candidates.append((ob[0], vm2.group(0) if vm2 else tok2))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

    return None

def extract_info_for_page(page_num: int, keys: list[str]) -> dict[str, str | None]:
    """
    page_{page_num}_box*.json 들을 슬랙앵커OCR 폴더에서 찾아,
    키워드별 최초 값을 뽑아 dict로 반환
    """
    # 함수 내에서 OCR 결과 폴더 정의
    anchor_ocr = os.path.join(BASE_DIR, "Slab_anchor_ocr")

    results = {k: None for k in keys}
    pattern = os.path.join(anchor_ocr, f"*page_{page_num}_box*.json")
    for jf in sorted(glob.glob(pattern)):
        ocr_data = json.load(open(jf, encoding="utf-8"))
        for k in keys:
            if results[k] is None:
                val = extract_field_from_json(ocr_data, k)
                if val:
                    results[k] = val
        if all(results.values()):
            break
    return results

def extract_all_pages(keys: list[str]) -> dict[int, dict[str, str | None]]:
    """
    앵커 크롭 이미지가 들어 있는 폴더에서 page 번호를 추출,
    모든 페이지에 대해 extract_info_for_page를 실행,
    Slab_elements 폴더에 JSON으로 저장한 뒤 결과 dict 리턴
    """

    elements_folder = os.path.join(BASE_DIR, "Slab_elements")
    os.makedirs(elements_folder, exist_ok=True)

    # page 번호 목록
    files = glob.glob(os.path.join(anchor_folder, "*page_*_box*.png"))
    page_nums = sorted({
        int(re.search(r"page_(\d+)_box", os.path.basename(f)).group(1))
        for f in files
    })

    all_info = {}
    for pnum in page_nums:
        info = extract_info_for_page(pnum, keys)
        outp = os.path.join(elements_folder, f"page_{pnum}_elements.json")
        with open(outp, "w", encoding="utf-8") as fp:
            json.dump(info, fp, ensure_ascii=False, indent=2)
        all_info[pnum] = info

    return all_info











def apply_surya_ocr_to_tables():
    """
    Slab_table 폴더의 모든 PNG/JPG 이미지에 대해 surya_ocr을 실행하고,
    결과 JSON을 Slab_table_ocr 폴더로 이동시킵니다.
    """





    # 1) 이미 OCR 결과가 있으면 스킵
    existing = glob.glob(os.path.join(table_ocr_folder, "*.json"))
    if existing:
        st.info(f"ℹ️ 이미 {len(existing)}개의 테이블 OCR 결과가 존재합니다. 생략합니다.")
        return

    # 2) 입력 이미지 수집
    imgs = sorted(f for f in os.listdir(Slab_table)
                  if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not imgs:
        st.error("❌ Slab_table 폴더에 이미지가 없습니다.")
        return

    progress = st.progress(0)
    status   = st.empty()
    st.write("🔍 테이블 OCR 실행 중...")

    # 3) 이미지별 OCR 호출
    for idx, img_name in enumerate(imgs, start=1):
        in_path = os.path.join(Slab_table, img_name)
        cmd = ["surya_ocr", in_path]
        res = subprocess.run(cmd, capture_output=True, text=True)

        stderr = res.stderr.strip()
        if stderr and all(x not in stderr for x in ("Detecting bboxes", "Recognizing Text")):
            st.warning(f"⚠️ 오류: {img_name} → {stderr}")

        pct = int(idx / len(imgs) * 100)
        progress.progress(pct)
        status.text(f"🖼️ OCR 중: {img_name} ({pct}%)")

    # 4) 결과 이동
    progress.empty()
    status.text("✅ OCR 완료. 결과 이동 중...")

    moved = skipped = 0
    for folder in os.listdir(SURYA_RESULTS_FOLDER):
        src_json = os.path.join(SURYA_RESULTS_FOLDER, folder, "results.json")
        if os.path.exists(src_json):
            dst_json = os.path.join(table_ocr_folder, f"{folder}.json")
            try:
                shutil.move(src_json, dst_json)
                moved += 1
            except Exception as e:
                st.error(f"❌ 이동 오류: {folder} → {e}")
        else:
            skipped += 1

    st.success(f"📁 테이블 OCR 결과 {moved}개 이동 완료 ({skipped}개 누락)")








def parse_ocr_jsons_to_excel(y_tol: int = 10, x_tol: int = 20) -> str:
    """
    OCR JSON 결과들을 wide 형식의 Excel 파일로 변환합니다.
    - 입출력 경로는 함수 내부에서 정의됩니다.
    - y_tol: 행 클러스터링 시 y 중심 차 허용치
    - x_tol: 열 클러스터링 시 x 좌표 차 허용치
    """
    # ▶ 실제 경로를 여기에 지정하세요
    input_folder = table_ocr_folder
    output_file = table_ocr_excel

    # JSON 파일 목록 가져오기
    json_files = sorted(glob.glob(os.path.join(input_folder, "*.json")))
    print(f"[INFO] JSON 파일 수: {len(json_files)} (폴더: {input_folder})")
    if not json_files:
        print(f"[WARN] JSON 파일을 찾을 수 없습니다.")
        return ""

    # 1) 텍스트 조각 수집
    items_all = []
    for jf in json_files:
        fname = os.path.basename(jf)
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] JSON 로드 실패: {fname} - {e}")
            continue

        val = next(iter(data.values()))
        rec = val[0] if isinstance(val, list) else val

        if isinstance(rec.get("text_lines"), list):
            lines = rec["text_lines"]
        elif isinstance(rec.get("results"), list):
            lines = rec["results"]
        else:
            lines = next((v for v in rec.values()
                          if isinstance(v, list) and v and isinstance(v[0], dict)), [])

        for line in lines:
            if not isinstance(line, dict):
                continue
            bbox = line.get("bbox") or line.get("box")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            yc = (y1 + y2) / 2

            text = (line.get("text") or line.get("value")
                    or (line.get("words") if isinstance(line.get("words"), str) else None))
            if not text:
                for v in line.values():
                    if isinstance(v, str) and v.strip():
                        text = v.strip()
                        break
            if not text:
                continue

            items_all.append({"file": fname, "text": text, "x1": x1, "yc": yc})
    print(f"[INFO] 추출된 텍스트 항목 수: {len(items_all)}")
    if not items_all:
        print("[WARN] 변환할 텍스트 항목이 없습니다.")
        return ""

    # 2) y축 클러스터링 → 행 그룹핑
    rows = []
    for itm in items_all:
        placed = False
        for row in rows:
            avg_y = sum(r["yc"] for r in row) / len(row)
            if abs(itm["yc"] - avg_y) <= y_tol:
                row.append(itm)
                placed = True
                break
        if not placed:
            rows.append([itm])
    print(f"[INFO] 그룹핑된 행 수: {len(rows)}")

    # 3) x축 클러스터링 → 열 센터 계산
    xs = sorted(it["x1"] for it in items_all)
    bins = []
    for x in xs:
        for b in bins:
            if abs(b[0] - x) <= x_tol:
                b[0] = (b[0] * b[1] + x) / (b[1] + 1)
                b[1] += 1
                break
        else:
            bins.append([x, 1])
    bins.sort(key=lambda b: b[0])
    col_centers = [b[0] for b in bins]
    print(f"[INFO] 열 센터 수: {len(col_centers)}")

    # 4) long 포맷 → wide 준비
    records = []
    for ridx, row in enumerate(rows, start=1):
        for itm in row:
            diffs = [abs(itm["x1"] - cx) for cx in col_centers]
            cidx = diffs.index(min(diffs)) + 1
            records.append({"file": itm["file"], "row": ridx, "col": cidx, "text": itm["text"]})
    print(f"[INFO] 생성된 레코드 수: {len(records)}")

    # 5) pivot → DataFrame wide
    df_long = pd.DataFrame(records)
    df_wide = df_long.pivot(index=["file", "row"], columns="col", values="text")
    df_wide.columns = [f"col{c}" for c in df_wide.columns]
    df_wide = df_wide.reset_index().fillna("")

    # 6) 엑셀 저장
    if not output_file.lower().endswith(".xlsx"):
        output_file += ".xlsx"
    df_wide.to_excel(output_file, index=False)
    print(f"[DONE] 저장 완료 → {output_file}")
    return output_file

############################################################################################################
#############################################################################################################
#구조계산서 끝 아래부터 구조도면 시작
###########################################################################################################
###########################################################################################################

def convert_pdf_to_images_SDD(uploaded):
    paths=[]
    for idx, img in enumerate(convert_from_bytes(uploaded.read(), dpi=300),1):
        p=os.path.join(Slab_raw_data_SDD, f"page_{idx}.png")
        img.save(p,"PNG"); paths.append(p)
    return paths


def draw_and_save_bounding_boxes_slab_SDD(canvas_key: str = "box_saver") -> None:
    """
    • RAW_DATA_FOLDER에서 이미지를 선택
    • 선택된 이미지를 주어진 MAX 크기에 맞춰 축소해 캔버스 배경으로 띄우고
    • 사각형을 그려 원본 좌표로 역스케일하여 BOX_COORD_FOLDER에 저장
    • 각 바운딩 박스 아래 영역을 잘라내어 같은 폴더에 이미지로 저장
    • crop된 이미지에 맞는 변환된 box 좌표도 각각 json으로 저장
    """
    imgs = [f for f in os.listdir(Slab_raw_data_SDD) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not imgs:
        st.warning(f"이미지 없음: {Slab_raw_data_SDD}")
        return

    selected = st.selectbox("이미지 선택", imgs, key=canvas_key + "_sel")
    img_path = os.path.join(Slab_raw_data_SDD, selected)
    image = Image.open(img_path).convert("RGB")

    MAX_W, MAX_H = 800, 600
    scale = min(1.0, MAX_W / image.width, MAX_H / image.height)
    disp_w, disp_h = int(image.width * scale), int(image.height * scale)
    img_small = image.resize((disp_w, disp_h), Image.LANCZOS)

    offset_x = (MAX_W - disp_w) // 2
    offset_y = (MAX_H - disp_h) // 2

    canvas_res = st_canvas(
        background_image=img_small,
        drawing_mode="rect",
        stroke_width=2,
        stroke_color="#FF0000",
        fill_color="rgba(255,0,0,0.3)",
        update_streamlit=True,
        key=canvas_key,
        width=MAX_W,
        height=MAX_H
    )

    if canvas_res and canvas_res.json_data and canvas_res.json_data.get("objects"):
        boxes = []
        for o in canvas_res.json_data["objects"]:
            if o.get("type") == "rect":
                boxes.append({
                    "left": round((o["left"] - offset_x) / scale, 2),
                    "top": round((o["top"] - offset_y) / scale, 2),
                    "width": round(o["width"] / scale, 2),
                    "height": round(o["height"] / scale, 2)
                })

        os.makedirs(Slab_column_SDD, exist_ok=True)

        # 전체 box 좌표 저장 (원본 좌표 기준)
        out_json = os.path.join(Slab_column_SDD, f"{os.path.splitext(selected)[0]}_boxes.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(boxes, f, ensure_ascii=False, indent=2)

        # 박스별로 잘라내고, 변환된 box 좌표 json 저장
        for idx, b in enumerate(boxes):
            y_start = int(b["top"])
            cropped = image.crop((0, y_start, image.width, image.height))
            crop_name = f"{os.path.splitext(selected)[0]}_{idx+1}.png"
            cropped.save(os.path.join(Slab_column_SDD, crop_name))

            # 변환된 box 좌표 (이 crop이미지 기준에서는 top만 -y_start)
            converted_box = {
                "left": b["left"],
                "top": 0,  # 항상 0, 왜냐면 crop된 이미지에서 이 box의 top은 0임
                "width": b["width"],
                "height": b["height"]
            }
            out_json_crop = os.path.join(Slab_column_SDD, f"{os.path.splitext(selected)[0]}_{idx+1}_boxes.json")
            with open(out_json_crop, "w", encoding="utf-8") as f:
                json.dump([converted_box], f, ensure_ascii=False, indent=2)

        st.success(f"✅ {len(boxes)}개 박스별 crop/좌표 저장 완료: {Slab_column_SDD}")




def apply_surya_ocr_Wall_slab_SDD():
    surya_output_folder = Slab_OCR_SDD

    os.makedirs(surya_output_folder, exist_ok=True)

    # ✅ 기존 OCR 결과가 있으면 생략
    existing_jsons = [f for f in os.listdir(surya_output_folder) if f.endswith(".json")]
    if existing_jsons:
        st.info(f"ℹ️ 이미 {len(existing_jsons)}개의 OCR 결과가 존재합니다. Surya OCR 생략.")
        return

    # ✅ 입력 이미지 확인
    image_files = [f for f in os.listdir(Slab_column_SDD) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        st.error("❌ plain text 이미지가 존재하지 않습니다.")
        return
    st.write("🔍OCR 실행 중...")
    progress_bar = st.progress(0)
    status_text = st.empty()


    for idx, image_file in enumerate(image_files):
        input_path = os.path.join(Slab_column_SDD, image_file)
        command = ["surya_ocr", input_path]
        result = subprocess.run(command, capture_output=True, text=True)

        stderr = result.stderr.strip()
        if stderr and all(x not in stderr for x in ["Detecting bboxes", "Recognizing Text"]):
            st.warning(f"⚠️ 오류: {image_file} → {stderr}")

        progress = int((idx + 1) / len(image_files) * 100)
        progress_bar.progress(progress)
        status_text.write(f"🖼️ OCR 실행 중: {image_file} ({progress}%)")

    progress_bar.empty()
    status_text.write("✅ OCR 완료. 결과 이동 중...")

    # ✅ 결과 이동
    moved, skipped = 0, 0
    for folder_name in os.listdir(SURYA_RESULTS_FOLDER):
        folder_path = os.path.join(SURYA_RESULTS_FOLDER, folder_name)
        if os.path.isdir(folder_path):
            json_file = os.path.join(folder_path, "results.json")
            if os.path.exists(json_file):
                dst_file = os.path.join(surya_output_folder, f"{folder_name}.json")
                try:
                    shutil.move(json_file, dst_file)
                    moved += 1
                except Exception as e:
                    st.error(f"❌ Move Error: {folder_name} → {e}")
            else:
                skipped += 1

    st.success(f"📁 OCR 결과 {moved}개 이동 완료 ({skipped}개는 누락됨)")
#---------------------------------------------------------------------------------------------
def draw_and_save_bounding_boxes_Slab_SDD(canvas_key: str = "box_line_scroll") -> None:
    """
    스크롤 가능한 캔버스 버전 (UI 버튼들 보이도록 개선)
    """
    
    imgs = [f for f in os.listdir(Slab_column_SDD) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not imgs:
        st.warning(f"이미지 없음: {Slab_column_SDD}")
        return

    selected = st.selectbox("이미지 선택", imgs, key=canvas_key + "_sel")
    img_path = os.path.join(Slab_column_SDD, selected)
    image = Image.open(img_path).convert("RGB")

    # 스케일링 옵션 제공
    col1, col2 = st.columns([3, 1])
    with col1:
        scale = st.slider("이미지 크기 조정", 0.3, 2.5, 1.0, 0.1, key=canvas_key + "_scale")
    with col2:
        st.write("**권장 크기:**")
        if max(image.width, image.height) < 500:
            recommended = 1.5
            st.info(f"작은이미지 → ×{recommended}")
        elif max(image.width, image.height) > 1500:
            recommended = 0.6
            st.info(f"큰이미지 → ×{recommended}")
        else:
            recommended = 1.0
            st.info(f"중간이미지 → ×{recommended}")
    
    canvas_width = int(image.width * scale)
    canvas_height = int(image.height * scale)
    
    # 이미지 리사이즈
    img_resized = image.resize((canvas_width, canvas_height), Image.LANCZOS)
    
    st.write(f"📐 **이미지 정보**: 원본 {image.width}×{image.height} → 캔버스 {canvas_width}×{canvas_height} (×{scale:.2f})")
    
    # 캔버스 높이 제한 (UI 버튼들이 보이도록)
    max_canvas_height = 700  # UI 버튼 공간 확보
    
    if canvas_height > max_canvas_height:
        st.warning(f"⚠️ 이미지가 너무 큽니다 ({canvas_height}px). 스케일을 줄이거나 스크롤해서 작업하세요.")
        
        # 방법 1: 컨테이너로 감싸기 (약간의 좌표 오차 있을 수 있음)
        use_container = st.checkbox("스크롤 컨테이너 사용 (UI 버튼 보임)", value=True, key=canvas_key + "_container")
        
        if use_container:
            st.markdown(f"""
            <div style="
                width: 100%; 
                height: {max_canvas_height}px; 
                overflow: auto; 
                border: 2px solid #ddd; 
                border-radius: 8px;
                background: #f8f9fa;
                padding: 5px;
                margin: 10px 0;
            ">
            """, unsafe_allow_html=True)
            
            canvas_res = st_canvas(
                background_image=img_resized,
                drawing_mode="rect",
                stroke_width=max(1, int(2 * scale)),
                stroke_color="#FF0000",
                fill_color="rgba(255,0,0,0.3)",
                update_streamlit=True,
                key=canvas_key + "_scroll",
                width=canvas_width,
                height=canvas_height,
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            # 방법 2: 그냥 원본 크기로 (UI는 잘릴 수 있지만 좌표 정확)
            st.info("💡 페이지를 아래로 스크롤하여 UI 버튼들을 찾으세요.")
            canvas_res = st_canvas(
                background_image=img_resized,
                drawing_mode="rect",
                stroke_width=max(1, int(2 * scale)),
                stroke_color="#FF0000",
                fill_color="rgba(255,0,0,0.3)",
                update_streamlit=True,
                key=canvas_key + "_full",
                width=canvas_width,
                height=canvas_height,
            )
    else:
        # 적당한 크기면 그냥 표시
        canvas_res = st_canvas(
            background_image=img_resized,
            drawing_mode="rect",
            stroke_width=max(1, int(2 * scale)),
            stroke_color="#FF0000",
            fill_color="rgba(255,0,0,0.3)",
            update_streamlit=True,
            key=canvas_key,
            width=canvas_width,
            height=canvas_height,
        )

    # 좌표 변환
    if canvas_res and canvas_res.json_data and canvas_res.json_data.get("objects"):
        boxes = []
        invalid_boxes = 0
        
        for o in canvas_res.json_data["objects"]:
            if o.get("type") == "rect":
                # 스케일 되돌리기
                orig_left = o["left"] / scale
                orig_top = o["top"] / scale
                orig_width = o["width"] / scale
                orig_height = o["height"] / scale
                
                # 경계 체크
                orig_left = max(0, min(orig_left, image.width))
                orig_top = max(0, min(orig_top, image.height))
                orig_right = max(orig_left, min(orig_left + orig_width, image.width))
                orig_bottom = max(orig_top, min(orig_top + orig_height, image.height))
                
                final_width = orig_right - orig_left
                final_height = orig_bottom - orig_top
                
                if final_width > 1 and final_height > 1:
                    boxes.append({
                        "left": round(orig_left, 2),
                        "top": round(orig_top, 2),
                        "width": round(final_width, 2),
                        "height": round(final_height, 2),
                        "right": round(orig_right, 2),
                        "bottom": round(orig_bottom, 2)
                    })
                else:
                    invalid_boxes += 1
        
        if boxes:
            out_file = os.path.join(Slab_same_line, f"{os.path.splitext(selected)[0]}_boxes.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(boxes, f, ensure_ascii=False, indent=2)
            st.success(f"✅ {len(boxes)}개 박스 저장완료!")
            
            # 박스 정보 미리보기 (expander 대신 간단 표시)
            st.info("📦 **저장된 박스 좌표:**")
            for i, box in enumerate(boxes):
                st.write(f"박스 {i+1}: ({box['left']:.1f}, {box['top']:.1f}) ~ ({box['right']:.1f}, {box['bottom']:.1f}) | 크기: {box['width']:.1f}×{box['height']:.1f}")
        else:
            st.info("박스를 그려주세요! 🎯")
        
        if invalid_boxes > 0:
            st.warning(f"⚠️ {invalid_boxes}개 박스 제외됨")

# --------------------------------------------------------------------------------







def detect_and_draw_lines_batch(
    img_dir, save_dir=None,
    min_line_length=80, max_line_gap=10,
    hough_threshold=100, morph_kernel_scale=30,
    max_examples=5, return_imgs=False,
    mode="contour",
    block_size=15, C=-2,
    tol=2
):
    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    out_imgs = []
    for i, img_file in enumerate(img_files):
        if max_examples is not None and i >= max_examples:
            break
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 1. 이진화 (리사이즈 제거)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            ~gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            int(block_size) | 1, C
        )

        # 2. 수평/수직 선 추출
        hs = max(1, binary.shape[1] // morph_kernel_scale)
        vs = max(1, binary.shape[0] // morph_kernel_scale)
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hs, 1))
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vs))
        horizontal = cv2.dilate(cv2.erode(binary, hor_kernel), hor_kernel)
        vertical   = cv2.dilate(cv2.erode(binary, ver_kernel), ver_kernel)

        result_img = img.copy()  # 원본 이미지 사용

        # 3. 선 그리기
        if mode == "hough":
            lines_h = cv2.HoughLinesP(horizontal, 1, np.pi/180, hough_threshold,
                                      minLineLength=min_line_length, maxLineGap=max_line_gap)
            if lines_h is not None:
                for l in lines_h:
                    x1, y1, x2, y2 = l[0]
                    cv2.line(result_img, (x1, y1), (x2, y2), (255,0,0), 2)
            lines_v = cv2.HoughLinesP(vertical, 1, np.pi/180, hough_threshold,
                                      minLineLength=min_line_length, maxLineGap=max_line_gap)
            if lines_v is not None:
                for l in lines_v:
                    x1, y1, x2, y2 = l[0]
                    cv2.line(result_img, (x1, y1), (x2, y2), (0,255,0), 2)
        else:
            contours_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_h:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > min_line_length and h < (binary.shape[0] // 10):
                    cv2.line(result_img, (x, y + h//2), (x + w, y + h//2), (255,0,0), 2)
            contours_v, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_v:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > min_line_length and w < (binary.shape[1] // 10):
                    cv2.line(result_img, (x + w//2, y), (x + w//2, y + h), (0,255,0), 2)

        # 4. 교차점 검출
        kh = np.ones((1, 2*tol+1), np.uint8)
        kv = np.ones((2*tol+1, 1), np.uint8)
        hor_dil = cv2.dilate(horizontal, kh)
        ver_dil = cv2.dilate(vertical, kv)
        mask = cv2.bitwise_and(hor_dil, ver_dil)
        pts = cv2.findNonZero(mask)

        # 5. 시각화 및 리스트 축적
        intersections = []
        if pts is not None:
            for p in pts:
                x_i, y_i = p[0]
                cv2.circle(result_img, (x_i, y_i), 4, (0,0,255), -1)
                intersections.append((int(x_i), int(y_i)))

        # 6. 결과 저장 (원본 해상도 그대로)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.splitext(img_file)[0]

            with open(os.path.join(save_dir, f"{base}_lines.pkl"), "wb") as f:
                pickle.dump({
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "intersections": intersections
                }, f)

            cv2.imwrite(os.path.join(save_dir, f"lines_{img_file}"), result_img)

        if return_imgs:
            out_imgs.append((img_file, result_img, len(intersections)))

    return out_imgs if return_imgs else None


def load_coordinate_files(boxes_file=None, ocr_file=None):
    """
    좌표 파일들을 로드하는 함수
    """
    boxes_data = None
    ocr_data = None
    
    if boxes_file and os.path.exists(boxes_file):
        with open(boxes_file, 'r', encoding='utf-8') as f:
            boxes_data = json.load(f)
    
    if ocr_file and os.path.exists(ocr_file):
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
    
    return boxes_data, ocr_data


def is_point_on_line(point_x, point_y, line_start, line_end, tolerance=10):
    """
    점이 선 위에 있는지 확인하는 함수
    """
    x1, y1 = line_start
    x2, y2 = line_end
    
    # 수직선인 경우
    if abs(x2 - x1) < tolerance:
        return abs(point_x - x1) < tolerance and min(y1, y2) <= point_y <= max(y1, y2)
    
    # 수평선인 경우
    if abs(y2 - y1) < tolerance:
        return abs(point_y - y1) < tolerance and min(x1, x2) <= point_x <= max(x1, x2)
    
    return False


def find_texts_on_lines(horizontal_lines, vertical_lines, ocr_data, tolerance=15):
    """
    선 위에 있는 텍스트들을 찾는 함수
    """
    if not ocr_data:
        return []
    
    # OCR 데이터에서 텍스트 정보 추출
    text_boxes = []
    for page_key, page_data in ocr_data.items():
        for item in page_data:
            if 'text_lines' in item:
                for text_line in item['text_lines']:
                    bbox = text_line['bbox']  # [x1, y1, x2, y2]
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    text_boxes.append({
                        'text': text_line['text'],
                        'bbox': bbox,
                        'center': (center_x, center_y),
                        'confidence': text_line['confidence']
                    })
    
    # 선과 텍스트 매칭
    texts_on_lines = []
    
    # 수평선과 텍스트 매칭
    for line in horizontal_lines:
        for text_box in text_boxes:
            center_x, center_y = text_box['center']
            if is_point_on_line(center_x, center_y, line[0], line[1], tolerance):
                texts_on_lines.append({
                    'text': text_box['text'],
                    'bbox': text_box['bbox'],
                    'line_type': 'horizontal',
                    'line_coords': line,
                    'confidence': text_box['confidence']
                })
    
    # 수직선과 텍스트 매칭
    for line in vertical_lines:
        for text_box in text_boxes:
            center_x, center_y = text_box['center']
            if is_point_on_line(center_x, center_y, line[0], line[1], tolerance):
                texts_on_lines.append({
                    'text': text_box['text'],
                    'bbox': text_box['bbox'],
                    'line_type': 'vertical',
                    'line_coords': line,
                    'confidence': text_box['confidence']
                })
    
    return texts_on_lines


def draw_texts_on_image(img, texts_on_lines, user_boxes=None):
    """
    이미지에 텍스트를 표시하는 함수
    """
    result_img = img.copy()
    
    # OCR 텍스트 표시 (선 위의 텍스트)
    for text_info in texts_on_lines:
        bbox = text_info['bbox']
        text = text_info['text']
        line_type = text_info['line_type']
        
        # 텍스트 박스 그리기
        color = (255, 255, 0) if line_type == 'horizontal' else (0, 255, 255)  # 노란색/시안색
        cv2.rectangle(result_img, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     color, 2)
        
        # 텍스트 표시
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 텍스트 배경 그리기
        cv2.rectangle(result_img,
                     (int(bbox[0]), int(bbox[1]) - text_height - 5),
                     (int(bbox[0]) + text_width, int(bbox[1])),
                     (0, 0, 0), -1)
        
        # 텍스트 그리기
        cv2.putText(result_img, text,
                   (int(bbox[0]), int(bbox[1]) - 5),
                   font, font_scale, (255, 255, 255), thickness)
    
    # 사용자 지정 박스 표시
    if user_boxes:
        for box in user_boxes:
            cv2.rectangle(result_img,
                         (int(box['left']), int(box['top'])),
                         (int(box['right']), int(box['bottom'])),
                         (255, 0, 255), 3)  # 마젠타색
            
            # 박스 라벨
            cv2.putText(result_img, "USER_BOX",
                       (int(box['left']), int(box['top']) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    return result_img
































def load_coordinate_files(boxes_file=None, ocr_file=None):
    """
    좌표 파일들을 로드하는 함수
    """
    boxes_data = None
    ocr_data = None
    
    if boxes_file and os.path.exists(boxes_file):
        with open(boxes_file, 'r', encoding='utf-8') as f:
            boxes_data = json.load(f)
    
    if ocr_file and os.path.exists(ocr_file):
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
    
    return boxes_data, ocr_data


def find_texts_in_same_vertical_line(user_boxes, ocr_data, tolerance=10):
    """
    사용자 박스와 동일한 수직선상(X좌표 범위)에 있는 텍스트들을 찾는 함수
    """
    if not ocr_data or not user_boxes:
        return []
    
    print(f"\n=== find_texts_in_same_vertical_line 디버깅 ===")
    print(f"user_boxes 개수: {len(user_boxes)}")
    
    # OCR 데이터에서 텍스트 정보 추출
    text_boxes = []
    for page_key, page_data in ocr_data.items():
        for item in page_data:
            if 'text_lines' in item:
                for text_line in item['text_lines']:
                    bbox = text_line['bbox']  # [x1, y1, x2, y2]
                    text_boxes.append({
                        'text': text_line['text'],
                        'bbox': bbox,
                        'confidence': text_line['confidence']
                    })
    
    print(f"전체 OCR 텍스트 개수: {len(text_boxes)}")
    print("모든 OCR 텍스트들:")
    for i, text_box in enumerate(text_boxes):
        bbox = text_box['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        print(f"  [{i}] '{text_box['text']}' - bbox:{bbox}, center_x:{center_x:.1f}, conf:{text_box['confidence']:.3f}")
    
    # 사용자 박스와 동일한 수직선상의 텍스트 찾기
    aligned_texts = []
    
    for box_idx, user_box in enumerate(user_boxes):
        user_left = user_box['left']
        user_right = user_box['right']
        print(f"\n--- User Box [{box_idx}]: left={user_left}, right={user_right} ---")
        
        box_aligned_texts = []
        for text_idx, text_box in enumerate(text_boxes):
            text_left = text_box['bbox'][0]  # x1
            text_right = text_box['bbox'][2]  # x2
            text_center_x = (text_left + text_right) / 2
            
            # 텍스트가 사용자 박스의 X 범위와 겹치는지 확인
            condition1 = user_left <= text_center_x <= user_right
            condition2 = text_left <= user_right and text_right >= user_left
            
            if condition1 or condition2:
                aligned_texts.append({
                    'text': text_box['text'],
                    'bbox': text_box['bbox'],
                    'confidence': text_box['confidence'],
                    'aligned_with_box': user_box
                })
                box_aligned_texts.append(text_box['text'])
                print(f"  ✓ 매칭: [{text_idx}] '{text_box['text']}' (center_x:{text_center_x:.1f})")
            else:
                print(f"  ✗ 불일치: [{text_idx}] '{text_box['text']}' (center_x:{text_center_x:.1f})")
        
        print(f"  이 박스와 매칭된 텍스트: {box_aligned_texts}")
    
    print(f"\n총 aligned_texts 개수: {len(aligned_texts)}")
    print("aligned_texts 리스트:")
    for i, text in enumerate(aligned_texts):
        print(f"  [{i}] '{text['text']}' - bbox:{text['bbox']}")
    
    return aligned_texts


def draw_texts_and_boxes_on_image(img, aligned_texts, user_boxes):
    """
    이미지에 사용자 박스와 수직 정렬된 텍스트를 표시하는 함수
    """
    result_img = img.copy()
    
    # 사용자 지정 박스 표시 (마젠타색)
    if user_boxes:
        for box in user_boxes:
            cv2.rectangle(result_img,
                         (int(box['left']), int(box['top'])),
                         (int(box['right']), int(box['bottom'])),
                         (255, 0, 255), 3)  # 마젠타색
            
            # 박스 라벨
            cv2.putText(result_img, "USER_BOX",
                       (int(box['left']), int(box['top']) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # 수직 정렬된 텍스트 표시 (초록색)
    for text_info in aligned_texts:
        bbox = text_info['bbox']
        text = text_info['text']
        
        # 텍스트 박스 그리기 (초록색)
        cv2.rectangle(result_img, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (0, 255, 0), 2)  # 초록색
        
        # 텍스트 표시
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 텍스트 배경 그리기
        cv2.rectangle(result_img,
                     (int(bbox[0]), int(bbox[1]) - text_height - 5),
                     (int(bbox[0]) + text_width, int(bbox[1])),
                     (0, 0, 0), -1)
        
        # 텍스트 그리기 (흰색)
        cv2.putText(result_img, text,
                   (int(bbox[0]), int(bbox[1]) - 5),
                   font, font_scale, (255, 255, 255), thickness)
    
    return result_img


def find_nearby_intersections(text_center, intersections, max_distance=100):
    """
    텍스트 중심 주변의 교차점들만 필터링하는 함수 (디버깅 추가)
    """
    nearby = []
    tx, ty = text_center
    
    for intersection in intersections:
        ix, iy = intersection
        distance = max(abs(ix - tx), abs(iy - ty))  # 맨하탄 거리
        if distance <= max_distance:
            nearby.append(intersection)
    
    return nearby


def is_point_on_line_segment(point, line_start, line_end, tolerance=5):
    """
    점이 선분 위에 있는지 확인하는 함수
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # 수직선인 경우
    if abs(x2 - x1) < tolerance:
        return abs(px - x1) < tolerance and min(y1, y2) <= py <= max(y1, y2)
    
    # 수평선인 경우
    if abs(y2 - y1) < tolerance:
        return abs(py - y1) < tolerance and min(x1, x2) <= px <= max(x1, x2)
    
    return False


def find_line_segment_between_points(point1, point2, horizontal_lines, vertical_lines, tolerance=5):
    """
    두 점 사이에 실제 선분이 존재하는지 확인하는 함수
    """
    x1, y1 = point1
    x2, y2 = point2
    
    # 수평 연결인 경우 (같은 Y좌표)
    if abs(y1 - y2) < tolerance:
        for line_start, line_end in horizontal_lines:
            # 두 점이 모두 이 수평선 위에 있고, 선분이 두 점을 포함하는지 확인
            if (is_point_on_line_segment(point1, line_start, line_end, tolerance) and
                is_point_on_line_segment(point2, line_start, line_end, tolerance)):
                return True
    
    # 수직 연결인 경우 (같은 X좌표)
    if abs(x1 - x2) < tolerance:
        for line_start, line_end in vertical_lines:
            # 두 점이 모두 이 수직선 위에 있고, 선분이 두 점을 포함하는지 확인
            if (is_point_on_line_segment(point1, line_start, line_end, tolerance) and
                is_point_on_line_segment(point2, line_start, line_end, tolerance)):
                return True
    
    return False


def is_valid_rectangle_on_grid(corners, horizontal_lines, vertical_lines, tolerance=5):
    """
    4개 모서리가 실제 격자선으로 연결될 수 있는지 확인하는 함수
    """
    # corners = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)] 순서로 정렬된 사각형 모서리
    if len(corners) != 4:
        return False
    
    # 사각형의 4개 변이 모두 실제 선분으로 연결되는지 확인
    edges = [
        (corners[0], corners[1]),  # 위쪽 가로
        (corners[1], corners[2]),  # 오른쪽 세로
        (corners[2], corners[3]),  # 아래쪽 가로
        (corners[3], corners[0])   # 왼쪽 세로
    ]
    
    for edge_start, edge_end in edges:
        if not find_line_segment_between_points(edge_start, edge_end, horizontal_lines, vertical_lines, tolerance):
            return False
    
    return True


def point_in_rectangle(point, rect_corners):
    """
    점이 사각형 내부에 있는지 확인하는 함수
    """
    px, py = point
    x1, y1, x2, y2 = rect_corners  # (left, top, right, bottom)
    return x1 <= px <= x2 and y1 <= py <= y2


def find_text_cells(intersections, horizontal_lines, vertical_lines, aligned_texts, tolerance=5):
    """
    각 텍스트를 둘러싼 최소 셀을 찾는 함수 (디버깅 추가 버전)
    """
    print(f"\n=== find_text_cells 디버깅 ===")
    print(f"전체 교차점 개수: {len(intersections)}")
    print(f"수평선 개수: {len(horizontal_lines)}")
    print(f"수직선 개수: {len(vertical_lines)}")
    print(f"처리할 aligned_texts 개수: {len(aligned_texts)}")
    
    text_cells = []
    
    for text_idx, text_info in enumerate(aligned_texts):
        print(f"\n--- 텍스트 [{text_idx}]: '{text_info['text']}' 처리 중 ---")
        
        bbox = text_info['bbox']
        text_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        print(f"텍스트 중심: {text_center}")
        
        # 1. 텍스트 주변 교차점만 필터링
        nearby_intersections = find_nearby_intersections(text_center, intersections, max_distance=300)
        print(f"주변 교차점 개수 (300px 내): {len(nearby_intersections)}")
        
        if len(nearby_intersections) < 4:
            print(f"  ⚠️ 주변 교차점이 부족함 (최소 4개 필요, 현재 {len(nearby_intersections)}개)")
            continue
        
        print(f"주변 교차점들: {nearby_intersections[:10]}...")  # 처음 10개만 출력
        
        min_area = float('inf')
        best_cell = None
        valid_rectangles_found = 0
        
        # 2. 주변 교차점들로만 사각형 생성 시도
        for i, p1 in enumerate(nearby_intersections):
            for j, p2 in enumerate(nearby_intersections):
                if i >= j:
                    continue
                for k, p3 in enumerate(nearby_intersections):
                    if k <= j:
                        continue
                    for l, p4 in enumerate(nearby_intersections):
                        if l <= k:
                            continue
                        
                        # 4개 점으로 사각형 만들기 시도
                        points = [p1, p2, p3, p4]
                        
                        # X, Y 좌표로 정렬해서 사각형 모서리 찾기
                        x_coords = sorted(set(p[0] for p in points))
                        y_coords = sorted(set(p[1] for p in points))
                        
                        # 정확히 2개의 서로 다른 X좌표와 2개의 서로 다른 Y좌표가 있어야 함
                        if len(x_coords) == 2 and len(y_coords) == 2:
                            left, right = x_coords
                            top, bottom = y_coords
                            
                            # 사각형 모서리 정의
                            corners = [
                                (left, top),    # 좌상
                                (right, top),   # 우상  
                                (right, bottom), # 우하
                                (left, bottom)   # 좌하
                            ]
                            
                            # 4개 모서리가 모두 주변 교차점들 중에 있는지 확인
                            if all(corner in nearby_intersections for corner in corners):
                                # 면적 조기 체크
                                area = (right - left) * (bottom - top)
                                if area >= min_area:
                                    continue
                                
                                # 실제 격자선으로 연결되는지 확인
                                if is_valid_rectangle_on_grid(corners, horizontal_lines, vertical_lines, tolerance):
                                    valid_rectangles_found += 1
                                    
                                    # 텍스트 중심이 사각형 내부에 있는지 확인
                                    rect_bounds = (left, top, right, bottom)
                                    if point_in_rectangle(text_center, rect_bounds):
                                        min_area = area
                                        best_cell = {
                                            'text_info': text_info,
                                            'corners': corners,
                                            'bounds': rect_bounds,
                                            'area': area
                                        }
                                        print(f"  ✓ 새로운 최적 셀 발견: bounds={rect_bounds}, area={area:.1f}")
        
        print(f"  검사한 유효한 사각형 개수: {valid_rectangles_found}")
        
        if best_cell:
            text_cells.append(best_cell)
            print(f"  ✅ 최종 셀 확정: {best_cell['bounds']}")
        else:
            print(f"  ❌ 유효한 셀을 찾지 못함")
    
    print(f"\n총 찾은 text_cells 개수: {len(text_cells)}")
    return text_cells



def draw_text_cells_on_image(img, text_cells):
    """
    찾은 텍스트 셀들을 이미지에 그리는 함수
    """
    result_img = img.copy()
    
    for cell in text_cells:
        corners = cell['corners']
        left, top, right, bottom = cell['bounds']
        
        # 셀 사각형 그리기 (주황색)
        cv2.rectangle(result_img, 
                     (int(left), int(top)), 
                     (int(right), int(bottom)), 
                     (0, 165, 255), 3)  # 주황색
        
        # 셀 라벨
        cv2.putText(result_img, "CELL",
                   (int(left), int(top) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    return result_img


# 기존 함수에 추가할 매개변수와 로직
def detect_and_draw_lines_batch(
    img_dir, save_dir=None,
    min_line_length=80, max_line_gap=10,
    hough_threshold=100, morph_kernel_scale=30,
    max_examples=5, return_imgs=False,
    mode="contour",
    block_size=15, C=-2,
    tol=2,
    boxes_coord_dir=None,
    ocr_coord_dir=None, 
    text_tolerance=15
):
    """
    선 검출 + 수직 정렬 텍스트 표시 + 셀 찾기 + 디버깅 출력
    """
    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    out_imgs = []

    for i, img_file in enumerate(img_files):
        if max_examples is not None and i >= max_examples:
            break

        print(f"\n{'='*50}")
        print(f"처리 중인 이미지: {img_file}")
        print(f"{'='*50}")

        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 1. 이진화
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            ~gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            int(block_size) | 1, C
        )

        # 2. 수평/수직 선 추출
        hs = max(1, binary.shape[1] // morph_kernel_scale)
        vs = max(1, binary.shape[0] // morph_kernel_scale)
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hs, 1))
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vs))
        horizontal = cv2.dilate(cv2.erode(binary, hor_kernel), hor_kernel)
        vertical   = cv2.dilate(cv2.erode(binary, ver_kernel), ver_kernel)

        result_img = img.copy()
        horizontal_lines, vertical_lines = [], []

        # 3. 선 그리기
        if mode == "hough":
            lines_h = cv2.HoughLinesP(horizontal, 1, np.pi/180,
                                      hough_threshold,
                                      minLineLength=min_line_length,
                                      maxLineGap=max_line_gap)
            if lines_h is not None:
                for l in lines_h:
                    x1, y1, x2, y2 = l[0]
                    cv2.line(result_img, (x1, y1), (x2, y2), (255,0,0), 2)
                    horizontal_lines.append(((x1, y1), (x2, y2)))
            lines_v = cv2.HoughLinesP(vertical, 1, np.pi/180,
                                      hough_threshold,
                                      minLineLength=min_line_length,
                                      maxLineGap=max_line_gap)
            if lines_v is not None:
                for l in lines_v:
                    x1, y1, x2, y2 = l[0]
                    cv2.line(result_img, (x1, y1), (x2, y2), (0,255,0), 2)
                    vertical_lines.append(((x1, y1), (x2, y2)))
        else:
            cnts_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts_h:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > min_line_length and h < (binary.shape[0] // 10):
                    cv2.line(result_img, (x, y+h//2), (x+w, y+h//2), (255,0,0), 2)
                    horizontal_lines.append(((x, y+h//2), (x+w, y+h//2)))
            cnts_v, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts_v:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > min_line_length and w < (binary.shape[1] // 10):
                    cv2.line(result_img, (x+w//2, y), (x+w//2, y+h), (0,255,0), 2)
                    vertical_lines.append(((x+w//2, y), (x+w//2, y+h)))

        # 4. 교차점 검출
        kh = np.ones((1, 2*tol+1), np.uint8)
        kv = np.ones((2*tol+1, 1), np.uint8)
        hor_dil = cv2.dilate(horizontal, kh)
        ver_dil = cv2.dilate(vertical, kv)
        mask = cv2.bitwise_and(hor_dil, ver_dil)
        pts = cv2.findNonZero(mask)

        intersections = []
        if pts is not None:
            for p in pts:
                x_i, y_i = p[0]
                cv2.circle(result_img, (x_i, y_i), 4, (0,0,255), -1)
                intersections.append((int(x_i), int(y_i)))

        # 5. OCR 좌표 로드 및 텍스트 셀 찾기
        # 5. OCR 좌표 로드 및 텍스트 셀 찾기
        aligned_texts, text_cells = [], []
        if boxes_coord_dir or ocr_coord_dir:
            base_name = os.path.splitext(img_file)[0]
            boxes_file = os.path.join(boxes_coord_dir, f"{base_name}_boxes.json") if boxes_coord_dir else None
            ocr_file   = os.path.join(ocr_coord_dir,   f"{base_name}.json") if ocr_coord_dir else None
            
            print(f"\n좌표 파일 로드:")
            print(f"  boxes_file: {boxes_file}")
            print(f"  ocr_file: {ocr_file}")
            
            boxes_data, ocr_data = load_coordinate_files(boxes_file, ocr_file)
            
            print(f"  boxes_data 로드됨: {boxes_data is not None}")
            print(f"  ocr_data 로드됨: {ocr_data is not None}")
            
            if boxes_data and ocr_data:
                aligned_texts = find_texts_in_same_vertical_line(
                    boxes_data, ocr_data, text_tolerance)
                result_img = draw_texts_and_boxes_on_image(
                    result_img, aligned_texts, boxes_data)
                if aligned_texts and intersections:
                    text_cells = find_text_cells(
                        intersections, horizontal_lines,
                        vertical_lines, aligned_texts,
                        text_tolerance)
                    result_img = draw_text_cells_on_image(result_img, text_cells)

        # 6. 행 영역 인식 및 OCR 텍스트 저장
        regions = []
        def has_horz_segment(x1, x2, y):
            for (sx, sy), (ex, ey) in horizontal_lines:
                if abs(sy-y) < tol and sx <= x1 and ex >= x2:
                    return True
            return False

        for cell in text_cells:
            # 1) 셀 내부 텍스트를 라벨로 추출
            label = cell['text_info']['text']            # 예: "cs1" 또는 "s2a"
            safe_label = re.sub(r'[\\/*?:"<>|]', '_', label)  # 파일명에 안전하게 변환

            # 2) 기존 x_end 계산 로직 유지
            x_start, y_top, _, y_bottom = cell['bounds']
            xs = sorted(set([x for x, y in intersections if x >= x_start]))
            x_end = x_start
            for i in range(len(xs)-1):
                x1, x2 = xs[i], xs[i+1]
                if has_horz_segment(x1, x2, y_top) and has_horz_segment(x1, x2, y_bottom):
                    x_end = x2
                else:
                    break

            # 3) 셀 크롭 후 라벨 기반으로 저장
            crop = img[y_top:y_bottom, x_start:x_end]
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                crop_path = os.path.join(save_dir, f"{safe_label}.png")
                cv2.imwrite(crop_path, crop)



            # 4) regions에 text 기반 라벨로 추가
            region = {
                'cell_label': label,                # 'cell_index' 대신 실제 텍스트
                'bounds': (x_start, y_top, x_end, y_bottom),
                'texts': []
            }
            # OCR 텍스트 필터링 로직 그대로
            if boxes_coord_dir and ocr_coord_dir:
                for page in ocr_data.values():
                    for item in page:
                        for line in item.get('text_lines', []):
                            bx1, by1, bx2, by2 = line['bbox']
                            cx, cy = (bx1+bx2)/2, (by1+by2)/2
                            if x_start <= cx <= x_end and y_top <= cy <= y_bottom:
                                region['texts'].append({
                                    'text': line['text'],
                                    'bbox': line['bbox'],
                                    'confidence': line.get('confidence')
                                })
            regions.append(region)

        # 7. 결과 저장
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.splitext(img_file)[0]
            # 이미지 저장
            cv2.imwrite(os.path.join(save_dir, f"lines_text_{img_file}"), result_img)
            # 교차점 등 데이터 피클
            with open(os.path.join(save_dir, f"{base}_lines.pkl"), "wb") as f:
                pickle.dump({
                    'horizontal': horizontal,
                    'vertical': vertical,
                    'intersections': intersections,
                    'horizontal_lines': horizontal_lines,
                    'vertical_lines': vertical_lines,
                    'aligned_text_count': len(aligned_texts),
                    'text_cells': text_cells,
                    'regions': regions
                }, f)
            # 행 영역 텍스트 JSON 저장
            with open(os.path.join(save_dir, f"{base}_regions.json"), 'w', encoding='utf-8') as jf:
                json.dump(regions, jf, ensure_ascii=False, indent=2)

        if return_imgs:
            out_imgs.append((img_file, result_img, len(intersections), len(aligned_texts), len(text_cells)))

    return out_imgs if return_imgs else None
#---------------------------------------------------------------------------------
def clean_text(raw: str) -> str:
    nospace = ''.join(raw.split())
    return re.sub(r'[^0-9A-Za-z@\-\~\(\),]', '', nospace)

def merge_close_texts(texts: list, merge_tol: int = 10) -> list:
    if not texts:
        return []
    items = []
    for t in texts:
        x1, y1, x2, y2 = t['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        items.append((t, cx, cy))
    items.sort(key=lambda it: it[1])
    groups, current = [], [items[0]]
    for txt, cx, cy in items[1:]:
        prev_cx, prev_cy = current[-1][1], current[-1][2]
        if abs(cx - prev_cx) <= merge_tol and abs(cy - prev_cy) <= merge_tol:
            current.append((txt, cx, cy))
        else:
            groups.append(current); current = [(txt, cx, cy)]
    groups.append(current)
    merged = []
    for grp in groups:
        if len(grp) == 1:
            merged.append(grp[0][0])
        else:
            texts = [it[0]['text'] for it in grp]
            bboxes = [it[0]['bbox'] for it in grp]
            confs  = [it[0].get('confidence', 0) for it in grp]
            new_text = "".join(texts)
            x1s = [b[0] for b in bboxes]; y1s = [b[1] for b in bboxes]
            x2s = [b[2] for b in bboxes]; y2s = [b[3] for b in bboxes]
            new_bbox = [min(x1s), min(y1s), max(x2s), max(y2s)]
            new_conf = max(confs)
            merged.append({
                'text': new_text,
                'bbox': new_bbox,
                'confidence': new_conf
            })
    return merged

def filter_regions_text(regions: list, merge_tol: int = 10) -> list:
    for region in regions:
        cleaned = []
        for txtinfo in region.get('texts', []):
            txt = clean_text(txtinfo['text'])
            if txt:
                txtinfo['text'] = txt
                cleaned.append(txtinfo)
        region['texts'] = merge_close_texts(cleaned, merge_tol)
    return regions

#----------------------------------------------------------------------------------


#---------------------------------------------------------------------





# def extract_texts_by_cell():
#     """
#     실제 격자선과 교차점을 이용해 정확한 셀별로 텍스트를 추출하는 함수
#     """
#     pkl_dir = Slab_table_region
    
#     # 피클 파일들 찾기
#     pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('_lines.pkl')]
    
#     if not pkl_files:
#         st.error("피클 파일을 찾을 수 없습니다!")
#         return
    
#     st.write(f"발견된 피클 파일: {len(pkl_files)}개")
    
#     for pkl_file in pkl_files:
#         st.write(f"처리 중: {pkl_file}")
        
#         try:
#             # 피클 파일 로드
#             with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
#                 data = pickle.load(f)
            
#             # 실제 격자 셀 생성
#             grid_cells = create_grid_cells(data['horizontal_lines'], data['vertical_lines'], data['intersections'])
            
#             # 각 격자 셀에 텍스트 할당
#             cell_results = []
#             merge_count = 0
            
#             for i, cell in enumerate(grid_cells):
#                 cell_texts = assign_texts_to_grid_cell(cell, data['regions'])
                
#                 # 같은 셀 내의 여러 텍스트들을 위치 기준으로 병합
#                 if len(cell_texts) > 1:
#                     merge_count += 1
#                     st.write(f"셀 {i}: {len(cell_texts)}개 텍스트 병합 중...")
#                     for text in cell_texts:
#                         st.write(f"  - '{text['text']}' at {text['bbox']}")
                
#                 merged_texts = merge_texts_in_cell(cell_texts)
                
#                 if len(cell_texts) > 1:
#                     st.write(f"병합 결과: '{merged_texts[0]['text']}'")
#                     st.write("---")
                
#                 cell_results.append({
#                     'cell_index': i,
#                     'cell_bounds': cell['bounds'],
#                     'cell_area': cell['area'],
#                     'row': cell['row'],
#                     'col': cell['col'],
#                     'extracted_texts': merged_texts
#                 })
            
#             st.write(f"총 {merge_count}개 셀에서 텍스트 병합 수행됨")
            
#             # 결과 저장
#             base_name = pkl_file.replace('_lines.pkl', '')
#             output_file = os.path.join(Slab_text_clean, f"{base_name}_grid_cells.json")
            
#             with open(output_file, 'w', encoding='utf-8') as f:
#                 json.dump(cell_results, f, ensure_ascii=False, indent=2)
            
#             st.success(f"격자 셀별 텍스트 추출 완료: {output_file}")
            
#             # 결과 미리보기
#             st.write(f"### {base_name} 격자 셀 결과 미리보기")
#             st.write(f"총 {len(cell_results)}개 셀 생성됨")
            
#             # 첫 번째 행만 표시 (최대 6개)
#             first_row_cells = [cell for cell in cell_results if cell['row'] == 0][:6]
            
#             if first_row_cells:
#                 cols = st.columns(len(first_row_cells))
                
#                 for idx, cell in enumerate(first_row_cells):
#                     with cols[idx]:
#                         st.write(f"**셀({cell['row']},{cell['col']})**")
#                         if cell['extracted_texts']:
#                             for text in cell['extracted_texts']:
#                                 st.write(f"`{text['text']}`")
#                         else:
#                             st.write("*빈 셀*")
#             else:
#                 st.write("첫 번째 행 셀을 찾을 수 없습니다.")
                
#             # 전체 결과 요약
#             total_texts = sum(len(cell['extracted_texts']) for cell in cell_results)
#             st.write(f"전체 추출된 텍스트: {total_texts}개")
                    
#         except Exception as e:
#             st.error(f"오류 발생 ({pkl_file}): {str(e)}")


def create_grid_cells(horizontal_lines, vertical_lines, intersections):
    """
    수평선과 수직선의 교차점을 이용해 실제 격자 셀들을 생성
    """
    # 수평선과 수직선 좌표 추출 및 정렬
    h_coords = sorted(set([line[0][1] for line in horizontal_lines] + [line[1][1] for line in horizontal_lines]))
    v_coords = sorted(set([line[0][0] for line in vertical_lines] + [line[1][0] for line in vertical_lines]))
    
    grid_cells = []
    
    # 각 격자 셀 생성
    for row in range(len(h_coords) - 1):
        for col in range(len(v_coords) - 1):
            y1, y2 = h_coords[row], h_coords[row + 1]
            x1, x2 = v_coords[col], v_coords[col + 1]
            
            # 셀이 실제로 존재하는지 확인 (4개 모서리 교차점이 모두 있는지)
            corners_needed = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            corners_found = []
            
            for corner in corners_needed:
                # 교차점 근처에 있는지 확인 (tolerance=5)
                found = any(abs(pt[0] - corner[0]) <= 5 and abs(pt[1] - corner[1]) <= 5 
                           for pt in intersections)
                if found:
                    corners_found.append(corner)
            
            # 4개 모서리가 모두 있으면 유효한 셀
            if len(corners_found) >= 3:  # 완전하지 않아도 3개 이상이면 셀로 인정
                cell = {
                    'bounds': (x1, y1, x2, y2),
                    'area': (x2 - x1) * (y2 - y1),
                    'row': row,
                    'col': col,
                    'corners': corners_found
                }
                grid_cells.append(cell)
    
    return grid_cells


def merge_texts_in_cell(cell_texts):
    """
    같은 셀 내의 여러 텍스트들을 위치 기준으로 병합
    """
    if len(cell_texts) <= 1:
        return cell_texts
    
    # X 좌표 기준으로 정렬 (왼쪽부터 오른쪽 순서)
    sorted_texts = sorted(cell_texts, key=lambda t: t['bbox'][0])
    
    # 텍스트 합치기 (공백으로 구분)
    merged_text = " ".join([t['text'].strip() for t in sorted_texts if t['text'].strip()])
    
    # 합쳐진 bbox 계산 (모든 텍스트를 포함하는 최소 사각형)
    all_bboxes = [t['bbox'] for t in sorted_texts]
    merged_bbox = [
        min([bb[0] for bb in all_bboxes]),  # x1 (가장 왼쪽)
        min([bb[1] for bb in all_bboxes]),  # y1 (가장 위쪽)
        max([bb[2] for bb in all_bboxes]),  # x2 (가장 오른쪽)
        max([bb[3] for bb in all_bboxes])   # y2 (가장 아래쪽)
    ]
    
    # 평균 신뢰도 계산
    avg_confidence = sum([t['confidence'] for t in sorted_texts]) / len(sorted_texts)
    
    # 병합된 하나의 텍스트만 반환
    return [{
        'text': merged_text,
        'bbox': merged_bbox,
        'confidence': avg_confidence,
        'split_type': 'merged',
        'original_count': len(sorted_texts)
    }]


def assign_texts_to_grid_cell(grid_cell, regions):
    """
    특정 격자 셀에 해당하는 텍스트들을 할당
    """
    x1, y1, x2, y2 = grid_cell['bounds']
    cell_texts = []
    
    # 모든 region의 텍스트들을 검사
    for region in regions:
        for text_item in region['texts']:
            text_bbox = text_item['bbox']  # [x1, y1, x2, y2]
            text_center_x = (text_bbox[0] + text_bbox[2]) / 2
            text_center_y = (text_bbox[1] + text_bbox[3]) / 2
            
            # 텍스트 중심점이 격자 셀 안에 있는지 확인
            if (x1 <= text_center_x <= x2 and y1 <= text_center_y <= y2):
                # 긴 텍스트인 경우 셀 경계에 맞춰 분할
                if is_long_text(text_item['text']) and grid_cell['col'] > 0:
                    split_text = split_long_text_by_position(
                        text_item, grid_cell, regions
                    )
                    if split_text:
                        cell_texts.append(split_text)
                else:
                    cell_texts.append({
                        'text': text_item['text'].strip(),
                        'bbox': text_bbox,
                        'confidence': text_item['confidence'],
                        'split_type': 'original'
                    })
    
    return cell_texts


def is_long_text(text):
    """
    긴 텍스트인지 판단 (여러 개의 값이 연결된 형태)
    """
    # HD, @, 숫자가 반복되는 패턴이면 긴 텍스트로 판단
    hd_count = text.count('HD')
    at_count = text.count('@')
    return hd_count > 1 or at_count > 1 or len(text) > 20


def split_long_text_by_position(text_item, current_cell, regions):
    """
    긴 텍스트를 현재 셀 위치에 맞는 부분만 추출
    """
    text = text_item['text']
    text_bbox = text_item['bbox']
    
    # 패턴 기반으로 텍스트 분할
    import re
    
    # "HD 숫자 @ 숫자" 패턴으로 분할
    pattern = r'(HD\s*\d+\s*@\s*\d+|HD\s*\d+|\d+\s*@\s*\d+|\|\s*HD\s*\||\|)'
    parts = re.findall(pattern, text)
    
    if not parts:
        # 패턴이 없으면 공백으로 분할
        parts = text.split()
    
    if len(parts) <= 1:
        return {
            'text': text.strip(),
            'bbox': text_bbox,
            'confidence': text_item['confidence'],
            'split_type': 'single'
        }
    
    # 현재 셀의 컬럼 위치에 맞는 부분 선택
    col_index = current_cell['col']
    
    # 첫 번째 컬럼(NAME)이면 첫 번째 부분
    if col_index <= 2:
        selected_part = parts[0] if parts else text
    else:
        # 나머지 컬럼들은 순서대로 배정
        part_index = min(col_index - 3, len(parts) - 1)
        selected_part = parts[part_index] if part_index >= 0 else ""
    
    if selected_part.strip():
        return {
            'text': selected_part.strip(),
            'bbox': text_bbox,
            'confidence': text_item['confidence'],
            'split_type': 'pattern_split'
        }
    
    return None














#--------------------------------------------------------------------------------

def get_row_bounds(cells_in_row):
    """
    행의 전체 경계 계산
    """
    if not cells_in_row:
        return [0, 0, 0, 0]
    
    x1 = min(cell['cell_bounds'][0] for cell in cells_in_row)
    y1 = min(cell['cell_bounds'][1] for cell in cells_in_row)
    x2 = max(cell['cell_bounds'][2] for cell in cells_in_row)
    y2 = max(cell['cell_bounds'][3] for cell in cells_in_row)
    
    return [x1, y1, x2, y2]


def add_row_labels_to_cells_fixed(cell_results, regions):
    """
    PICKLE 파일의 cell_label을 Y좌표 기준으로 직접 매핑하는 수정된 함수
    """
    # 행별로 셀들을 그룹핑
    rows_dict = {}
    for cell in cell_results:
        row_idx = cell['row']
        if row_idx not in rows_dict:
            rows_dict[row_idx] = []
        rows_dict[row_idx].append(cell)
    
    # ★ PICKLE의 regions에서 cell_label을 Y좌표 기준으로 직접 추출
    region_labels = []
    for region in regions:
        if 'cell_label' in region and region['cell_label']:
            y_coord = region['bounds'][1]  # Y 좌표
            label = region['cell_label'].strip()
            region_labels.append({
                'y_coord': y_coord,
                'label': label,
                'bounds': region['bounds']
            })
    
    # Y좌표 기준으로 정렬
    region_labels.sort(key=lambda x: x['y_coord'])
    
    # 각 행의 라벨 매핑
    row_labels = []
    
    for row_idx, cells_in_row in rows_dict.items():
        if not cells_in_row:
            continue
            
        # 현재 행의 Y좌표 범위 계산
        row_y_min = min(cell['cell_bounds'][1] for cell in cells_in_row)
        row_y_max = max(cell['cell_bounds'][3] for cell in cells_in_row)
        row_y_center = (row_y_min + row_y_max) / 2
        
        # ★ 가장 가까운 region의 label 찾기
        best_label = "Unknown"
        min_distance = float('inf')
        
        for region_info in region_labels:
            region_y = region_info['y_coord']
            distance = abs(region_y - row_y_center)
            
            # Y좌표가 50px 이내이고 가장 가까운 것 선택
            if distance < 50 and distance < min_distance:
                min_distance = distance
                best_label = region_info['label']
        
        row_labels.append({
            'row_index': row_idx,
            'label': best_label,  # ★ 직접 매핑된 라벨
            'cell_count': len(cells_in_row),
            'row_bounds': get_row_bounds(cells_in_row),
            'y_center': row_y_center  # 디버깅용
        })
    
    # 라벨 매핑 결과 검증
    preserved_count = sum(1 for row in row_labels if row['label'] != "Unknown")
    total_regions = len([r for r in regions if r.get('cell_label', '').strip()])
    
    st.write(f"라벨 보존 결과: {preserved_count}/{total_regions} 개 보존됨")
    if preserved_count < total_regions:
        st.write(f"❌ {total_regions - preserved_count}개 라벨 손실!")
        lost_labels = []
        for region in regions:
            label = region.get('cell_label', '').strip()
            if label and label not in [row['label'] for row in row_labels]:
                lost_labels.append(label)
        st.write(f"손실된 라벨들: {lost_labels}")
    
    # 행 라벨 정보와 셀 정보를 함께 반환
    return {
        'row_labels': sorted(row_labels, key=lambda x: x['row_index']),
        'cells': cell_results,
        'metadata': {
            'total_rows': len(rows_dict),
            'total_cells': len(cell_results),
            'preserved_labels': preserved_count,
            'total_original_labels': total_regions
        }
    }


def extract_texts_by_cell_with_proper_labels():
    """
    라벨 보존이 제대로 되는 수정된 메인 함수
    """
    pkl_dir = Slab_table_region
    
    # 피클 파일들 찾기
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('_lines.pkl')]
    
    if not pkl_files:
        st.error("피클 파일을 찾을 수 없습니다!")
        return
    
    st.write(f"발견된 피클 파일: {len(pkl_files)}개")
    
    for pkl_file in pkl_files:
        st.write(f"처리 중: {pkl_file}")
        
        try:
            # 피클 파일 로드
            with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
                data = pickle.load(f)
            
            # ★ 원본 라벨 확인
            st.write("📋 원본 라벨들:")
            original_labels = []
            for i, region in enumerate(data['regions']):
                label = region.get('cell_label', '').strip()
                if label:
                    original_labels.append(label)
                    st.write(f"  {i}: '{label}' (Y: {region['bounds'][1]})")
            
            # 실제 격자 셀 생성
            grid_cells = create_grid_cells(data['horizontal_lines'], data['vertical_lines'], data['intersections'])
            
            # 각 격자 셀에 텍스트 할당
            cell_results = []
            merge_count = 0
            
            for i, cell in enumerate(grid_cells):
                cell_texts = assign_texts_to_grid_cell(cell, data['regions'])
                
                # 같은 셀 내의 여러 텍스트들을 위치 기준으로 병합
                if len(cell_texts) > 1:
                    merge_count += 1
                
                merged_texts = merge_texts_in_cell(cell_texts)
                
                cell_results.append({
                    'cell_index': i,
                    'cell_bounds': cell['bounds'],
                    'cell_area': cell['area'],
                    'row': cell['row'],
                    'col': cell['col'],
                    'extracted_texts': merged_texts
                })
            
            st.write(f"총 {merge_count}개 셀에서 텍스트 병합 수행됨")
            
            # ★ 수정된 라벨 매핑 함수 사용
            final_results = add_row_labels_to_cells_fixed(cell_results, data['regions'])
            
            # ★ 라벨 보존 결과 확인
            st.write("🏷️ 라벨 보존 결과:")
            preserved_labels = [row['label'] for row in final_results['row_labels'] if row['label'] != "Unknown"]
            unknown_count = len([row for row in final_results['row_labels'] if row['label'] == "Unknown"])
            
            st.write(f"✅ 보존된 라벨: {len(preserved_labels)}개")
            st.write(f"❌ Unknown 라벨: {unknown_count}개")
            
            if len(preserved_labels) < len(original_labels):
                lost_labels = set(original_labels) - set(preserved_labels)
                st.error(f"🚨 손실된 라벨들: {lost_labels}")
            else:
                st.success("🎉 모든 라벨이 성공적으로 보존되었습니다!")
            
            # 결과 저장
            base_name = pkl_file.replace('_lines.pkl', '')
            output_file = os.path.join(Slab_text_clean, f"{base_name}_grid_cells.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            
            st.success(f"격자 셀별 텍스트 추출 완료: {output_file}")
            
            # 결과 미리보기
            st.write(f"### {base_name} 결과 미리보기")
            st.write(f"총 {len(final_results['cells'])}개 셀, {len(final_results['row_labels'])}개 행")
            
            # 보존된 라벨들 표시
            if preserved_labels:
                st.write("**보존된 라벨들:**")
                for label in preserved_labels[:10]:  # 처음 10개만
                    st.write(f"- {label}")
                    
        except Exception as e:
            st.error(f"오류 발생 ({pkl_file}): {str(e)}")


def grid_cells_to_dataframe_with_labels(grid_data: dict) -> pd.DataFrame:
    """
    라벨 정보가 포함된 grid_data에서 DataFrame을 생성 (수정된 버전)
    """
    # row_labels에서 라벨 정보 추출
    row_label_map = {}
    if 'row_labels' in grid_data:
        for row_info in grid_data['row_labels']:
            row_label_map[row_info['row_index']] = row_info['label']
    
    # cells 데이터 처리
    cells = grid_data.get('cells', grid_data)  # 이전 버전 호환성
    
    # row별로 데이터 구조화
    rows = {}
    max_col = 0
    
    for cell in cells:
        row_idx = cell['row']
        col_idx = cell['col']
        max_col = max(max_col, col_idx)
        
        # 행 초기화
        if row_idx not in rows:
            rows[row_idx] = {}
        
        # 셀에 텍스트가 있으면 추가
        if cell['extracted_texts']:
            # 여러 텍스트가 있으면 첫 번째 사용 (이미 병합되었음)
            text = cell['extracted_texts'][0]['text'].strip()
            rows[row_idx][col_idx] = text
    
    # DataFrame 생성
    df = pd.DataFrame.from_dict(rows, orient='index')
    
    # 컬럼 정렬 (0부터 max_col까지)
    all_cols = list(range(max_col + 1))
    df = df.reindex(columns=all_cols, fill_value='')
    
    # 행 인덱스 정렬
    df = df.sort_index()
    
    # ★ 라벨 정보를 행 인덱스로 추가
    if row_label_map:
        new_index = []
        for row_idx in df.index:
            label = row_label_map.get(row_idx, f"Row_{row_idx}")
            new_index.append(f"{label} (Row {row_idx})")
        df.index = new_index
    
    # 컬럼명을 더 의미있게 변경
    col_names = []
    for i in range(len(df.columns)):
        if i == 0:
            col_names.append('TYPE/FLOOR')
        elif i == 1:
            col_names.append('NAME')
        elif i == 2:
            col_names.append('THK(mm)')
        else:
            col_names.append(f'Column_{i}')
    
    df.columns = col_names[:len(df.columns)]
    
    return df


def preview_first_grid_file_with_labels(input_folder: str) -> pd.DataFrame:
    """
    첫 번째 grid_cells 파일을 라벨 정보와 함께 미리보기
    """
    files = sorted(f for f in os.listdir(input_folder) 
                  if f.lower().endswith("_grid_cells.json"))
    
    if not files:
        print("grid_cells.json 파일을 찾을 수 없습니다!")
        return pd.DataFrame()
    
    first_path = os.path.join(input_folder, files[0])
    
    with open(first_path, encoding="utf-8") as f:
        grid_data = json.load(f)
    
    return grid_cells_to_dataframe_with_labels(grid_data)


def save_all_grid_files_with_labels(input_folder: str, output_folder: str) -> list:
    """
    모든 grid_cells 파일을 라벨 정보와 함께 엑셀로 변환하여 저장
    """
    os.makedirs(output_folder, exist_ok=True)
    saved_paths = []
    
    files = sorted(f for f in os.listdir(input_folder) 
                  if f.lower().endswith("_grid_cells.json"))
    
    for fname in files:
        path = os.path.join(input_folder, fname)
        
        with open(path, encoding="utf-8") as f:
            grid_data = json.load(f)
        
        df = grid_cells_to_dataframe_with_labels(grid_data)
        
        # 출력 파일명 생성
        base = fname.replace('_grid_cells.json', '')
        out_path = os.path.join(output_folder, f"{base}_table_with_labels.xlsx")
        
        # 엑셀 저장
        df.to_excel(out_path, index=True, index_label='Row_Label')
        saved_paths.append(out_path)
        print(f"저장됨: {out_path}")
    
    return saved_paths

















#------------------------------------------------------------------------------

def grid_cells_to_dataframe(grid_data: dict) -> pd.DataFrame:
    """
    새로운 격자 셀 구조를 직접 활용하여 DataFrame을 생성
    클러스터링 없이 정확한 row, col 정보 사용
    """
    # 새로운 JSON 구조에서 cells 추출
    grid_cells = grid_data['cells']
    row_labels_info = {row['row_index']: row['label'] for row in grid_data['row_labels']}
    
    # row_labels에서 모든 행 번호 가져오기 (누락 방지)
    all_row_indices = sorted([row['row_index'] for row in grid_data['row_labels']])
    
    # 최대 컬럼 번호 찾기
    max_col = 0
    for cell in grid_cells:
        max_col = max(max_col, cell['col'])
    
    # 모든 행에 대해 데이터 구조화 (빈 행도 포함)
    rows = {}
    
    # 먼저 모든 행을 빈 딕셔너리로 초기화
    for row_idx in all_row_indices:
        rows[row_idx] = {}
    
    # 셀 데이터로 채우기
    for cell in grid_cells:
        row_idx = cell['row']
        col_idx = cell['col']
        
        # 셀에 텍스트가 있으면 추가
        if cell['extracted_texts']:
            # 여러 텍스트가 있으면 첫 번째 사용 (이미 병합되었음)
            text = cell['extracted_texts'][0]['text'].strip()
            rows[row_idx][col_idx] = text
    
    # DataFrame 생성
    df = pd.DataFrame.from_dict(rows, orient='index')
    
    # 컬럼 정렬 (0부터 max_col까지)
    all_cols = list(range(max_col + 1))
    df = df.reindex(columns=all_cols, fill_value='')
    
    # 행 인덱스를 row_labels 순서로 정렬
    df = df.reindex(all_row_indices, fill_value='')
    
    # 행 라벨 정보를 첫 번째 컬럼으로 추가
    row_label_column = []
    for row_idx in df.index:
        label = row_labels_info.get(row_idx, 'Unknown')
        row_label_column.append(label)
    
    # 행 라벨 컬럼을 DataFrame의 첫 번째 컬럼으로 삽입
    df.insert(0, 'ROW_LABEL', row_label_column)
    
    # 컬럼명을 더 의미있게 변경
    col_names = ['ROW_LABEL']  # 첫 번째는 행 라벨
    for i in range(1, len(df.columns)):
        original_col_idx = i - 1  # ROW_LABEL 때문에 1 빼기
        if original_col_idx == 0:
            col_names.append('TYPE')
        elif original_col_idx == 1:
            col_names.append('NAME')
        elif original_col_idx == 2:
            col_names.append('THK(mm)')
        else:
            col_names.append(f'Column_{original_col_idx}')
    
    df.columns = col_names[:len(df.columns)]
    
    print(f"DataFrame 생성 완료: {len(df)}행 × {len(df.columns)}컬럼")
    print(f"행 인덱스 범위: {df.index.min()} ~ {df.index.max()}")
    
    return df


def preview_first_grid_file(input_folder: str) -> pd.DataFrame:
    """
    첫 번째 grid_cells 파일을 미리보기
    """
    files = sorted(f for f in os.listdir(input_folder) 
                  if f.lower().endswith("_grid_cells.json"))
    
    if not files:
        print("grid_cells.json 파일을 찾을 수 없습니다!")
        return pd.DataFrame()
    
    first_path = os.path.join(input_folder, files[0])
    
    with open(first_path, encoding="utf-8") as f:
        grid_data = json.load(f)  # 새로운 구조로 로드
    
    return grid_cells_to_dataframe(grid_data)


def save_all_grid_files(input_folder: str, output_folder: str) -> list:
    """
    모든 grid_cells 파일을 엑셀로 변환하여 저장
    """
    os.makedirs(output_folder, exist_ok=True)
    saved_paths = []
    
    files = sorted(f for f in os.listdir(input_folder) 
                  if f.lower().endswith("_grid_cells.json"))
    
    for fname in files:
        path = os.path.join(input_folder, fname)
        
        with open(path, encoding="utf-8") as f:
            grid_data = json.load(f)  # 새로운 구조로 로드
        
        df = grid_cells_to_dataframe(grid_data)
        
        # 출력 파일명 생성
        base = fname.replace('_grid_cells.json', '')
        out_path = os.path.join(output_folder, f"{base}_table.xlsx")
        
        # 엑셀 저장 (행 라벨이 있으므로 index는 제거)
        df.to_excel(out_path, index=False)
        saved_paths.append(out_path)
        print(f"저장됨: {out_path}")
        
        # 미리보기 정보 출력
        print(f"  - 총 {len(df)}개 행, {len(df.columns)}개 컬럼")
        print(f"  - 행 라벨: {df['ROW_LABEL'].nunique()}개 (Unknown 제외: {df[df['ROW_LABEL'] != 'Unknown']['ROW_LABEL'].nunique()}개)")
    
    return saved_paths


def analyze_grid_structure(grid_data: dict) -> dict:
    """
    격자 구조 분석 정보 제공
    """
    grid_cells = grid_data['cells']
    row_labels = grid_data['row_labels']
    
    if not grid_cells:
        return {}
    
    # 실제 존재하는 행과 열 정보
    rows = sorted(set(cell['row'] for cell in grid_cells))
    cols = sorted(set(cell['col'] for cell in grid_cells))
    
    # 텍스트가 있는 셀 개수
    filled_cells = sum(1 for cell in grid_cells if cell['extracted_texts'])
    
    # 각 행별 텍스트 개수
    row_text_counts = {}
    for cell in grid_cells:
        row = cell['row']
        if row not in row_text_counts:
            row_text_counts[row] = 0
        if cell['extracted_texts']:
            row_text_counts[row] += 1
    
    # 행 라벨 통계
    label_stats = {}
    for row_label in row_labels:
        label = row_label['label']
        if label not in label_stats:
            label_stats[label] = 0
        label_stats[label] += 1
    
    # 실제 행 개수 정확히 계산
    actual_row_count = len(rows)
    missing_rows = []
    if rows:
        expected_rows = list(range(min(rows), max(rows) + 1))
        missing_rows = [r for r in expected_rows if r not in rows]
    
    return {
        'total_cells': len(grid_cells),
        'filled_cells': filled_cells,
        'empty_cells': len(grid_cells) - filled_cells,
        'rows_range': (min(rows) if rows else 0, max(rows) if rows else 0),
        'cols_range': (min(cols) if cols else 0, max(cols) if cols else 0),
        'rows_count': actual_row_count,  # 실제 존재하는 행 개수
        'cols_count': len(cols),
        'missing_rows': missing_rows,  # 누락된 행 번호들
        'actual_rows': rows,  # 실제 존재하는 행 번호들
        'row_text_counts': row_text_counts,
        'total_row_labels': len(row_labels),
        'unique_labels': len(label_stats),
        'label_distribution': label_stats
    }


def display_excel_preview(input_folder: str, max_rows: int = 10):
    """
    Excel 변환 결과를 미리보기로 표시
    """
    df = preview_first_grid_file(input_folder)
    
    if df.empty:
        print("미리보기할 데이터가 없습니다.")
        return
    
    print(f"\n### Excel 변환 미리보기 (처음 {min(max_rows, len(df))}행)")
    print(f"전체 크기: {len(df)}행 × {len(df.columns)}컬럼")
    print("\n" + "="*100)
    
    # 처음 몇 행만 출력
    preview_df = df.head(max_rows)
    
    # 컬럼 너비 조정해서 출력
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)
    
    print(preview_df.to_string(index=False))
    
    # 통계 정보
    print("\n" + "="*100)
    print("### 통계 정보")
    print(f"- 비어있지 않은 행 라벨: {len(df[df['ROW_LABEL'] != 'Unknown'])}")
    print(f"- Unknown 행: {len(df[df['ROW_LABEL'] == 'Unknown'])}")
    print(f"- 고유 행 라벨: {df['ROW_LABEL'].nunique()}개")
    print(f"- 실제 행 번호 범위: {df.index.min()} ~ {df.index.max()}")
    
    # 행 라벨별 개수
    label_counts = df['ROW_LABEL'].value_counts()
    print(f"- 행 라벨 분포:")
    for label, count in label_counts.head(10).items():
        if label != 'Unknown':
            print(f"  • {label}: {count}행")
    if 'Unknown' in label_counts:
        print(f"  • Unknown: {label_counts['Unknown']}행")










# def visualize_saved_bounding_boxes(selectbox_key: str = "viz_select") -> None:
#     """
#     저장된 바운딩박스 JSON 파일을 불러와서
#     원본 이미지 위에 박스들을 시각화해서 보여주는 함수
#     """
    
#     # JSON 파일들 찾기
#     json_files = [f for f in os.listdir(Slab_same_line) if f.endswith("_boxes.json")]
    
#     if not json_files:
#         st.warning(f"저장된 박스 파일이 없어요: {Slab_same_line}")
#         return
    
#     # 파일 선택
#     selected_json = st.selectbox("박스 파일 선택", json_files, key=selectbox_key)
    
#     # JSON 파일 읽기
#     json_path = os.path.join(Slab_same_line, selected_json)
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             boxes = json.load(f)
#     except Exception as e:
#         st.error(f"파일 읽기 실패: {e}")
#         return
    
#     if not boxes:
#         st.warning("박스 데이터가 없어요!")
#         return
    
#     # 해당하는 원본 이미지 찾기
#     img_name = selected_json.replace("_boxes.json", "")
#     img_extensions = [".png", ".jpg", ".jpeg"]
#     img_path = None
    
#     for ext in img_extensions:
#         test_path = os.path.join(Slab_column_SDD, img_name + ext)
#         if os.path.exists(test_path):
#             img_path = test_path
#             break
    
#     if not img_path:
#         st.error(f"원본 이미지를 찾을 수 없어요: {img_name}")
#         return
    
#     # 이미지 열기
#     image = Image.open(img_path).convert("RGB")
#     img_array = np.array(image)
    
#     # matplotlib으로 시각화
#     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#     ax.imshow(img_array)
    
#     # 각 박스 그리기
#     colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'pink']
    
#     for i, box in enumerate(boxes):
#         color = colors[i % len(colors)]
        
#         # Rectangle 그리기
#         rect = Rectangle(
#             (box['left'], box['top']),
#             box['width'],
#             box['height'],
#             linewidth=2,
#             edgecolor=color,
#             facecolor='none',
#             alpha=0.8
#         )
#         ax.add_patch(rect)
        
#         # 박스 번호 표시
#         ax.text(
#             box['left'], 
#             box['top'] - 5,
#             f'#{i+1}',
#             fontsize=12,
#             color=color,
#             fontweight='bold',
#             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
#         )
    
#     ax.set_xlim(0, image.width)
#     ax.set_ylim(image.height, 0)  # y축 뒤집기
#     ax.set_title(f"바운딩박스 시각화: {img_name} ({len(boxes)}개)", fontsize=14, fontweight='bold')
#     ax.axis('off')
    
#     # Streamlit에 표시
#     st.pyplot(fig)
#     plt.close()  # 메모리 정리
    
#     # 박스 정보 표시
#     with st.expander(f"📊 박스 정보 ({len(boxes)}개)", expanded=False):
#         for i, box in enumerate(boxes):
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.write(f"**박스 #{i+1}**")
#                 st.write(f"위치: ({box['left']:.1f}, {box['top']:.1f})")
#             with col2:
#                 st.write(f"크기: {box['width']:.1f} × {box['height']:.1f}")
#                 st.write(f"끝점: ({box['right']:.1f}, {box['bottom']:.1f})")


####################################################################





# def load_pickle_files(input_dir="./"):
#     """pickle 파일들을 로드"""
#     pickle_files = []
    
#     # 현재 디렉토리의 모든 .pkl 파일 찾기
#     for file_path in Path(input_dir).glob("*.pkl"):
#         pickle_files.append(str(file_path))
    
#     # lines가 포함된 파일이 있으면 우선적으로 선택
#     lines_files = [f for f in pickle_files if "lines" in f]
#     if lines_files:
#         return lines_files
    
#     return pickle_files

# def load_pickle_data(file_path):
#     """개별 pickle 파일 로드"""
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     return data

# def create_grid_from_lines(horizontal_lines, vertical_lines):
#     """수평선과 수직선으로 격자 생성"""
#     # 수평선과 수직선 정렬
#     h_lines = sorted(horizontal_lines, key=lambda x: x[0][1])  # y좌표로 정렬
#     v_lines = sorted(vertical_lines, key=lambda x: x[0][0])   # x좌표로 정렬
    
#     # 격자 셀 정의 (각 셀은 (x1,y1,x2,y2) 형태)
#     grid_cells = []
    
#     for i in range(len(h_lines)-1):
#         row_cells = []
#         for j in range(len(v_lines)-1):
#             # 현재 셀의 경계 계산
#             y1 = h_lines[i][0][1]
#             y2 = h_lines[i+1][0][1]
#             x1 = v_lines[j][0][0]  
#             x2 = v_lines[j+1][0][0]
            
#             cell_bounds = (x1, y1, x2, y2)
#             row_cells.append(cell_bounds)
#         grid_cells.append(row_cells)
    
#     return grid_cells

# def assign_texts_to_grid(grid_cells, regions, vertical_lines):
#     """텍스트들을 격자 셀에 할당 (수직선 분할 포함)"""
    
#     def split_text_by_vertical_lines(text_info, vertical_lines):
#         """텍스트 bbox를 수직선으로 물리적으로 분할 (내부 함수)"""
#         text_bbox = text_info['bbox']
#         text = text_info['text'].strip()
        
#         # 현재 텍스트가 걸쳐있는 수직선들 찾기
#         text_left = text_bbox[0]
#         text_right = text_bbox[2]
        
#         # 텍스트 bbox 내에 있는 수직선들 찾기
#         intersecting_v_lines = []
#         for v_line in vertical_lines:
#             v_x = v_line[0][0]  # 수직선의 x 좌표
#             if text_left < v_x < text_right:
#                 intersecting_v_lines.append(v_x)
        
#         if not intersecting_v_lines:
#             return [text_info]  # 나눌 수직선이 없으면 원본 반환
        
#         # 수직선들을 정렬
#         intersecting_v_lines.sort()
        
#         # bbox 분할 포인트들 = [시작점] + [수직선들] + [끝점]
#         split_points = [text_left] + intersecting_v_lines + [text_right]
        
#         # 분할된 bbox들 생성
#         split_text_parts = []
        
#         for i in range(len(split_points) - 1):
#             segment_left = split_points[i]
#             segment_right = split_points[i + 1]
            
#             # 새로운 bbox 생성 (원본 텍스트는 그대로, bbox만 분할)
#             new_bbox = [
#                 segment_left,
#                 text_bbox[1],  # y1 그대로
#                 segment_right,
#                 text_bbox[3]   # y2 그대로
#             ]
            
#             # 분할된 텍스트 객체 생성
#             new_text_info = text_info.copy()
#             new_text_info['bbox'] = new_bbox
#             new_text_info['text'] = text  # 원본 텍스트 그대로
            
#             split_text_parts.append(new_text_info)
        
#         return split_text_parts
    
#     # 모든 region의 텍스트들을 하나로 모음
#     all_texts = []
#     for region in regions:
#         for text_info in region['texts']:
#             text_info['cell_label'] = region['cell_label']
#             all_texts.append(text_info)
    
#     # 격자와 같은 크기의 빈 테이블 생성
#     table = []
#     for i in range(len(grid_cells)):
#         row = []
#         for j in range(len(grid_cells[i])):
#             row.append([])  # 각 셀은 빈 리스트로 시작
#         table.append(row)
    
#     # 각 텍스트를 해당하는 셀에 할당
#     for text_info in all_texts:
#         text_bbox = text_info['bbox']
#         text_center_x = (text_bbox[0] + text_bbox[2]) / 2
#         text_center_y = (text_bbox[1] + text_bbox[3]) / 2
        
#         # 텍스트가 속하는 셀 찾기
#         assigned = False
#         for i, row in enumerate(grid_cells):
#             for j, cell_bounds in enumerate(row):
#                 x1, y1, x2, y2 = cell_bounds
#                 if x1 <= text_center_x <= x2 and y1 <= text_center_y <= y2:
#                     # 텍스트가 여러 열에 걸쳐있는지 확인
#                     text_width = text_bbox[2] - text_bbox[0]
#                     cell_width = x2 - x1
                    
#                     # 텍스트가 셀 너비의 1.2배 이상이면 분할 시도
#                     if text_width > cell_width * 1.2:
#                         split_texts = split_text_by_vertical_lines(text_info, vertical_lines)
                        
#                         # 분할된 텍스트들을 각각 해당 셀에 배정
#                         for split_text in split_texts:
#                             split_center_x = (split_text['bbox'][0] + split_text['bbox'][2]) / 2
#                             split_center_y = (split_text['bbox'][1] + split_text['bbox'][3]) / 2
                            
#                             # 분할된 텍스트의 셀 찾기
#                             for ii, rrow in enumerate(grid_cells):
#                                 for jj, ccell_bounds in enumerate(rrow):
#                                     xx1, yy1, xx2, yy2 = ccell_bounds
#                                     if xx1 <= split_center_x <= xx2 and yy1 <= split_center_y <= yy2:
#                                         table[ii][jj].append(split_text)
#                                         break
#                     else:
#                         table[i][j].append(text_info)
                    
#                     assigned = True
#                     break
#             if assigned:
#                 break
    
#     return table

# def merge_texts_in_cell(cell_texts):
#     """같은 셀 내의 텍스트들을 합치기"""
#     if not cell_texts:
#         return ""
    
#     if len(cell_texts) == 1:
#         # 단일 텍스트인 경우 - 원본 텍스트에서 해당 셀 영역의 텍스트만 추출
#         text_info = cell_texts[0]
#         return extract_text_for_cell(text_info)
    
#     # 여러 텍스트가 있는 경우 - x좌표 기준으로 정렬해서 합치기
#     sorted_texts = sorted(cell_texts, key=lambda x: x['bbox'][0])
    
#     merged_text = ""
#     for text_info in sorted_texts:
#         cell_text = extract_text_for_cell(text_info)
#         if cell_text.strip():
#             if merged_text and not merged_text.endswith(' '):
#                 merged_text += " " + cell_text.strip()
#             else:
#                 merged_text += cell_text.strip()
    
#     return merged_text.strip()

# def extract_text_for_cell(text_info):
#     """텍스트에서 해당 셀 영역에 해당하는 부분만 스마트하게 추출"""
#     original_text = text_info['text'].strip()
    
#     # 공백으로 토큰화
#     tokens = original_text.split()
#     if not tokens:
#         return original_text
    
#     # HD + 숫자 + @ + 숫자 패턴 조합 찾기
#     result_tokens = []
#     i = 0
    
#     while i < len(tokens):
#         current_token = tokens[i]
        
#         # HD로 시작하는 경우
#         if current_token.startswith('HD'):
#             pattern_tokens = [current_token]
#             j = i + 1
            
#             # HD 다음에 숫자가 올 때까지 수집
#             while j < len(tokens) and (tokens[j].isdigit() or tokens[j] in ['10', '13', '16']):
#                 pattern_tokens.append(tokens[j])
#                 j += 1
            
#             # @ 기호 찾기
#             if j < len(tokens) and tokens[j] == '@':
#                 pattern_tokens.append(tokens[j])
#                 j += 1
                
#                 # @ 다음 숫자 수집
#                 while j < len(tokens) and (tokens[j].isdigit() or tokens[j] in ['200', '300', '400']):
#                     pattern_tokens.append(tokens[j])
#                     j += 1
            
#             # 완성된 패턴이 있으면 조합해서 반환
#             if len(pattern_tokens) >= 2:  # HD + 숫자 최소
#                 result_tokens.extend(pattern_tokens)
#                 return ' '.join(result_tokens)
            
#             i = j
#         else:
#             # HD가 아닌 일반 토큰
#             result_tokens.append(current_token)
#             i += 1
            
#             # 하나의 의미있는 토큰을 찾으면 바로 반환
#             if len(result_tokens) >= 1:
#                 break
    
#     return ' '.join(result_tokens) if result_tokens else original_text

# def process_table_with_labels(table, regions):
#     """테이블 처리할 때 cell_label도 함께 처리"""
#     # cell_label별로 텍스트들을 그룹핑
#     label_to_texts = {}
    
#     # 먼저 모든 셀의 텍스트에서 cell_label 정보 수집
#     for row_idx, row in enumerate(table):
#         for col_idx, cell_texts in enumerate(row):
#             for text_info in cell_texts:
#                 if 'cell_label' in text_info:
#                     label = text_info['cell_label']
#                     if label not in label_to_texts:
#                         label_to_texts[label] = []
                    
#                     # 셀 위치와 텍스트 정보 저장
#                     label_to_texts[label].append({
#                         'row': row_idx,
#                         'col': col_idx,
#                         'text_info': text_info,
#                         'merged_text': merge_texts_in_cell([text_info])
#                     })
    
#     # 각 label별로 행 구성
#     result_table = []
    
#     for label, text_infos in label_to_texts.items():
#         # 해당 label의 텍스트들을 열 순서대로 정렬
#         text_infos.sort(key=lambda x: (x['row'], x['col']))
        
#         # 행별로 그룹핑
#         rows_by_row_idx = {}
#         for text_info in text_infos:
#             row_idx = text_info['row']
#             if row_idx not in rows_by_row_idx:
#                 rows_by_row_idx[row_idx] = []
#             rows_by_row_idx[row_idx].append(text_info)
        
#         # 각 행을 처리
#         for row_idx in sorted(rows_by_row_idx.keys()):
#             row_texts = rows_by_row_idx[row_idx]
            
#             # 해당 행의 모든 열 처리
#             max_col = max([t['col'] for t in row_texts]) if row_texts else 0
#             processed_row = [''] * (max_col + 2)  # +2는 label 열 추가용
            
#             # 첫 번째 열에 label 넣기
#             processed_row[0] = label
            
#             # 나머지 열에 텍스트 넣기
#             for text_info in row_texts:
#                 col_idx = text_info['col'] + 1  # label 열 때문에 +1
#                 if col_idx < len(processed_row):
#                     processed_row[col_idx] = text_info['merged_text']
            
#             result_table.append(processed_row)
    
#     return result_table

# def save_to_excel(processed_tables, output_dir="./output"):
#     """처리된 테이블들을 엑셀로 저장"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     with pd.ExcelWriter(f"{output_dir}/converted_tables.xlsx", engine='openpyxl') as writer:
#         for page_name, table_data in processed_tables.items():
#             # 빈 행들 제거
#             filtered_table = [row for row in table_data if any(cell.strip() for cell in row)]
            
#             if filtered_table:
#                 df = pd.DataFrame(filtered_table)
#                 # 시트명에서 특수문자 제거
#                 safe_sheet_name = page_name.replace('/', '_').replace('\\', '_')[:31]
#                 df.to_excel(writer, sheet_name=safe_sheet_name, index=False, header=False)
    
#     return f"{output_dir}/converted_tables.xlsx"

# def convert_uploaded_files(file_paths):
#     """업로드된 파일들을 직접 처리하는 함수"""
#     output_dir = "./output"
    
#     if not file_paths:
#         st.error("업로드된 파일이 없습니다.")
#         return None
    
#     st.write(f"처리할 파일들: {file_paths}")
    
#     processed_tables = {}
    
#     # 각 pickle 파일 처리
#     for pickle_file in file_paths:
#         st.write(f"처리 중: {pickle_file}")
        
#         try:
#             # pickle 데이터 로드
#             data = load_pickle_data(pickle_file)
            
#             # 데이터 구조 검증
#             required_keys = ['horizontal_lines', 'vertical_lines', 'regions']
#             missing_keys = [key for key in required_keys if key not in data]
            
#             if missing_keys:
#                 st.warning(f"{pickle_file}: 필요한 키가 없습니다 ({missing_keys}). 건너뜁니다.")
#                 continue
            
#             # 격자 생성
#             grid_cells = create_grid_from_lines(data['horizontal_lines'], data['vertical_lines'])
#             st.write(f"격자 크기: {len(grid_cells)} x {len(grid_cells[0]) if grid_cells else 0}")
            
#             # 텍스트를 격자에 할당
#             table = assign_texts_to_grid(grid_cells, data['regions'], data['vertical_lines'])
            
#             # 텍스트 병합 처리 (cell_label 포함)
#             processed_table = process_table_with_labels(table, data['regions'])
            
#             # 페이지명 생성
#             page_name = Path(pickle_file).stem
#             processed_tables[page_name] = processed_table
            
#             st.success(f"{page_name} 처리 완료")
            
#         except Exception as e:
#             st.error(f"{pickle_file} 처리 중 오류: {str(e)}")
#             import traceback
#             st.text(traceback.format_exc())
#             continue
    
#     if processed_tables:
#         # 엑셀 파일로 저장
#         excel_path = save_to_excel(processed_tables, output_dir)
#         st.success(f"변환 완료! 저장 경로: {excel_path}")
#         return excel_path
    
#     return None

# def convert_pickle_to_excel():
#     """메인 변환 함수"""
#     return convert_uploaded_files(load_pickle_files("./"))












# ==================== 함수부 ====================



# ==================== 함수부 ====================


# ==================== 함수부 ====================


def load_pickle_data(pickle_file):
    """업로드된 피클 파일 로드"""
    try:
        data = pickle.load(pickle_file)
        return data
    except Exception as e:
        st.error(f"피클 파일 로드 중 오류: {e}")
        return None

def extract_line_coordinates(horizontal_lines, vertical_lines):
    """격자선에서 좌표 추출"""
    
    def extract_coords(lines, is_horizontal=True):
        """선 데이터에서 좌표 추출"""
        coords = []
        for line in lines:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                if isinstance(line[0], (list, tuple)):
                    # [[x1,y1], [x2,y2]] 형태
                    coord = line[0][1] if is_horizontal else line[0][0]
                else:
                    # [x1, y1, x2, y2] 형태
                    coord = line[1] if is_horizontal else line[0]
                coords.append(coord)
            elif isinstance(line, (int, float)):
                coords.append(line)
        return sorted(list(set(coords)))
    
    h_coords = extract_coords(horizontal_lines, True)   # y좌표들
    v_coords = extract_coords(vertical_lines, False)    # x좌표들
    
    return h_coords, v_coords

def calculate_bbox_area(bbox):
    """bbox의 면적 계산"""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def calculate_bbox_intersection(bbox1, bbox2):
    """두 bbox의 교집합 영역 계산"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 교집합 좌표 계산
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)
    
    # 교집합이 없으면 0
    if x1_int >= x2_int or y1_int >= y2_int:
        return 0
    
    return (x2_int - x1_int) * (y2_int - y1_int)

def calculate_overlap_ratio(bbox1, bbox2):
    """두 bbox의 겹침 비율 계산 (작은 박스 기준)"""
    intersection_area = calculate_bbox_intersection(bbox1, bbox2)
    if intersection_area == 0:
        return 0
    
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    
    # 작은 박스 기준으로 겹침 비율 계산
    smaller_area = min(area1, area2)
    if smaller_area == 0:
        return 0
    
    return intersection_area / smaller_area

def remove_duplicate_texts(texts, overlap_threshold=0.7, show_details=False):
    """중복된 bbox를 가진 텍스트들 제거 (큰 박스만 유지)"""
    if len(texts) <= 1:
        return texts, []
    
    removal_log = []
    texts_to_keep = []
    texts_to_remove = set()
    
    # 모든 텍스트 쌍에 대해 겹침 검사
    for i in range(len(texts)):
        if i in texts_to_remove:
            continue
            
        current_text = texts[i]
        current_bbox = current_text['bbox']
        current_area = calculate_bbox_area(current_bbox)
        
        for j in range(i + 1, len(texts)):
            if j in texts_to_remove:
                continue
                
            other_text = texts[j]
            other_bbox = other_text['bbox']
            other_area = calculate_bbox_area(other_bbox)
            
            # 겹침 비율 계산
            overlap_ratio = calculate_overlap_ratio(current_bbox, other_bbox)
            
            if overlap_ratio > overlap_threshold:
                # 겹침이 임계값 이상이면 작은 박스 제거
                if current_area >= other_area:
                    # current가 더 크거나 같으면 other 제거
                    texts_to_remove.add(j)
                    if show_details:
                        removal_log.append(f"    🗑️ 중복 제거: '{other_text['text']}' (겹침률: {overlap_ratio:.2f})")
                        removal_log.append(f"      유지: '{current_text['text']}' (면적: {current_area:.0f})")
                else:
                    # other가 더 크면 current 제거
                    texts_to_remove.add(i)
                    if show_details:
                        removal_log.append(f"    🗑️ 중복 제거: '{current_text['text']}' (겹침률: {overlap_ratio:.2f})")
                        removal_log.append(f"      유지: '{other_text['text']}' (면적: {other_area:.0f})")
                    break  # current가 제거되었으므로 더 이상 비교할 필요 없음
    
    # 제거할 텍스트들을 제외하고 유지할 텍스트들만 수집
    for i, text in enumerate(texts):
        if i not in texts_to_remove:
            texts_to_keep.append(text)
    
    if show_details and removal_log:
        removal_log.insert(0, f"  🔍 중복 검사: {len(texts)}개 → {len(texts_to_keep)}개 (임계값: {overlap_threshold})")
    
    return texts_to_keep, removal_log
    """텍스트들을 셀 단위로 그룹핑"""
    cell_groups = {}  # {(row, col): [text_info, ...]}
    
    for text_info in texts:
        bbox = text_info['bbox']
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        
        # 셀 위치 찾기
        col_idx = 0
        for i in range(len(v_coords) - 1):
            if v_coords[i] <= x_center <= v_coords[i + 1]:
                col_idx = i
                break
        
        row_idx = 0
        for i in range(len(h_coords) - 1):
            if h_coords[i] <= y_center <= h_coords[i + 1]:
                row_idx = i
                break
        
        cell_key = (row_idx, col_idx)
        if cell_key not in cell_groups:
            cell_groups[cell_key] = []
        cell_groups[cell_key].append(text_info)
    
    return cell_groups

def merge_cell_texts(text_list):
    """같은 셀 내의 텍스트들을 x좌표 순으로 정렬해서 합치고 새로운 bbox 생성"""
    if not text_list:
        return None
    
    if len(text_list) == 1:
        return text_list[0]
    
    # x좌표 기준으로 정렬
    sorted_texts = sorted(text_list, key=lambda x: x['bbox'][0])
    
    # 텍스트 합치기
    merged_text = ' '.join([t['text'].strip() for t in sorted_texts])
    
    # 새로운 bbox 계산 (모든 텍스트를 포함하는 최소 박스)
    min_x = min(t['bbox'][0] for t in sorted_texts)
    min_y = min(t['bbox'][1] for t in sorted_texts)
    max_x = max(t['bbox'][2] for t in sorted_texts)
    max_y = max(t['bbox'][3] for t in sorted_texts)
    
    merged_bbox = [min_x, min_y, max_x, max_y]
    
    # 평균 신뢰도 계산
    avg_confidence = sum(t.get('confidence', 0) for t in sorted_texts) / len(sorted_texts)
    
    return {
        'bbox': merged_bbox,
        'text': merged_text,
        'confidence': avg_confidence,
        'original_count': len(sorted_texts)
    }
    """bbox가 걸쳐진 모든 열 인덱스 찾기"""
    x1, y1, x2, y2 = bbox
    overlapping_cols = []
    
    for i in range(len(v_coords) - 1):
        col_left = v_coords[i]
        col_right = v_coords[i + 1]
        
        # bbox와 열이 겹치는지 확인
        if not (x2 < col_left or x1 > col_right):  # 겹침 조건
            overlapping_cols.append(i)
    
    return overlapping_cols

def find_overlapping_columns(bbox, v_coords):
    """bbox가 걸쳐진 모든 열 인덱스 찾기"""
    x1, y1, x2, y2 = bbox
    overlapping_cols = []
    
    for i in range(len(v_coords) - 1):
        col_left = v_coords[i]
        col_right = v_coords[i + 1]
        
        # bbox와 열이 겹치는지 확인
        if not (x2 < col_left or x1 > col_right):  # 겹침 조건
            overlapping_cols.append(i)
    
    return overlapping_cols

def distribute_rebars_to_cells(rebar_list, overlapping_cols):
    """파싱된 철근들을 걸쳐진 셀들에 개별 분배 - 절대 합치지 않음"""
    distribution = {}
    
    if not rebar_list or not overlapping_cols:
        return distribution
    
    # 각 철근을 개별 셀에 배치
    for i, rebar in enumerate(rebar_list):
        if i < len(overlapping_cols):
            # 원래 걸쳐진 열에 배치
            col_idx = overlapping_cols[i]
        else:
            # 걸쳐진 열을 벗어나면 마지막 열 이후로 계속 확장
            col_idx = overlapping_cols[-1] + (i - len(overlapping_cols) + 1)
        
        # 해당 열에 이미 값이 있으면 다음 열로
        while col_idx in distribution:
            col_idx += 1
        
        distribution[col_idx] = rebar
    
    return distribution
    return distribution

def merge_text_in_cell(texts):
    """같은 셀 내의 텍스트들을 x좌표 기준으로 정렬하여 합치기"""
    if not texts:
        return ""
    
    # x좌표 기준으로 정렬
    sorted_texts = sorted(texts, key=lambda x: x['bbox'][0])
    
    # 텍스트 합치기
    merged_text = " ".join([t['text'].strip() for t in sorted_texts])
    return merged_text.strip()

def parse_rebar_info(text):
    """철근 정보 파싱: (철근종류)(직경)@(간격) 패턴 - 디버깅 강화"""
    
    # 1. 텍스트 전처리 (파이프 문자 공백으로 변경)
    cleaned_text = text.replace('|', ' ').replace('"', '').strip()
    # 여러 공백을 하나로 통일
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    if '@ ' in cleaned_text or '@' in cleaned_text:
        print(f"🔍 파싱 디버깅:")
        print(f"  원본: '{text}'")
        print(f"  정리: '{cleaned_text}'")
    
    # 2. 모든 철근 패턴을 한번에 찾기
    all_rebar_pattern = r'(UHD|SUHD|SHD|HD|D|UD)\s*(\d+)(?:\+(\d+))?\s*@\s*(\d+)'
    
    # 모든 매치 찾기
    matches = re.findall(all_rebar_pattern, cleaned_text, re.IGNORECASE)
    
    if '@ ' in cleaned_text or '@' in cleaned_text:
        print(f"  매치 결과: {matches}")
    
    rebar_list = []
    
    for match in matches:
        rebar_type, diameter1, diameter2, spacing = match
        
        if diameter2:  # 복합 철근 (예: HD 10+13 @ 200)
            rebar_info = f"{rebar_type}{diameter1}+{diameter2}@{spacing}"
        else:  # 기본 철근 (예: HD 10 @ 400)
            rebar_info = f"{rebar_type}{diameter1}@{spacing}"
        
        rebar_list.append(rebar_info)
    
    if '@ ' in cleaned_text or '@' in cleaned_text:
        print(f"  최종 결과: {rebar_list}")
    
    # 3. 파싱 검증 및 디버깅
    if len(rebar_list) == 0 and cleaned_text.strip():
        # 철근 관련 키워드가 있는데 파싱이 안된 경우
        iron_keywords = ['SHD', 'UHD', 'SUHD', 'HD', 'D', 'UD']
        has_iron_keyword = any(keyword in cleaned_text.upper() for keyword in iron_keywords)
        has_at_symbol = '@' in cleaned_text
        
        if has_iron_keyword and has_at_symbol:
            print(f"❌ 파싱 실패 - 재시도: '{cleaned_text}'")
            
            # 더 관대한 패턴으로 재시도
            loose_pattern = r'([A-Z]+)\s*(\d+)(?:\+(\d+))?\s*@\s*(\d+)'
            loose_matches = re.findall(loose_pattern, cleaned_text, re.IGNORECASE)
            
            print(f"  관대한 매치: {loose_matches}")
            
            for match in loose_matches:
                rebar_type, diameter1, diameter2, spacing = match
                # 유효한 철근 타입인지 확인
                if rebar_type.upper() in iron_keywords:
                    if diameter2:
                        rebar_info = f"{rebar_type.upper()}{diameter1}+{diameter2}@{spacing}"
                    else:
                        rebar_info = f"{rebar_type.upper()}{diameter1}@{spacing}"
                    rebar_list.append(rebar_info)
            
            print(f"  재시도 결과: {rebar_list}")
            
            # 여전히 파싱 실패면 디버깅 정보 추가
            if len(rebar_list) == 0:
                rebar_list.append(f"[파싱실패: {cleaned_text}]")
    
    return rebar_list

def find_overlapping_columns(bbox, v_coords):
    """bbox가 걸쳐진 모든 열 인덱스 찾기"""
    x1, y1, x2, y2 = bbox
    overlapping_cols = []
    
    for i in range(len(v_coords) - 1):
        col_left = v_coords[i]
        col_right = v_coords[i + 1]
        
        # bbox와 열이 겹치는지 확인
        if not (x2 < col_left or x1 > col_right):  # 겹침 조건
            overlapping_cols.append(i)
    
    return overlapping_cols

def group_texts_by_cells(texts, v_coords, h_coords):
    """텍스트들을 셀 단위로 그룹핑"""
    cell_groups = {}  # {(row, col): [text_info, ...]}
    
    for text_info in texts:
        bbox = text_info['bbox']
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        
        # 셀 위치 찾기
        col_idx = 0
        for i in range(len(v_coords) - 1):
            if v_coords[i] <= x_center <= v_coords[i + 1]:
                col_idx = i
                break
        
        row_idx = 0
        for i in range(len(h_coords) - 1):
            if h_coords[i] <= y_center <= h_coords[i + 1]:
                row_idx = i
                break
        
        cell_key = (row_idx, col_idx)
        if cell_key not in cell_groups:
            cell_groups[cell_key] = []
        cell_groups[cell_key].append(text_info)
    
    return cell_groups

def clean_text_content(text):
    """텍스트에서 불필요한 특수기호 제거 - ( ) + @ 는 유지"""
    if not text or not isinstance(text, str):
        return text
    
    # 유지할 문자: 한글, 영문, 숫자, 공백, ( ) + @
    # 정규식: 한글(\uAC00-\uD7AF), 영문(a-zA-Z), 숫자(0-9), 공백(\s), ()+ @
    cleaned = re.sub(r'[^\uAC00-\uD7AFa-zA-Z0-9\s()+ @]', '', text)
    
    # 여러 공백을 하나로 통일
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def process_regions_by_rows(regions, v_coords, h_coords, show_details=False):
    """region을 행 기준으로 처리 - 각 region을 개별 행으로 처리"""
    
    processing_log = []
    table_data = {}  # {unique_key: {col_index: content, y_position: float, cell_label: str}}
    
    skipped_regions = []
    processed_regions = []
    
    # 각 region을 개별적으로 처리 (병합하지 않음)
    for region_idx, region in enumerate(regions):
        cell_label = region.get('cell_label', '').strip()
        texts = region.get('texts', [])
        region_bounds = region.get('bounds', None)
        
        # 스킵 조건 체크 및 로깅
        if not cell_label:
            skipped_regions.append(f"region#{region_idx}: 빈 cell_label")
            if show_details:
                processing_log.append(f"⏭️ region#{region_idx} 스킵: 빈 cell_label")
            continue
            
        if show_details:
            processing_log.append(f"🔍 처리 중인 region#{region_idx}: '{cell_label}' (원본 텍스트 {len(texts)}개)")
        
        processed_regions.append(f"region#{region_idx}: {cell_label}")
        
        # 고유 키 생성 (region 인덱스 사용)
        unique_key = f"{cell_label}_region{region_idx}"
        
        # 행 데이터 초기화 (y 위치 정보 포함)
        if region_bounds:
            y_position = (region_bounds[1] + region_bounds[3]) / 2
        else:
            y_positions = [t['bbox'][1] for t in texts if 'bbox' in t]
            y_position = sum(y_positions) / len(y_positions) if y_positions else 0
        
        table_data[unique_key] = {
            'y_position': y_position, 
            'data': {},
            'cell_label': cell_label,
            'region_idx': region_idx
        }
        
        if show_details:
            processing_log.append(f"  ✅ 새 행 생성: '{unique_key}' (y={y_position:.0f})")
        
        # === 0단계: 중복 bbox 제거 ===
        cleaned_texts, removal_log = remove_duplicate_texts(texts, overlap_threshold=0.7, show_details=show_details)
        processing_log.extend(removal_log)
        
        if show_details and len(cleaned_texts) != len(texts):
            processing_log.append(f"  ✅ 중복 제거 완료: {len(texts)}개 → {len(cleaned_texts)}개")
        
        # === 1단계: 같은 셀에 있는 텍스트들 그룹핑 및 병합 ===
        cell_groups = group_texts_by_cells(cleaned_texts, v_coords, h_coords)
        merged_texts = []  # 병합된 텍스트들을 저장
        
        if show_details:
            processing_log.append(f"  📊 1단계 - 셀별 그룹핑: {len(cell_groups)}개 셀에 분산")
        
        for cell_key, cell_texts in cell_groups.items():
            row_idx, col_idx = cell_key
            
            # 같은 셀 내 텍스트들 병합
            merged_text_info = merge_cell_texts(cell_texts)
            
            if merged_text_info:
                original_count = merged_text_info.get('original_count', 1)
                
                if show_details and original_count > 1:
                    processing_log.append(f"    ✅ 셀({row_idx},{col_idx}): {original_count}개 텍스트 병합 → '{merged_text_info['text']}'")
                
                merged_texts.append(merged_text_info)
        
        # === 2단계: 병합된 텍스트들을 대상으로 파싱 및 분배 ===
        if show_details:
            processing_log.append(f"  📊 2단계 - 파싱 및 분배: {len(merged_texts)}개 병합 텍스트 처리")
        
        # 이미 사용된 열 추적을 위한 set
        used_columns = set()
        
        for merged_text_info in merged_texts:
            merged_text = merged_text_info['text']
            merged_bbox = merged_text_info['bbox']
            
            # 철근 정보 파싱
            rebar_list = parse_rebar_info(merged_text)
            
            if rebar_list:
                # 병합된 bbox로 걸쳐진 열들 찾기
                overlapping_cols = find_overlapping_columns(merged_bbox, v_coords)
                
                if show_details:
                    processing_log.append(f"    🎯 '{merged_text}' → 파싱: {rebar_list}")
                    processing_log.append(f"    📍 bbox 걸쳐진 열들: {overlapping_cols}")
                
                # 파싱된 철근들을 걸쳐진 셀들에 분배 (충돌 감지 포함)
                if overlapping_cols:
                    distribution = distribute_rebars_to_cells_with_conflict_detection(
                        rebar_list, overlapping_cols, used_columns
                    )
                    
                    for col_idx, rebar_content in distribution.items():
                        # 철근 정보 정리
                        cleaned_rebar = clean_text_content(rebar_content)
                        table_data[unique_key]['data'][col_idx] = cleaned_rebar
                        used_columns.add(col_idx)  # 사용된 열 기록
                        
                        if show_details:
                            processing_log.append(f"      📍 열 {col_idx}: {cleaned_rebar}")
            
            else:
                # 철근이 아닌 경우 (라벨, 숫자 등) - 원래 위치에 배치
                x_center = (merged_bbox[0] + merged_bbox[2]) / 2
                target_col = 0
                for i in range(len(v_coords) - 1):
                    if v_coords[i] <= x_center <= v_coords[i + 1]:
                        target_col = i
                        break
                
                # 충돌 확인 및 회피
                while target_col in used_columns:
                    target_col += 1
                
                # 비철근 텍스트 정리
                cleaned_text = clean_text_content(merged_text)
                table_data[unique_key]['data'][target_col] = cleaned_text
                used_columns.add(target_col)
                
                if show_details:
                    processing_log.append(f"    📍 비철근 텍스트 → 열 {target_col}: {cleaned_text}")
    
    # 처리 요약 로그 추가
    if show_details:
        processing_log.append(f"\n📊 === 처리 요약 ===")
        processing_log.append(f"전체 regions: {len(regions)}개")
        processing_log.append(f"처리된 regions: {len(processed_regions)}개")
        processing_log.append(f"스킵된 regions: {len(skipped_regions)}개")
        processing_log.append(f"최종 행 수: {len(table_data)}개 (각 region = 1행)")
        
        if skipped_regions:
            processing_log.append(f"\n⏭️ 스킵된 regions:")
            for skip_reason in skipped_regions:
                processing_log.append(f"  • {skip_reason}")
        
        # 같은 라벨을 가진 regions 표시 (병합하지 않고 정보만)
        label_groups = {}
        for region_info in processed_regions:
            region_num, label = region_info.split(': ', 1)
            if label in label_groups:
                label_groups[label].append(region_num)
            else:
                label_groups[label] = [region_num]
        
        same_label_groups = {k: v for k, v in label_groups.items() if len(v) > 1}
        if same_label_groups:
            processing_log.append(f"\n📋 같은 라벨을 가진 regions (각각 별도 행으로 처리):")
            for label, region_nums in same_label_groups.items():
                processing_log.append(f"  • '{label}': {region_nums}")
    
    return table_data, processing_log

def distribute_rebars_to_cells_with_conflict_detection(rebar_list, overlapping_cols, used_columns):
    """파싱된 철근들을 걸쳐진 셀들에 개별 분배 - 충돌 감지 및 회피"""
    distribution = {}
    
    if not rebar_list or not overlapping_cols:
        return distribution
    
    # 각 철근을 개별 셀에 배치
    for i, rebar in enumerate(rebar_list):
        if i < len(overlapping_cols):
            # 원래 걸쳐진 열에 배치 시도
            col_idx = overlapping_cols[i]
        else:
            # 걸쳐진 열을 벗어나면 마지막 열 이후로 계속 확장
            col_idx = overlapping_cols[-1] + (i - len(overlapping_cols) + 1)
        
        # 충돌 회피: 이미 사용된 열이면 다음 사용 가능한 열 찾기
        while col_idx in used_columns or col_idx in distribution:
            col_idx += 1
        
        distribution[col_idx] = rebar
    
    return distribution

def create_dataframe_from_table_data(table_data):
    """표 데이터를 DataFrame으로 변환 (실제 위치 순서대로 정렬) - 고유키 지원"""
    
    if not table_data:
        return pd.DataFrame()
    
    # 모든 열 인덱스 수집
    all_columns = set()
    for row_info in table_data.values():
        all_columns.update(row_info['data'].keys())
    
    max_col = max(all_columns) if all_columns else 0
    num_cols = max_col + 1
    
    # 행들을 y 위치 기준으로 정렬 (위에서 아래로)
    sorted_rows = sorted(table_data.items(), key=lambda x: x[1]['y_position'])
    
    # 데이터 매트릭스 생성
    data_matrix = []
    row_labels = []
    
    for unique_key, row_info in sorted_rows:
        # 실제 cell_label을 행 라벨로 사용
        cell_label = row_info.get('cell_label', unique_key)
        row_labels.append(cell_label)
        
        row_data = []
        for col_index in range(num_cols):
            content = row_info['data'].get(col_index, '')
            row_data.append(content)
        data_matrix.append(row_data)
    
    # DataFrame 생성 (행 라벨 포함, y 위치순 정렬)
    df = pd.DataFrame(data_matrix, index=row_labels)
    
    return df

def create_excel_file_from_dataframe(df):
    """DataFrame을 엑셀 파일로 생성 (바이트 스트림 반환)"""
    
    # 메모리에서 엑셀 파일 생성
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='표데이터', index=True, header=False)
        
        # 워크북과 워크시트 가져오기
        workbook = writer.book
        worksheet = writer.sheets['표데이터']
        
        # 셀 스타일링 (테두리 추가)
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # 모든 셀에 테두리 적용
        for row in range(1, df.shape[0] + 2):  # +2 for header
            for col in range(1, df.shape[1] + 2):  # +2 for index
                cell = worksheet.cell(row=row, column=col)
                cell.border = thin_border
                cell.alignment = Alignment(horizontal='center', vertical='center')
    
    output.seek(0)
    return output.getvalue()

def process_table_to_excel(pickle_file, show_details=False):
    """메인 처리 함수"""
    try:
        # 1. 피클 데이터 로드
        data = load_pickle_data(pickle_file)
        if data is None:
            return None, None, None
        
        horizontal_lines = data.get('horizontal_lines', [])
        vertical_lines = data.get('vertical_lines', [])
        regions = data.get('regions', [])
        
        st.success(f"✅ 데이터 로드 완료: 영역 {len(regions)}개")
        
        # === cell_label 분석 추가 ===
        all_labels = []
        empty_labels = 0
        label_counts = {}
        
        for i, region in enumerate(regions):
            cell_label = region.get('cell_label', '').strip()
            if not cell_label:
                empty_labels += 1
                all_labels.append(f"[빈라벨_{i}]")
            else:
                all_labels.append(cell_label)
                if cell_label in label_counts:
                    label_counts[cell_label] += 1
                else:
                    label_counts[cell_label] = 1
        
        valid_labels = [label for label in all_labels if not label.startswith('[빈라벨_')]
        unique_label_names = list(set(valid_labels))
        multiple_occurrence_labels = {k: v for k, v in label_counts.items() if v > 1}
        
        # cell_label 통계 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 regions", len(regions))
        with col2:
            st.metric("유효한 라벨", len(valid_labels))
        with col3:
            st.metric("빈 라벨", empty_labels)
        with col4:
            st.metric("고유 라벨명", len(unique_label_names))
        
        if show_details:
            st.write("**📋 cell_label 상세 분석:**")
            st.write(f"- 모든 라벨: {all_labels}")
            
            if multiple_occurrence_labels:
                st.write("**📊 여러 번 나타나는 라벨들 (정상):**")
                for label, count in multiple_occurrence_labels.items():
                    st.write(f"  • '{label}': {count}번 등장")
            
            if empty_labels > 0:
                st.write(f"**❌ 빈 라벨 {empty_labels}개 발견**")
        
        # 2. 격자선에서 좌표 추출
        h_coords, v_coords = extract_line_coordinates(horizontal_lines, vertical_lines)
        
        st.write(f"📊 격자 정보: 수평선 {len(h_coords)}개, 수직선 {len(v_coords)}개")
        
        # 디버깅 정보 (일반 출력)
        if show_details:
            st.write("**🔍 격자선 상세 정보:**")
            st.write(f"- 수평선 좌표(y): {h_coords[:10]}{'...' if len(h_coords) > 10 else ''}")
            st.write(f"- 수직선 좌표(x): {v_coords[:10]}{'...' if len(v_coords) > 10 else ''}")
            
            # 행 정렬 정보 표시
            region_positions = []
            region_text_counts = []
            
            for i, region in enumerate(regions):
                cell_label = region.get('cell_label', f'region_{i}').strip()
                region_bounds = region.get('bounds', None)
                texts = region.get('texts', [])
                
                if region_bounds:
                    y_pos = (region_bounds[1] + region_bounds[3]) / 2
                    region_positions.append((cell_label, y_pos, i))
                    region_text_counts.append((cell_label, len(texts)))
            
            region_positions.sort(key=lambda x: x[1])  # y 위치순 정렬
            st.write("**📍 행 순서 (위→아래):**")
            for j, (label, y_pos, region_idx) in enumerate(region_positions[:15]):
                text_count = next((count for name, count in region_text_counts if name == label), 0)
                empty_status = "🔴" if not label or label.startswith('[빈라벨_') else "🟢"
                st.write(f"  {j+1}. {empty_status} '{label}' (y={y_pos:.0f}, 텍스트 {text_count}개, region#{region_idx})")
            if len(region_positions) > 15:
                st.write(f"  ... 총 {len(region_positions)}개 행")
            
            # 전체 텍스트 통계
            total_texts = sum(len(region.get('texts', [])) for region in regions)
            st.write(f"**📊 전체 OCR 텍스트: {total_texts}개**")
        
        # 3. 행 기준으로 region 데이터 처리 (h_coords 추가)
        table_data, processing_log = process_regions_by_rows(regions, v_coords, h_coords, show_details)
        
        # === 처리 결과 검증 ===
        processed_labels = list(table_data.keys())
        expected_count = len(valid_labels)  # 빈 라벨 제외한 유효한 라벨 개수
        
        st.write("**🔍 처리 결과 검증:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("예상 처리 가능 행", expected_count)
        with col2:
            st.metric("실제 처리된 행", len(processed_labels))
        with col3:
            difference = len(processed_labels) - expected_count
            st.metric("차이", difference, delta=f"{difference:+d}")
        
        if show_details:
            st.write(f"**📊 처리된 라벨들:** {processed_labels}")
            
            # 빈 라벨로 인한 누락 확인
            if empty_labels > 0:
                st.write(f"**ℹ️ {empty_labels}개 region은 빈 라벨로 인해 처리 제외됨**")
        
        # 4. DataFrame 생성
        df = create_dataframe_from_table_data(table_data)
        
        if df.empty:
            st.error("생성된 데이터가 비어있습니다.")
            return None, None, processing_log
        
        # 5. 엑셀 파일 생성
        excel_data = create_excel_file_from_dataframe(df)
        
        return excel_data, df, processing_log
        
    except Exception as e:
        st.error(f"처리 중 오류 발생: {e}")
        import traceback
        if show_details:
            st.code(traceback.format_exc())
        return None, None, None
#------------------------------------------













# ───────────── Streamlit 호출부 ─────────────
st.set_page_config(page_title="Slab 추출", layout="wide")
st.title("Slab 추출")



tab1, tab2, tab3 = st.tabs(["SCD Slab_info extraction", "SDD Slab_info extraction", "Human Error detection"])

with tab1:
    # 초기화
    for key, d in [("boxes",[]),("anchors",[])]:
        if key not in st.session_state: st.session_state[key]=d

    uploaded = st.file_uploader("PDF 업로드", type="pdf")
    if uploaded and st.button("Step 0: PDF→이미지"):
        paths = convert_pdf_to_images(uploaded)
        st.success(f"{len(paths)}페이지 변환 완료")

    # Step 1
    with st.expander("Step 1: 레이아웃 분석"):
        if st.button("분석 실행"):
            img, boxes = analyze_first_page()
            if img:
                # 1) 중복 박스 제거 (IoU > 0.5 이면 같은 영역으로 간주)
                unique = []
                for b in boxes:
                    if not any(compute_iou(b, u) > 0.5 for u in unique):
                        unique.append(b)
                boxes = unique

                # 2) 세션에 원본 이미지와 박스 리스트 저장
                st.session_state["first_img"] = img.copy()
                st.session_state["boxes"]    = boxes

                # 3) 이미지에 박스와 번호 그리기
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 150)
                except IOError:
                    font = ImageFont.load_default()

                for idx, (x1, y1, x2, y2) in enumerate(boxes):
                    # 박스
                    draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=20)
                    # 번호
                    text = str(idx)
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    tx = x1 + (x2 - x1 - tw) // 2
                    ty = max(0, y1 - th - 5)
                    draw.text((tx, ty), text, fill="red", font=font)

                # 4) 결과 표시
                st.image(img, use_column_width=True)
            else:
                st.error("이미지가 없습니다")

    # Step 2
    with st.expander("Step 2: 앵커 선택", expanded=True):
        first_img = st.session_state.get("first_img", None)
        boxes     = st.session_state.get("boxes", [])

        if first_img is None or not boxes:
            st.info("먼저 Step 1에서 레이아웃 분석을 실행하세요")
        else:
            # 인덱스 입력 UI
            idxs = st.text_input("박스 인덱스 입력 (쉼표로 구분)", key="idxs_input")

            if st.button("✅ 앵커 저장"):
                # 선택된 앵커 인덱스 리스트
                selected = [int(s) for s in idxs.split(",") if s.strip().isdigit()]
                # 유효한 인덱스만 필터
                anchors = [boxes[i] for i in selected if 0 <= i < len(boxes)]
                st.session_state["anchors"] = anchors

                # 미리보기용 이미지 생성
                preview = first_img.copy()
                draw    = ImageDraw.Draw(preview)
                for (x1, y1, x2, y2) in anchors:
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=20)

                # 화면에 표시
                st.image(preview, caption=f"선택된 앵커: {selected}", use_column_width=True)
                st.success(f"저장된 앵커 좌표: {anchors}")

    # Step 3
    with st.expander("Step 3: 추출 실행", expanded=True):
        if st.button("➡️ 추출 시작"):
            anchors = st.session_state.get("anchors", [])
            if not anchors:
                st.error("먼저 Step 2에서 앵커를 저장하세요")
            else:
                extract_with_offset(anchors, margin_right=500)


    with st.expander("Step 4: OCR 적용", expanded=True):
        if st.button("🔍 OCR 실행"):
            apply_surya_ocr_to_anchors()


    with st.expander("Step 4: 모든 페이지 정보 추출", expanded=True):
        # 1) 첫 페이지 예시 이미지 보여주기

        anchor_folder = os.path.join(BASE_DIR, "Slab_anchor_img")
        preview_paths = sorted(glob.glob(os.path.join(anchor_folder, "*page_1_box*.png")))

        if preview_paths:
            cols = st.columns(len(preview_paths))
            for col, img_p in zip(cols, preview_paths):
                col.image(Image.open(img_p), use_column_width=True)
                col.caption(os.path.basename(img_p))
        else:
            st.info("먼저 Step 3에서 크롭을 완료하세요.")

        # 2) 키워드 입력 (최대 4개)
        st.markdown("**추출할 키워드를 최대 4개 입력하세요. 빈 칸은 무시됩니다.**")
        kws = [st.text_input(f"키워드 {i+1}", key=f"kw_all_{i}") for i in range(4)]
        keys = [k.strip() for k in kws if k.strip()]

        # 3) 전체 페이지 추출 실행
        if st.button("🚀 전체 페이지 추출 실행"):
            if not keys:
                st.warning("최소 하나의 키워드를 입력해주세요.")
            else:
                # 전체 페이지 정보를 추출·저장
                all_results = extract_all_pages(keys)

                # 첫 2페이지만 화면에 테이블로 표시
                for page_num in sorted(all_results)[:2]:
                    info = all_results[page_num]
                    st.subheader(f"▶️ Page {page_num}")
                    rows = [{"키워드": k, "값": info.get(k) or "—"} for k in keys]
                    st.table(rows)

                st.info(f"⚙️ 총 {len(all_results)}페이지 처리 완료. 나머지 결과는 '{Slab_elements}'에 저장되었습니다.")


    with st.expander("Step 6: 테이블 OCR 적용", expanded=True):
        if st.button("🔍 테이블 OCR 실행"):
            apply_surya_ocr_to_tables()


    with st.expander("Step 7 : 테이블 정렬"):
        y_tol = st.number_input("Y tolerance", value=10, min_value=0)
        x_tol = st.number_input("X tolerance", value=20, min_value=0)

        if st.button("변환 실행"):
            # 1) JSON → Excel 변환
            result = parse_ocr_jsons_to_excel(y_tol, x_tol)
            if not result:
                st.error("변환할 데이터가 없습니다")
                st.stop()
            st.success(f"변환 완료: {result}")

            # 2) 결과 로드 및 파일 선택
            df = pd.read_excel(result)
            files = df['file'].unique().tolist()
            selected = st.selectbox("미리보기할 파일 선택", files)
            base = os.path.splitext(selected)[0]

            # 3) 이미지 로드
            shown = False
            for ext in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"):
                img_path = os.path.join(Slab_table, base + ext)
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, caption=os.path.basename(img_path), use_column_width=True)
                    shown = True
                    break
            if not shown:
                st.warning(f"이미지 파일이 없습니다: {base}(.png/.jpg/.jpeg 등)")

            # 4) 표 결과 미리보기 (가로 꽉 채우기)
            preview = df[df['file'] == selected].drop(columns=["file"])
            st.dataframe(preview, use_container_width=True, height=300)












































#-------------------------------------------------------------------------------------


with tab2:
    uploaded = st.file_uploader("SDD PDF 업로드", type="pdf")
    if uploaded and st.button("Step 0:SDD: PDF→이미지"):
        paths = convert_pdf_to_images_SDD(uploaded)
        st.success(f"{len(paths)}페이지 변환 완료")



    with st.expander("바운딩박스 그리기_크롭용"):
        # "바운딩박스 시작" 버튼을 누르면 UI가 나타나게
        if "show_box_ui" not in st.session_state:
            st.session_state.show_box_ui = False

        if st.button("바운딩박스 시작"):
            st.session_state.show_box_ui = True

        if st.session_state.show_box_ui:
            draw_and_save_bounding_boxes_slab_SDD(canvas_key="box_crop")
            # "완료하기" 버튼을 넣어서 누르면 UI 닫힘
            if st.button("완료하기"):
                st.session_state.show_box_ui = False




    with st.expander("전체 OCR 때리기"):
        # 단계 1: OCR
        if st.button("OCR 실행"):
            apply_surya_ocr_Wall_slab_SDD()




    with st.expander("바운딩박스_라인따기"):
        # "바운딩박스 시작" 버튼을 누르면 UI가 나타나게
        if "show_box_ui" not in st.session_state:
            st.session_state.show_box_ui = False

        if st.button("바운딩박스 시작_라인따기"):
            st.session_state.show_box_ui = True

        if st.session_state.show_box_ui:
            draw_and_save_bounding_boxes_Slab_SDD(canvas_key="box_line_scroll")
            
            # "완료하기" 버튼을 넣어서 누르면 UI 닫힘
            if st.button("완료하기_라인따기"):
                st.session_state.show_box_ui = False



    with st.expander("수평/수직선 + 교차점 + 텍스트 인식", expanded=True):
            img_dir = st.text_input("입력 이미지 폴더 경로", value=r"D:\4parts_complete\slab\raw_data_column_label_SDD")
            save_dir = st.text_input("결과 저장 폴더명 (비우면 저장 X)", value=r"D:\4parts_complete\slab\Slab_table_region")
            boxes_coord_dir = st.text_input("사용자 박스 좌표 폴더 (*_boxes.json)", value=r"D:\4parts_complete\slab\Slab_same_line")
            ocr_coord_dir = st.text_input("OCR 결과 폴더 (*.json)", value=r"D:\4parts_complete\slab\raw_data_OCR_SDD")
            text_tolerance = st.slider("텍스트-선 매칭 허용 오차", 5, 30, 15, 5)
            
            mode = st.radio("선 검출 방식", options=["contour", "hough"], index=0)
            min_line_length = st.slider("min_line_length", 20, 300, 80, 5)
            max_line_gap = st.slider("max_line_gap", 2, 40, 10, 2)
            hough_threshold = st.slider("Hough threshold", 30, 300, 100, 5)
            morph_kernel_scale = st.slider("Morph kernel 비율", 10, 60, 30, 2)
            block_size = st.slider("adaptiveThreshold blockSize(홀수)", 7, 31, 15, 2)
            C = st.slider("adaptiveThreshold C", -10, 10, -2, 1)
            tol = st.slider("교차점 허용 오차 (tol)", 0, 10, 2, 1)

            run_btn = st.button("예시 5개 보기")
            save_btn = st.button("전체 저장")

            if run_btn:
                imgs = detect_and_draw_lines_batch(
                    img_dir=img_dir,
                    save_dir=None,
                    min_line_length=min_line_length,
                    max_line_gap=max_line_gap,
                    hough_threshold=hough_threshold,
                    morph_kernel_scale=morph_kernel_scale,
                    max_examples=5,
                    return_imgs=True,
                    mode=mode,
                    block_size=block_size | 1,
                    C=C,
                    tol=tol,
                    boxes_coord_dir=boxes_coord_dir if boxes_coord_dir else None,
                    ocr_coord_dir=ocr_coord_dir if ocr_coord_dir else None,
                    text_tolerance=text_tolerance
                )
                if not imgs:
                    st.warning("이미지가 없습니다")
                else:
                    for result in imgs:
                        if len(result) == 4:
                            fname, img, inter_cnt, text_cnt = result
                            st.subheader(f"{fname} (교차점: {inter_cnt}개, 선상 텍스트: {text_cnt}개)")
                        else:
                            fname, img, inter_cnt = result[:3]
                            st.subheader(f"{fname} (교차점: {inter_cnt}개)")
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="검출 결과", use_column_width=True)

            if save_btn:
                detect_and_draw_lines_batch(
                    img_dir=img_dir,
                    save_dir=save_dir if save_dir else None,
                    min_line_length=min_line_length,
                    max_line_gap=max_line_gap,
                    hough_threshold=hough_threshold,
                    morph_kernel_scale=morph_kernel_scale,
                    max_examples=None,
                    return_imgs=False,
                    mode=mode,
                    block_size=block_size | 1,
                    C=C,
                    tol=tol,
                    boxes_coord_dir=boxes_coord_dir if boxes_coord_dir else None,
                    ocr_coord_dir=ocr_coord_dir if ocr_coord_dir else None,
                    text_tolerance=text_tolerance
                )
                st.success(f"전체 이미지 결과 저장 완료! → '{save_dir}' 폴더 확인")





    # with st.expander("📁 폴더 내 모든 JSON 정제하기"):
    #     folder = st.text_input("JSON 폴더 경로", value=Slab_table_region)
    #     merge_tol = st.number_input("병합 기준 픽셀 거리 (merge_tol)", min_value=0, max_value=100, value=10, step=1)
    #     if st.button("✅ 정제 실행"):
    #         if not os.path.isdir(folder):
    #             st.error("유효한 폴더 경로를 입력하세요")
    #         else:
    #             json_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".json")])
    #             if not json_files:
    #                 st.warning("폴더에 JSON 파일이 없습니다")
    #             else:
    #                 save_folder = os.path.join(folder, "cleaned_merged")
    #                 os.makedirs(save_folder, exist_ok=True)
    #                 downloaded = []

    #                 for fname in json_files:
    #                     path = os.path.join(folder, fname)
    #                     with open(path, encoding="utf-8") as f:
    #                         regions = json.load(f)

    #                     # 필터링 + 병합
    #                     cleaned = filter_regions_text(regions, merge_tol)

    #                     out_name = fname.replace(".json", "_cleaned.json")
    #                     out_path = os.path.join(save_folder, out_name)
    #                     with open(out_path, "w", encoding="utf-8") as wf:
    #                         json.dump(cleaned, wf, ensure_ascii=False, indent=2)

    #                     downloaded.append((out_name, json.dumps(cleaned, ensure_ascii=False, indent=2)))

    #                 st.success(f"{len(downloaded)}개 파일 정제 및 병합 완료! (폴더: {save_folder})")
    #                 for out_name, content in downloaded:
    #                     st.download_button(
    #                         label=f"📥 {out_name} 다운로드",
    #                         data=content,
    #                         file_name=out_name,
    #                         mime="application/json"
    #                     )








    # with st.expander("클러스터링 설정 및 미리보기/저장"):
    #     tol = st.number_input("tol 값 (픽셀)", min_value=0, max_value=500, value=60, step=10)

    #     # 1️⃣ 시범 실행 (첫 번째 클린된 파일만)
    #     if st.button("1️⃣ 시범 실행"):
    #         df_preview = preview_first_file(Slab_text_clean, tol)
    #         if df_preview.empty:
    #             st.warning("클린된 폴더에 JSON 파일이 없거나 tol 값이 너무 작습니다")
    #         else:
    #             st.write("### 첫 번째 클린된 파일 미리보기")
    #             st.dataframe(df_preview)

    #     # 📂 전체 엑셀 저장 (클린된 JSON → 엑셀)
    #     if st.button("📂 전체 엑셀 저장"):
    #         saved = save_all_files(Slab_text_clean, Slab_table_excel, tol)  # ← 여기만 변경
    #         if saved:
    #             st.success(f"{len(saved)}개 파일을 '{Slab_table_excel}' 폴더에 저장했습니다")
    #         else:
    #             st.warning("클린된 폴더에 저장할 JSON 파일이 없습니다")





















    with st.expander("🔍 피클 데이터 완전 분석", expanded=False):
        st.markdown("### 피클 파일들의 모든 데이터를 cell_label별로 상세 분석합니다")
        
        # 폴더 경로 입력
        analysis_folder = st.text_input(
            "📁 분석할 피클 파일 폴더 경로",
            placeholder="예: D:/data/pickle_files",
            help="OCR 처리된 표 데이터가 담긴 피클 파일들이 있는 폴더 경로",
            key="analysis_folder_unique"
        )
        
        if analysis_folder:
            analyze_button = st.button("🔍 완전 분석 시작", type="primary", key="analyze_folder_button_unique")
            
            if analyze_button:
                try:
                    # 폴더 존재 확인
                    if not os.path.exists(analysis_folder):
                        st.error(f"❌ 폴더가 존재하지 않습니다: {analysis_folder}")
                        st.stop()
                    
                    # 피클 파일 찾기
                    pickle_files = []
                    for file in os.listdir(analysis_folder):
                        if file.lower().endswith('.pkl'):
                            pickle_files.append(os.path.join(analysis_folder, file))
                    
                    if not pickle_files:
                        st.warning(f"⚠️ 폴더에 피클 파일이 없습니다: {analysis_folder}")
                        st.stop()
                    
                    st.success(f"✅ {len(pickle_files)}개의 피클 파일을 발견했습니다!")
                    
                    # 분석할 파일 선택
                    selected_file = st.selectbox(
                        "분석할 파일 선택",
                        options=pickle_files,
                        format_func=lambda x: os.path.basename(x),
                        key="analysis_file_select"
                    )
                    
                    if selected_file:
                        # 선택된 파일 분석
                        with open(selected_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        if data is not None:
                            regions = data.get('regions', [])
                            st.success(f"✅ 피클 데이터 로드 완료: {len(regions)}개 regions")
                            
                            # 전체 통계
                            total_texts = sum(len(region.get('texts', [])) for region in regions)
                            st.info(f"📊 전체 OCR 텍스트: {total_texts}개")
                            
                            # cell_label별 상세 분석
                            st.markdown("### 📋 cell_label별 상세 분석")
                            
                            for i, region in enumerate(regions):
                                cell_label = region.get('cell_label', f'region_{i}').strip()
                                texts = region.get('texts', [])
                                bounds = region.get('bounds', None)
                                
                                with st.container():
                                    st.markdown(f"#### 🏷️ **{cell_label}** ({len(texts)}개 텍스트)")
                                    
                                    # Region 정보
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if bounds:
                                            st.write(f"📍 영역 bounds: {bounds}")
                                            y_center = (bounds[1] + bounds[3]) / 2
                                            st.write(f"📍 Y 중심: {y_center:.0f}")
                                    
                                    with col2:
                                        st.write(f"📝 텍스트 개수: {len(texts)}")
                                    
                                    # 텍스트 상세 정보
                                    if texts:
                                        st.write("**📝 인식된 텍스트들:**")
                                        
                                        # 테이블 형태로 정리
                                        text_data = []
                                        for j, text_info in enumerate(texts):
                                            bbox = text_info.get('bbox', [0,0,0,0])
                                            text = text_info.get('text', '')
                                            confidence = text_info.get('confidence', 0)
                                            x_center = (bbox[0] + bbox[2]) / 2 if len(bbox) >= 4 else 0
                                            
                                            # 철근 정보인지 확인
                                            rebar_parsed = parse_rebar_info(text)
                                            is_rebar = "✅" if rebar_parsed else "❌"
                                            
                                            text_data.append({
                                                '순번': j+1,
                                                '텍스트': text,
                                                'X중심': f"{x_center:.0f}",
                                                '신뢰도': f"{confidence:.2f}",
                                                '철근정보': is_rebar,
                                                '파싱결과': ', '.join(rebar_parsed) if rebar_parsed else '-'
                                            })
                                        
                                        # DataFrame으로 표시
                                        df_texts = pd.DataFrame(text_data)
                                        st.dataframe(df_texts, use_container_width=True)
                                        
                                        # 철근 정보 통계
                                        rebar_count = sum(1 for text_info in texts if parse_rebar_info(text_info.get('text', '')))
                                        total_parsed = sum(len(parse_rebar_info(text_info.get('text', ''))) for text_info in texts)
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("전체 텍스트", len(texts))
                                        with col2:
                                            st.metric("철근 텍스트", rebar_count)
                                        with col3:
                                            st.metric("파싱된 철근", total_parsed)
                                    
                                    else:
                                        st.warning("텍스트가 없습니다.")
                                    
                                    st.divider()
                            
                            # 전체 요약
                            st.markdown("### 📊 전체 요약")
                            
                            total_rebar_texts = 0
                            total_parsed_rebars = 0
                            summary_data = []
                            
                            for region in regions:
                                cell_label = region.get('cell_label', '').strip()
                                texts = region.get('texts', [])
                                
                                rebar_texts = sum(1 for text_info in texts if parse_rebar_info(text_info.get('text', '')))
                                parsed_rebars = sum(len(parse_rebar_info(text_info.get('text', ''))) for text_info in texts)
                                
                                total_rebar_texts += rebar_texts
                                total_parsed_rebars += parsed_rebars
                                
                                summary_data.append({
                                    'cell_label': cell_label,
                                    '전체_텍스트': len(texts),
                                    '철근_텍스트': rebar_texts,
                                    '파싱된_철근': parsed_rebars
                                })
                            
                            # 요약 테이블
                            df_summary = pd.DataFrame(summary_data)
                            st.dataframe(df_summary, use_container_width=True)
                            
                            # 최종 통계
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("전체 OCR 텍스트", total_texts)
                            with col2:
                                st.metric("철근 관련 텍스트", total_rebar_texts)
                            with col3:
                                st.metric("파싱된 철근 정보", total_parsed_rebars)
                            with col4:
                                efficiency = (total_parsed_rebars / total_texts * 100) if total_texts > 0 else 0
                                st.metric("파싱 효율", f"{efficiency:.1f}%")
                                
                except Exception as e:
                    st.error(f"분석 중 오류: {e}")
        else:
            st.info("👆 분석할 피클 파일 폴더 경로를 입력해주세요")

    # 표 데이터 엑셀 변환 메인 expander

# 표 데이터 엑셀 변환 메인 expander
    with st.expander("📊 표 데이터 엑셀 변환", expanded=False):
        st.markdown("### 피클 파일들을 선택하여 표 형태로 엑셀 파일을 생성합니다")
        
        # 폴더 경로 입력
        excel_folder = st.text_input(
            "📁 변환할 피클 파일 폴더 경로",
            placeholder="예: D:/data/pickle_files",
            help="OCR 처리된 표 데이터가 담긴 피클 파일들이 있는 폴더 경로",
            key="excel_folder_unique"
        )
        
        if excel_folder:
            try:
                # 폴더 존재 확인
                if not os.path.exists(excel_folder):
                    st.error(f"❌ 폴더가 존재하지 않습니다: {excel_folder}")
                    st.stop()
                
                # 피클 파일 찾기
                pickle_files = []
                for file in os.listdir(excel_folder):
                    if file.lower().endswith('.pkl'):
                        pickle_files.append(os.path.join(excel_folder, file))
                
                if not pickle_files:
                    st.warning(f"⚠️ 폴더에 피클 파일이 없습니다: {excel_folder}")
                    st.stop()
                
                st.success(f"✅ {len(pickle_files)}개의 피클 파일을 발견했습니다!")
                
                # 변환할 파일 선택
                selected_file = st.selectbox(
                    "변환할 파일 선택",
                    options=pickle_files,
                    format_func=lambda x: os.path.basename(x),
                    key="excel_file_select"
                )
                
                if selected_file:
                    # 처리 옵션
                    col1, col2 = st.columns(2)
                    with col1:
                        show_details = st.checkbox("🔍 상세 처리 과정 보기", value=False, key="table_excel_details_unique")
                    with col2:
                        process_button = st.button("🚀 변환 시작", type="primary", key="table_excel_process_unique")
                    
                    if process_button:
                        with st.spinner("📋 표 데이터를 처리 중입니다..."):
                            # 선택된 파일을 파일 객체처럼 처리하기 위해 open
                            with open(selected_file, 'rb') as file_obj:
                                # 처리 실행
                                excel_data, df, processing_log = process_table_to_excel(
                                    file_obj, show_details
                                )
                            
                            if excel_data is not None:
                                st.success("🎉 변환이 완료되었습니다!")
                                
                                # 결과 정보 (파싱 통계 추가)
                                st.info(f"📏 최종 데이터 크기: {df.shape[0]} x {df.shape[1]}")
                                
                                # 파싱 통계 정보
                                total_parsed_items = 0
                                parsing_stats = []
                                
                                for index, row in df.iterrows():
                                    non_empty_cells = (row != '').sum()
                                    total_parsed_items += non_empty_cells
                                    parsing_stats.append(f"행 '{index}': {non_empty_cells}개 항목")
                                
                                st.success(f"🎯 총 파싱된 항목: {total_parsed_items}개")
                                
                                # 파싱 상세 통계
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**📊 행별 파싱 통계:**")
                                    for stat in parsing_stats[:10]:  # 처음 10개만
                                        st.write(f"• {stat}")
                                    if len(parsing_stats) > 10:
                                        st.write(f"• ... (총 {len(parsing_stats)}개 행)")
                                
                                with col2:
                                    st.write("**📈 열별 데이터 분포:**")
                                    for col_idx in range(min(10, df.shape[1])):  # 처음 10열만
                                        col_data_count = (df.iloc[:, col_idx] != '').sum()
                                        st.write(f"• 열 {col_idx}: {col_data_count}개")
                                    if df.shape[1] > 10:
                                        st.write(f"• ... (총 {df.shape[1]}개 열)")
                                
                                # 미리보기
                                st.markdown("### 📋 데이터 미리보기")
                                st.dataframe(df.head(10), use_container_width=True)
                                
                                # 상세 처리 과정 표시 (일반 출력으로 변경)
                                if show_details and processing_log:
                                    st.markdown("### 🔍 상세 처리 과정")
                                    # 스크롤 가능한 텍스트 영역으로 표시
                                    log_text = "\n".join(processing_log)
                                    st.text_area("처리 로그", log_text, height=200, key="table_excel_log_unique")
                                
                                # 다운로드 버튼 - 원본 파일명 기반으로 생성
                                base_filename = os.path.splitext(os.path.basename(selected_file))[0]
                                excel_filename = f"{base_filename}_표데이터.xlsx"
                                
                                st.download_button(
                                    label="📥 엑셀 파일 다운로드",
                                    data=excel_data,
                                    file_name=excel_filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="table_excel_download_unique"
                                )
                                
                                # 통계 정보 (일반 출력으로 변경)
                                st.markdown("### 📊 변환 통계")
                                non_empty_cells = (df != '').sum().sum()
                                total_cells = df.shape[0] * df.shape[1]
                                fill_rate = (non_empty_cells / total_cells * 100) if total_cells > 0 else 0
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("전체 셀", total_cells)
                                with col2:
                                    st.metric("데이터 있는 셀", non_empty_cells)
                                with col3:
                                    st.metric("채움률", f"{fill_rate:.1f}%")
                            else:
                                st.error("❌ 변환에 실패했습니다.")
                
            except Exception as e:
                st.error(f"처리 중 오류: {e}")
        
        else:
            st.info("👆 변환할 피클 파일 폴더 경로를 입력해주세요")









    # with st.expander("격자 셀 → Excel 변환"):
    #     # "Excel 변환 시작" 버튼을 누르면 처리 시작
    #     if "show_excel_convert_ui" not in st.session_state:
    #         st.session_state.show_excel_convert_ui = False

    #     if st.button("Excel 변환 시작"):
    #         st.session_state.show_excel_convert_ui = True

    #     if st.session_state.show_excel_convert_ui:
    #         input_folder = Slab_text_clean
    #         output_folder = Slab_table_excel
            
    #         st.write("### 격자 셀 → Excel 변환")
            
    #         # 첫 번째 파일 미리보기
    #         st.write("#### 📋 첫 번째 파일 미리보기:")
    #         preview_df = preview_first_grid_file(input_folder)
            
    #         if not preview_df.empty:
    #             # 구조 분석 - 이 부분만 수정됨
    #             files = [f for f in os.listdir(input_folder) if f.endswith("_grid_cells.json")]
    #             if files:
    #                 with open(os.path.join(input_folder, files[0]), 'r') as f:
    #                     grid_data = json.load(f)  # ← 이름만 변경
                    
    #                 analysis = analyze_grid_structure(grid_data)  # ← grid_data로 전달
                    
    #                 col1, col2 = st.columns(2)
    #                 with col1:
    #                     st.write("**구조 정보:**")
    #                     st.write(f"- 총 셀: {analysis['total_cells']}개")
    #                     st.write(f"- 텍스트 있는 셀: {analysis['filled_cells']}개")
    #                     st.write(f"- 빈 셀: {analysis['empty_cells']}개")
                    
    #                 with col2:
    #                     st.write(f"- 행 범위: {analysis['rows_range'][0]} ~ {analysis['rows_range'][1]}")
    #                     st.write(f"- 열 범위: {analysis['cols_range'][0]} ~ {analysis['cols_range'][1]}")
    #                     st.write(f"- 테이블 크기: {analysis['rows_count']} x {analysis['cols_count']}")
                    
    #                 # 추가 정보 표시 (새로운 기능)
    #                 st.write("**행 라벨 정보:**")
    #                 st.write(f"- 총 행 라벨: {analysis['total_row_labels']}개")
    #                 st.write(f"- 고유 라벨: {analysis['unique_labels']}개")
    #                 st.write(f"- 실제 행 개수: {analysis['rows_count']}개")
    #                 if analysis.get('missing_rows'):
    #                     st.write(f"- 누락된 행: {analysis['missing_rows']}")
                    
    #                 # 라벨 분포 표시 (상위 5개)
    #                 if 'label_distribution' in analysis:
    #                     st.write("**주요 라벨 분포:**")
    #                     label_dist = analysis['label_distribution']
    #                     for label, count in list(label_dist.items())[:5]:
    #                         if label != 'Unknown':
    #                             st.write(f"- {label}: {count}개")
                
    #             st.write("**변환된 테이블:**")
    #             st.dataframe(preview_df)
                
    #             # 전체 변환 버튼
    #             if st.button("🚀 모든 파일 Excel 변환"):
    #                 with st.spinner("변환 중..."):
    #                     saved_paths = save_all_grid_files(input_folder, output_folder)
                    
    #                 st.success(f"✅ {len(saved_paths)}개 파일 변환 완료!")
    #                 for path in saved_paths:
    #                     st.write(f"📄 {os.path.basename(path)}")
    #         else:
    #             st.warning("grid_cells.json 파일을 찾을 수 없습니다!")
            
    #         # "변환 완료" 버튼을 넣어서 누르면 UI 닫힘
    #         if st.button("변환 완료"):
    #             st.session_state.show_excel_convert_ui = False






    # with st.expander("격자 셀별 텍스트 추출"):
    #     # "격자 추출 시작" 버튼을 누르면 처리 시작
    #     if "show_grid_extract_ui" not in st.session_state:
    #         st.session_state.show_grid_extract_ui = False

    #     if st.button("격자 추출 시작"):
    #         st.session_state.show_grid_extract_ui = True

    #     if st.session_state.show_grid_extract_ui:
    #         extract_texts_by_cell_with_proper_labels()  # 이 함수만 위에서 업데이트한 버전으로 바뀜
    #         # "격자 추출 완료" 버튼을 넣어서 누르면 UI 닫힘
    #         if st.button("격자 추출 완료"):
    #             st.session_state.show_grid_extract_ui = False



















    # with st.expander("통합 검출: 선/교차점/동일선상", expanded=True):
    #     img_dir = st.text_input("이미지 폴더", "D:\slab\raw_data_column_label_SDD")
    #     save_dir = st.text_input("저장 폴더 (비우면 미저장)", "D:\slab\Slab_table_region")
    #     boxes_dir = st.text_input("박스 폴더", "D:\slab\Slab_same_line")
    #     ocr_dir = st.text_input("OCR 폴더", "D:\slab\raw_data_OCR_SDD")
    #     mode = st.radio("방식", ["contour", "hough"])
    #     min_line_length = st.slider("min_length", 20, 300, 80, 5)
    #     max_line_gap = st.slider("max_gap", 2, 40, 10, 2)
    #     hough_threshold = st.slider("h_thresh", 30, 300, 100, 5)
    #     morph_kernel_scale = st.slider("morph_scale", 10, 60, 30, 2)
    #     resize_scale = st.slider("resize", 0.3, 1.0, 0.7, 0.05)
    #     block_size = st.slider("blockSize", 7, 31, 15, 2)
    #     C = st.slider("C", -10, 10, -2, 1)
    #     tol = st.slider("tol", 0, 300, 50, 1)
    #     run = st.button("예시 5개 보기")
    #     save = st.button("전체 저장")
    #     if run:
    #         imgs = detect_all_features_batch_Slab_SDD_with_crop(
    #             img_dir, None, boxes_dir, ocr_dir, mode,
    #             min_line_length, max_line_gap, hough_threshold,
    #             morph_kernel_scale, resize_scale, tol, 5, True,
    #             block_size, C
    #         )
    #         if not imgs:
    #             st.warning("이미지 없음")
    #         for fn, im, ints in imgs:
    #             st.subheader(f"{fn}: 교차점 {len(ints)}개")
    #             st.image(im, use_column_width=True)
    #     if save:
    #         detect_all_features_batch_Slab_SDD_with_crop(
    #             img_dir, save_dir, boxes_dir, ocr_dir, mode,
    #             min_line_length, max_line_gap, hough_threshold,
    #             morph_kernel_scale, resize_scale, tol, None, False,
    #             block_size, C
    #         )
    #         st.success(f"저장됨: {save_dir}")




    # with st.expander("👁️ 바운딩박스 시각화"):
    #     if "show_viz_ui" not in st.session_state:
    #         st.session_state.show_viz_ui = False

    #     if st.button("시각화 시작"):
    #         st.session_state.show_viz_ui = True

    #     if st.session_state.show_viz_ui:
    #         visualize_saved_bounding_boxes(selectbox_key="viz_select")
            
    #         if st.button("시각화 완료"):
    #             st.session_state.show_viz_ui = False


    # with st.expander("수평/수직선 + 교차점 인식", expanded=False):
    #     img_dir = st.text_input("입력 이미지 폴더 경로", value=r"D:\slab\Slab_table_region")
    #     save_dir = st.text_input("결과 저장 폴더명 (비우면 저장 X)", value=r"D:\slab\Slab_table_region\lines_detected")
    #     mode = st.radio("선 검출 방식", options=["contour", "hough"], index=0)
    #     min_line_length = st.slider("min_line_length", 20, 300, 80, 5)
    #     max_line_gap = st.slider("max_line_gap", 2, 40, 10, 2)
    #     hough_threshold = st.slider("Hough threshold", 30, 300, 100, 5)
    #     morph_kernel_scale = st.slider("Morph kernel 비율", 10, 60, 30, 2)
    #     resize_scale = st.slider("이미지 축소 비율 (1=원본)", 0.3, 1.0, 0.7, 0.05)
    #     block_size = st.slider("adaptiveThreshold blockSize(홀수)", 7, 31, 15, 2)
    #     C = st.slider("adaptiveThreshold C", -10, 10, -2, 1)
    #     tol = st.slider("교차점 허용 오차 (tol)", 0, 10, 2, 1)

    #     run_btn = st.button("예시 5개 보기")
    #     save_btn = st.button("전체 저장")

    #     if run_btn:
    #         imgs = detect_and_draw_lines_batch_Slab_SDD(
    #             img_dir=img_dir,
    #             save_dir=None,
    #             min_line_length=min_line_length,
    #             max_line_gap=max_line_gap,
    #             hough_threshold=hough_threshold,
    #             morph_kernel_scale=morph_kernel_scale,
    #             resize_scale=resize_scale,
    #             max_examples=5,
    #             return_imgs=True,
    #             mode=mode,
    #             block_size=block_size | 1,
    #             C=C,
    #             tol=tol
    #         )
    #         if not imgs:
    #             st.warning("이미지가 없습니다")
    #         else:
    #             for fname, img, inter_cnt in imgs:
    #                 st.subheader(f"{fname} (교차점: {inter_cnt}개)")
    #                 st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="검출 결과", use_column_width=True)

    #     if save_btn:
    #         detect_and_draw_lines_batch_Slab_SDD(
    #             img_dir=img_dir,
    #             save_dir=save_dir if save_dir else None,
    #             min_line_length=min_line_length,
    #             max_line_gap=max_line_gap,
    #             hough_threshold=hough_threshold,
    #             morph_kernel_scale=morph_kernel_scale,
    #             resize_scale=resize_scale,
    #             max_examples=None,
    #             return_imgs=False,
    #             mode=mode,
    #             block_size=block_size | 1,
    #             C=C,
    #             tol=tol
    #         )
    #         st.success(f"전체 이미지 결과 저장 완료! → '{save_dir}' 폴더 확인")




    # with st.expander("3) 통합 엑셀 파일 생성", expanded=False):
    #     img_dir_cons = st.text_input("입력 이미지 폴더", value=r"D:\slab\Slab_table_region")
    #     save_line_dir_cons = st.text_input("라인 pkl 폴더", value=r"D:\slab\Slab_table_region\lines_detected")
    #     ocr_dir_cons = st.text_input("OCR json 폴더", value=r"D:\slab\Slab_table_crop_OCR")
    #     consolidated_path = st.text_input("통합 엑셀 파일 경로", value=r"D:\slab\Slab_table_region\All_In_One.xlsx")
    #     if st.button("통합 엑셀 추출"):
    #         batch_extract_consolidated_Slab_SDD(
    #             img_dir=img_dir_cons,
    #             save_line_dir=save_line_dir_cons,
    #             ocr_dir=ocr_dir_cons,
    #             consolidated_path=consolidated_path,
    #             mode=mode,
    #             min_line_length=min_line_length,
    #             max_line_gap=max_line_gap,
    #             hough_threshold=hough_threshold,
    #             morph_kernel_scale=morph_kernel_scale,
    #             resize_scale=resize_scale,
    #             block_size=block_size|1,
    #             C=C
    #         )
    #         st.success(f"통합 엑셀 파일 생성 완료 → {consolidated_path}")[


























