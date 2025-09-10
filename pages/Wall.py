#####################################################
##구조도면 벽체 시작

import os
import subprocess
import tempfile
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import mode
from scipy.ndimage import gaussian_filter1d
import shutil
from tqdm import tqdm
from PIL import Image, ImageDraw
import cv2
import glob
import concurrent.futures
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor  # 변경된 부분
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import DBSCAN
import pickle
import os
import random
import joblib
import torch
import numpy as np
from PIL import Image
from collections import Counter
from pdf2image import convert_from_bytes
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
import streamlit as st
import shutil
import json
import os
import subprocess
import json
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import glob
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import cv2
from tqdm import tqdm
import os, glob, json, re, csv
import pandas as pd
import sys, os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app_config import get_base_dir, get_ocr_results_folder

BASE_DIR = get_base_dir()
SURYA_RESULTS_FOLDER =get_ocr_results_folder()





Wall_rawdata_SD = os.path.join(BASE_DIR,"wall", "Wall_raw_data_SD")
Wall_line = os.path.join(BASE_DIR,"wall", "Wall_line")
Wall_column_label = os.path.join(BASE_DIR,"wall", "Wall_column_label")
Wall_table_region =os.path.join(BASE_DIR,"wall", "Wall_table_region")
Wall_OCR =os.path.join(BASE_DIR,"wall", "Wall_OCR")
Wall_cell =os.path.join(BASE_DIR,"wall", "Wall_cell")
Wall_cell_crop_ocr =os.path.join(BASE_DIR,"wall", "Wall_cell_crop_ocr")
STRUCTURED_EXCEL_DIR = os.path.join(BASE_DIR,"wall", "Wall_cell_structure")



os.makedirs(BASE_DIR,exist_ok= True)
os.makedirs(Wall_rawdata_SD, exist_ok=True)
os.makedirs(Wall_column_label, exist_ok=True)
os.makedirs(Wall_table_region, exist_ok=True)
os.makedirs(Wall_OCR, exist_ok=True)
os.makedirs(Wall_cell, exist_ok=True)
os.makedirs(Wall_cell_crop_ocr, exist_ok=True)
os.makedirs(STRUCTURED_EXCEL_DIR, exist_ok=True)




def convert_pdf_to_png(pdf_bytes: bytes) -> list[str]:
    """
    Convert PDF bytes to PNG images using pdf2image.convert_from_bytes.
    Saves each page as a PNG in OUTPUT_FOLDER and returns list of file paths.
    """
    os.makedirs(Wall_rawdata_SD, exist_ok=True)
    images = convert_from_bytes(pdf_bytes, dpi=300)
    paths = []
    for idx, img in enumerate(images, start=1):
        filename = f"page_{idx}.png"
        path = os.path.join(Wall_rawdata_SD, filename)
        img.save(path, "PNG")
        paths.append(path)
    return paths

def draw_and_save_bounding_boxes(canvas_key: str = "box_saver") -> None:
    """
    • RAW_DATA_FOLDER에서 이미지를 선택
    • 선택된 이미지를 주어진 MAX 크기에 맞춰 축소해 캔버스 배경으로 띄우고
    • 사각형을 그려 원본 좌표로 역스케일하여 BOX_COORD_FOLDER에 저장
    """
    os.makedirs(Wall_column_label, exist_ok=True)
    imgs = [f for f in os.listdir(Wall_rawdata_SD) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not imgs:
        st.warning(f"이미지 없음: {Wall_rawdata_SD}")
        return

    # 이미지 선택
    selected = st.selectbox("이미지 선택", imgs, key=canvas_key + "_sel")
    img_path = os.path.join(Wall_rawdata_SD, selected)
    image = Image.open(img_path).convert("RGB")

    # 표시용 축소 계산
    MAX_W, MAX_H = 800, 600
    scale = min(1.0, MAX_W / image.width, MAX_H / image.height)
    disp_w, disp_h = int(image.width * scale), int(image.height * scale)
    img_small = image.resize((disp_w, disp_h), Image.LANCZOS)

    # offset 계산 (캔버스 중앙 정렬 시)
    offset_x = (MAX_W - disp_w) // 2
    offset_y = (MAX_H - disp_h) // 2

    # 캔버스 생성
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

    # 박스 저장 (offset 및 scale 역변환 적용)
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
        out_file = os.path.join(Wall_column_label, f"{os.path.splitext(selected)[0]}_boxes.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(boxes, f, ensure_ascii=False, indent=2)
        st.success(f"✅ {len(boxes)}개 좌표 저장됨: {out_file}")



# def detect_lines_and_intersections():
#     """
#     RAW_DATA_FOLDER에서 이미지 선택 후 가로/세로 선 검출, 교차점 계산 및 시각화
#     """
#     imgs = [f for f in os.listdir(Wall_rawdata_SD) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#     if not imgs:
#         st.warning(f"이미지 없음: {Wall_rawdata_SD}")
#         return
#     selected = st.selectbox("검출할 이미지 선택", imgs, key="line_sel")
#     img_path = os.path.join(Wall_rawdata_SD, selected)
#     pil_img = Image.open(img_path).convert("RGB")
#     img_np = np.array(pil_img)
#     gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
#     horiz, vert = [], []
#     if lines is not None:
#         for l in lines:
#             x1, y1, x2, y2 = l[0]
#             if abs(y1 - y2) < abs(x1 - x2) * 0.1:
#                 horiz.append((x1, y1, x2, y2))
#             if abs(x1 - x2) < abs(y1 - y2) * 0.1:
#                 vert.append((x1, y1, x2, y2))
#     intersections = []
#     for x1, y1, x2, y2 in horiz:
#         for x3, y3, x4, y4 in vert:
#             # 교차점 (x3, y1)
#             intersections.append((x3, y1))
#     overlay = img_np.copy()
#     for x1, y1, x2, y2 in horiz:
#         cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     for x1, y1, x2, y2 in vert:
#         cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     for x, y in intersections:
#         cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
#     st.image(overlay, caption=f"Lines & Intersections ({selected})", use_column_width=True)
#     st.write("교차점 좌표:", intersections)






def apply_surya_ocr_Wall_SD():
    surya_output_folder = Wall_OCR

    os.makedirs(surya_output_folder, exist_ok=True)

    # ✅ 기존 OCR 결과가 있으면 생략
    existing_jsons = [f for f in os.listdir(surya_output_folder) if f.endswith(".json")]
    if existing_jsons:
        st.info(f"ℹ️ 이미 {len(existing_jsons)}개의 OCR 결과가 존재합니다. Surya OCR 생략.")
        return

    # ✅ 입력 이미지 확인
    image_files = [f for f in os.listdir(Wall_rawdata_SD) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        st.error("❌ plain text 이미지가 존재하지 않습니다.")
        return
    st.write("🔍OCR 실행 중...")
    progress_bar = st.progress(0)
    status_text = st.empty()


    for idx, image_file in enumerate(image_files):
        input_path = os.path.join(Wall_rawdata_SD, image_file)
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








def process_all_and_show_one(tol=100):
    img_dir = Wall_rawdata_SD
    ocr_dir = Wall_OCR
    boxes_dir = Wall_column_label
    result_dir = Wall_table_region

    os.makedirs(result_dir, exist_ok=True)
    example_img = None
    example_path = None

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for img_filename in img_files:
        img_path = os.path.join(img_dir, img_filename)
        ocr_path = os.path.join(ocr_dir, f"{os.path.splitext(img_filename)[0]}.json")
        boxes_path = os.path.join(boxes_dir, f"{os.path.splitext(img_filename)[0]}_boxes.json")
        save_path = os.path.join(result_dir, f"highlighted_{img_filename}")

        if not (os.path.exists(img_path) and os.path.exists(ocr_path) and os.path.exists(boxes_path)):
            continue

        img = cv2.imread(img_path)
        with open(boxes_path, 'r', encoding='utf-8') as f:
            boxes = json.load(f)
        with open(ocr_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        ref_xs = [box['left'] + box['width']/2 for box in boxes]

        # ---- OCR 구조 자동 감지 ----
        # case1: {'blocks': [...]}
        if isinstance(ocr_data, dict) and 'blocks' in ocr_data:
            blocks = ocr_data['blocks']
        # case2: {'page_1': [...]}
        elif isinstance(ocr_data, dict) and 'page_1' in ocr_data:
            blocks = ocr_data['page_1']
        # case3: 바로 리스트
        elif isinstance(ocr_data, list):
            blocks = ocr_data
        else:
            continue

        # text_lines key 자동 감지
        for block in blocks:
            text_lines = block.get('text_lines') if isinstance(block, dict) and 'text_lines' in block else []
            for line in text_lines:
                bbox = line.get('bbox')
                if bbox and len(bbox) == 4:
                    x_min, y_min, x_max, y_max = bbox
                    center_x = (x_min + x_max) / 2
                    if any(abs(center_x - ref_x) <= tol for ref_x in ref_xs):
                        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 2)
                        cv2.putText(img, line.get('text', ''), (int(x_min), int(y_min)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
        cv2.imwrite(save_path, img)

        if example_img is None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            example_img = img_rgb
            example_path = save_path

    return example_img, example_path








# def extract_table_cells_from_image(img_path, debug_dir=None):
#     img = cv2.imread(img_path)
#     if img is None:
#         st.error("이미지 로드 실패")
#         return []

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
#     h_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1]//30, 1))
#     v_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0]//30))
#     horizontal = cv2.dilate(cv2.erode(binary.copy(), h_struct), h_struct)
#     vertical = cv2.dilate(cv2.erode(binary.copy(), v_struct), v_struct)
#     mask = cv2.bitwise_and(horizontal, vertical)
#     coords = cv2.findNonZero(mask)
#     if coords is None:
#         st.warning("교차점 없음")
#         return []

#     pts = [tuple(pt[0]) for pt in coords]
#     pts_set = set(pts)

#     # 모든 교차점 조합으로 셀 후보 생성
#     raw_cells = []
#     progress = st.progress(0)
#     total = len(pts)
#     idx = 0
#     for x1, y1 in pts:
#         idx += 1
#         progress.progress(idx / total)
#         for x2, y2 in pts:
#             if x2 > x1 and y2 > y1 and (x1, y2) in pts_set and (x2, y1) in pts_set:
#                 raw_cells.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
#     progress.empty()

#     # 중첩된 큰 셀 제거, 가장 작은 셀만 남김
#     cells = []
#     progress = st.progress(0)
#     total = len(raw_cells)
#     for i, cell in enumerate(raw_cells):
#         progress.progress(i / total)
#         contained = False
#         area_cell = (cell['x2'] - cell['x1']) * (cell['y2'] - cell['y1'])
#         for other in raw_cells:
#             if other is cell:
#                 continue
#             area_other = (other['x2'] - other['x1']) * (other['y2'] - other['y1'])
#             if (other['x1'] <= cell['x1'] <= other['x2'] and other['y1'] <= cell['y1'] <= other['y2']
#                 and other['x2'] >= cell['x2'] and other['y2'] >= cell['y2']
#                 and area_other < area_cell):
#                 contained = True
#                 break
#         if not contained:
#             cells.append(cell)
#     progress.empty()

#     # 교차점 및 셀 시각화
#     if debug_dir:
#         os.makedirs(debug_dir, exist_ok=True)
#         vis = img.copy()
#         for x, y in pts:
#             cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
#         for cell in cells:
#             cv2.rectangle(vis, (cell['x1'], cell['y1']), (cell['x2'], cell['y2']), (0, 255, 0), 2)
#         cv2.imwrite(os.path.join(debug_dir, f"debug_{os.path.basename(img_path)}.png"), vis)

#     return cells

# # 텍스트 포함 셀 크롭 함수
# def run_find_vertical_texts_and_cells_batch(img_path):
#     base = os.path.splitext(os.path.basename(img_path))[0]
#     box_json_path = os.path.join(Wall_column_label, f"{base}_boxes.json")
#     ocr_json_path = os.path.join(Wall_OCR, f"{base}.json")
#     if not (os.path.exists(box_json_path) and os.path.exists(ocr_json_path)):
#         st.warning("box 또는 ocr json을 찾을 수 없습니다.")
#         return 0

#     with open(box_json_path, encoding='utf-8') as f:
#         boxes = json.load(f)
#     with open(ocr_json_path, encoding='utf-8') as f:
#         ocr = json.load(f)

#     text_lines = []
#     for page in ocr.values():
#         for line in page:
#             for t in line['text_lines']:
#                 text_lines.append({'bbox': t['bbox']})

#     cells = extract_table_cells_from_image(img_path)
#     image = cv2.imread(img_path)

#     count = 0
#     progress = st.progress(0)
#     total = len(boxes)
#     for idx, b in enumerate(boxes):
#         progress.progress((idx+1) / total)
#         x_c = b['left'] + b['width'] / 2
#         tlines = [t for t in text_lines if t['bbox'][0] <= x_c <= t['bbox'][2]]
#         if not tlines:
#             continue
#         t = tlines[0]
#         cx = (t['bbox'][0] + t['bbox'][2]) / 2
#         cy = (t['bbox'][1] + t['bbox'][3]) / 2

#         best = None
#         min_area = float('inf')
#         for cell in cells:
#             if cell['x1'] <= cx <= cell['x2'] and cell['y1'] <= cy <= cell['y2']:
#                 area = (cell['x2'] - cell['x1']) * (cell['y2'] - cell['y1'])
#                 if area < min_area:
#                     min_area = area
#                     best = cell
#         if best:
#             x1, y1, x2, y2 = best['x1'], best['y1'], best['x2'], best['y2']
#             crop = image[y1:y2, x1:x2]
#             cv2.imwrite(os.path.join(Wall_cell, f"{base}_cell_{idx}.png"), crop)
#             count += 1
#     progress.empty()
#     return count








def visualize_lines_batch():
    IMG_DIR = Wall_rawdata_SD  # 처리할 이미지 폴더 경로
    DEBUG_DIR =Wall_cell    # 디버그 결과 저장 폴더
    BOXES_DIR =Wall_column_label   # boxes json 폴더 경로
    OCR_DIR = Wall_OCR        # OCR json 폴더 경로
    img_files = [f for f in os.listdir(IMG_DIR)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        st.warning("처리할 이미지가 IMG_DIR에 없습니다.")
        return

    for fname in img_files:
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(IMG_DIR, fname)
        boxes_path = os.path.join(BOXES_DIR, f"{base}_boxes.json")
        ocr_path = os.path.join(OCR_DIR, f"{base}.json")

        img = cv2.imread(img_path)
        if img is None:
            st.error(f"이미지 로드 실패: {img_path}")
            continue

        # 그레이스케일 및 이진화
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(~gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 15, -2)

        # 수평 선 검출
        horizontal = binary.copy()
        horizontalsize = max(1, horizontal.shape[1] // 30)
        hor_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
        horizontal = cv2.erode(horizontal, hor_structure)
        horizontal = cv2.dilate(horizontal, hor_structure)

        # 수직 선 검출
        vertical = binary.copy()
        verticalsize = max(1, vertical.shape[0] // 30)
        ver_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        vertical = cv2.erode(vertical, ver_structure)
        vertical = cv2.dilate(vertical, ver_structure)

        # 교차점 검출
        mask = cv2.bitwise_and(horizontal, vertical)
        pts = cv2.findNonZero(mask)

        # 시각화
        vis = img.copy()
        # 수평 컨투어 그리기
        contours_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_h:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.line(vis, (x, y + h // 2), (x + w, y + h // 2), (0, 255, 0), 2)
        # 수직 컨투어 그리기
        contours_v, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_v:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.line(vis, (x + w // 2, y), (x + w // 2, y + h), (255, 0, 0), 2)
        # 교차점 시각화
        intersect_count = 0
        if pts is not None:
            for p in pts:
                x_i, y_i = p[0]
                cv2.circle(vis, (x_i, y_i), 5, (0, 0, 255), -1)
            intersect_count = len(pts)

                # 타겟 텍스트 시각화
        text_count = 0
        if os.path.exists(boxes_path) and os.path.exists(ocr_path):
            with open(boxes_path, encoding='utf-8') as f:
                boxes = json.load(f)
            with open(ocr_path, encoding='utf-8') as f:
                ocr = json.load(f)
            # 전체 텍스트 라인 추출
            text_lines = []
            for page in ocr.values():
                for line in page:
                    for t in line['text_lines']:
                        text_lines.append({'text': t['text'], 'bbox': t['bbox']})
            # boxes 기준 x축으로 필터링
            for bx in boxes:
                x_center = bx['left'] + bx['width']/2
                # 일치하는 텍스트 찾기
                for tl in text_lines:
                    x1, y1, x2, y2 = tl['bbox']
                    if x1 <= x_center <= x2:
                        # 텍스트 박스와 문자열 표시 (정수로 변환)
                        pt1 = (int(x1), int(y1))
                        pt2 = (int(x2), int(y2))
                        cv2.rectangle(vis, pt1, pt2, (255, 255, 0), 2)
                        text_pos = (int(x1), max(0, int(y1)-10))
                        cv2.putText(vis, tl['text'], text_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        text_count += 1

        # Streamlit 출력
        st.subheader(f"{fname} - 수평선 {len(contours_h)}개, 수직선 {len(contours_v)}개, 교차점 {intersect_count}개, 텍스트 {text_count}개 검출")
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=fname)

        # 디버그 저장
        if DEBUG_DIR:
            out_dir = os.path.join(DEBUG_DIR, 'visualized')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"vis_{fname}")
            out_path = os.path.join(out_dir, f"vis_{fname}.png")
            cv2.imwrite(out_path, vis)
            st.write(f"저장됨: {out_path}")

# Streamlit UI
        st.subheader(f"{fname} - 수평선 {len(contours_h)}개, 수직선 {len(contours_v)}개, 교차점 {intersect_count}개, 텍스트 {text_count}개 검출")
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=fname)

        # 디버그 저장
        if DEBUG_DIR:
            out_dir = os.path.join(DEBUG_DIR, 'visualized')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"vis_{fname}")
            cv2.imwrite(out_path, vis)
            st.write(f"저장됨: {out_path}")

#-----------------------------------------------------------




# def detect_intersections(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     binary = cv2.adaptiveThreshold(~gray, 255,
#                                    cv2.ADAPTIVE_THRESH_MEAN_C,
#                                    cv2.THRESH_BINARY, 15, -2)
#     hsize = max(1, binary.shape[1]//30)
#     vsize = max(1, binary.shape[0]//30)
#     hor_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (hsize,1))
#     vert_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1,vsize))
#     horizontal = cv2.dilate(cv2.erode(binary, hor_struct), hor_struct)
#     vertical   = cv2.dilate(cv2.erode(binary, vert_struct), vert_struct)
#     mask = cv2.bitwise_and(horizontal, vertical)
#     pts = cv2.findNonZero(mask)
#     if pts is None:
#         return [], horizontal, vertical
#     intersections = [tuple(p[0]) for p in pts]
#     return intersections, horizontal, vertical

# # 2) OCR 텍스트 로드
# def load_text_boxes(base, BOXES_DIR, OCR_DIR):
#     boxes_path = os.path.join(BOXES_DIR, f"{base}_boxes.json")
#     ocr_path   = os.path.join(OCR_DIR, f"{base}.json")
#     if not os.path.exists(boxes_path) or not os.path.exists(ocr_path):
#         return []
#     with open(boxes_path, encoding='utf-8') as f:
#         boxes = json.load(f)
#     with open(ocr_path, encoding='utf-8') as f:
#         ocr = json.load(f)
#     text_centers = []
#     for b in boxes:
#         x_center = b['left'] + b['width']/2
#         for page in ocr.values():
#             for line in page:
#                 for t in line['text_lines']:
#                     x1,y1,x2,y2 = t['bbox']
#                     if x1 <= x_center <= x2:
#                         cx = (x1+x2)/2; cy = (y1+y2)/2
#                         text_centers.append({'text': t['text'], 'center': (int(cx), int(cy))})
#     return text_centers

# # 3) 텍스트를 감싸는 최소 사각형 찾기
# def find_min_rectangles(intersections, horizontal, vertical, text_centers):
#     rects = []
#     pts_set = set(intersections)
#     xs = sorted({x for x,y in intersections})
#     ys = sorted({y for x,y in intersections})
#     for tc in text_centers:
#         cx, cy = tc['center']
#         best = None
#         min_area = float('inf')
#         for i in range(len(xs)):
#             for j in range(i+1, len(xs)):
#                 xl, xr = xs[i], xs[j]
#                 if not (xl <= cx <= xr): continue
#                 for u in range(len(ys)):
#                     for v in range(u+1, len(ys)):
#                         yt, yb = ys[u], ys[v]
#                         if not (yt <= cy <= yb): continue
#                         if (xl, yt) in pts_set and (xl, yb) in pts_set and (xr, yt) in pts_set and (xr, yb) in pts_set:
#                             top = horizontal[yt, xl:xr+1]
#                             bot = horizontal[yb, xl:xr+1]
#                             left = vertical[yt:yb+1, xl]
#                             right= vertical[yt:yb+1, xr]
#                             if top.all() and bot.all() and left.all() and right.all():
#                                 area = (xr-xl)*(yb-yt)
#                                 if area < min_area:
#                                     min_area = area
#                                     best = (xl, yt, xr, yb)
#         if best:
#             rects.append({'text': tc['text'], 'rect': best})
#     return rects

# # 4) 배치 처리 및 크롭
# def process_batch():
#     # 입출력 경로 함수 내 정의
#     IMG_DIR = Wall_rawdata_SD          # 처리할 이미지 폴더
#     BOXES_DIR = Wall_column_label     # boxes json 폴더 경로
#     OCR_DIR = Wall_OCR                # OCR json 폴더 경로
#     CELL_OUT_DIR = Wall_table_region  # 크롭된 셀 저장 폴더

#     img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]
#     if not img_files:
#         st.warning("처리할 이미지가 없습니다.")
#         return

#     progress = st.progress(0)
#     total = len(img_files)

#     for idx, fname in enumerate(img_files):
#         img_path = os.path.join(IMG_DIR, fname)
#         img = cv2.imread(img_path)
#         if img is None:
#             st.error(f"이미지 로드 실패: {img_path}")
#             progress.progress((idx+1)/total)
#             continue

#         intersections, hor, ver = detect_intersections(img)
#         text_centers = load_text_boxes(os.path.splitext(fname)[0], BOXES_DIR, OCR_DIR)
#         rects = find_min_rectangles(intersections, hor, ver, text_centers)

#         vis = img.copy()
#         for x,y in intersections:
#             cv2.circle(vis, (x,y), 4, (0,0,255), -1)
#         for r in rects:
#             x1,y1,x2,y2 = r['rect']
#             cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
#             cv2.putText(vis, r['text'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

#         st.subheader(f"{fname} - Rects: {len(rects)}")
#         st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=fname)

#         os.makedirs(CELL_OUT_DIR, exist_ok=True)
#         for j, r in enumerate(rects):
#             x1,y1,x2,y2 = r['rect']
#             crop = img[y1:y2, x1:x2]
#             outp = os.path.join(CELL_OUT_DIR, f"{os.path.splitext(fname)[0]}_cell_{j}.png")
#             cv2.imwrite(outp, crop)

#         progress.progress((idx+1)/total)
# 1) 교차점 검출
# 0) 컬럼 박스 불러오기
def load_column_boxes(base, BOXES_DIR):
    boxes_path = os.path.join(BOXES_DIR, f"{base}_boxes.json")
    if not os.path.exists(boxes_path):
        return []
    with open(boxes_path, encoding='utf-8') as f:
        return json.load(f)

# 0) 인접 박스 간 수평 거리 평균 계산
def compute_avg_offset(column_boxes):
    if len(column_boxes) < 2:
        return 0
    # left 값 기준 오름차순 정렬
    sorted_boxes = sorted(column_boxes, key=lambda b: b['left'])
    lefts = [b['left'] for b in sorted_boxes]
    # 인접 간 차이 계산
    diffs = [lefts[i+1] - lefts[i] for i in range(len(lefts)-1)]
    return int(sum(diffs) / len(diffs))

# 1) 교차점 검출
def detect_intersections(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        ~gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, -2
    )
    hsize = max(1, binary.shape[1] // 30)
    vsize = max(1, binary.shape[0] // 30)
    hor_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (hsize, 1))
    vert_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vsize))
    horizontal = cv2.dilate(cv2.erode(binary, hor_struct), hor_struct)
    vertical   = cv2.dilate(cv2.erode(binary, vert_struct), vert_struct)
    mask = cv2.bitwise_and(horizontal, vertical)
    pts = cv2.findNonZero(mask)
    if pts is None:
        return [], horizontal, vertical
    intersections = [tuple(p[0]) for p in pts]
    return intersections, horizontal, vertical

# 2) OCR 텍스트 로드
def load_text_boxes(base, BOXES_DIR, OCR_DIR):
    boxes_path = os.path.join(BOXES_DIR, f"{base}_boxes.json")
    ocr_path   = os.path.join(OCR_DIR, f"{base}.json")
    if not os.path.exists(boxes_path) or not os.path.exists(ocr_path):
        return []
    with open(boxes_path, encoding='utf-8') as f:
        boxes = json.load(f)
    with open(ocr_path, encoding='utf-8') as f:
        ocr = json.load(f)

    centers = []
    for b in boxes:
        x_center = b['left'] + b['width'] / 2
        for page in ocr.values():
            for line in page:
                for t in line['text_lines']:
                    x1, y1, x2, y2 = t['bbox']
                    if x1 <= x_center <= x2:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        centers.append({'text': t['text'], 'center': (cx, cy)})
    return centers

# 3) 텍스트를 감싸는 최소 사각형 찾기
def find_min_rectangles(intersections, horizontal, vertical, text_centers):
    rects = []
    pts_set = set(intersections)
    xs = sorted({x for x, y in intersections})
    ys = sorted({y for x, y in intersections})

    for tc in text_centers:
        cx, cy = tc['center']
        best = None
        min_area = float('inf')
        for i in range(len(xs)):
            for j in range(i+1, len(xs)):
                xl, xr = xs[i], xs[j]
                if not (xl <= cx <= xr): continue
                for u in range(len(ys)):
                    for v in range(u+1, len(ys)):
                        yt, yb = ys[u], ys[v]
                        if not (yt <= cy <= yb): continue
                        corners = {(xl, yt), (xl, yb), (xr, yt), (xr, yb)}
                        if corners.issubset(pts_set):
                            if (horizontal[yt, xl:xr+1].all() and
                                horizontal[yb, xl:xr+1].all() and
                                vertical[yt:yb+1, xl].all() and
                                vertical[yt:yb+1, xr].all()):
                                area = (xr - xl) * (yb - yt)
                                if area < min_area:
                                    min_area = area
                                    best = (xl, yt, xr, yb)
        if best:
            rects.append({'text': tc['text'], 'rect': best})
    return rects

# 4) 이미지 하나 처리 함수
def process_image(fname, IMG_DIR, BOXES_DIR, OCR_DIR, DEBUG_DIR, CELL_OUT_DIR, margin=5):  # margin 파라미터 추가
    base = os.path.splitext(fname)[0]
    img_path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        return fname, None, 0

    # 테이블 그리드 검출 & OCR 매칭
    intersections, hor, ver = detect_intersections(img)
    text_centers = load_text_boxes(base, BOXES_DIR, OCR_DIR)
    rects = find_min_rectangles(intersections, hor, ver, text_centers)

    # 평균 컬럼 간격 계산
    column_boxes = load_column_boxes(base, BOXES_DIR)
    avg_offset   = compute_avg_offset(column_boxes)

    # 시각화(Debug) 저장
    vis = img.copy()
    for x, y in intersections:
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
    for r in rects:
        x1, y1, x2, y2 = r['rect']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, r['text'], (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"vis_{fname}.png"), vis)

    # 셀 + 확장 영역 크롭 저장
    os.makedirs(CELL_OUT_DIR, exist_ok=True)
    for idx, r in enumerate(rects):
        x1, y1, x2, y2 = r['rect']
        cell_w = x2 - x1
        ext = max(0, avg_offset - cell_w)
        x2_ext = min(x2 + ext, img.shape[1] - 1)

        # 여기서 마진 적용!
        x1_crop = max(x1 - margin, 0)
        y1_crop = max(y1 - margin, 0)
        x2_crop = min(x2_ext + margin, img.shape[1] - 1)
        y2_crop = min(y2 + margin, img.shape[0] - 1)

        crop = img[y1_crop:y2_crop, x1_crop:x2_crop]  # 마진 적용된 크롭

        text_label = r['text'].replace(' ', '_')
        out_name = f"{base}_{text_label}_{idx}.png"
        cv2.imwrite(os.path.join(CELL_OUT_DIR, out_name), crop)

    return fname, vis, len(rects)

# 5) 배치 처리 함수
def process_batch():
    IMG_DIR      = Wall_rawdata_SD
    DEBUG_DIR    = Wall_cell
    BOXES_DIR    = Wall_column_label
    OCR_DIR      = Wall_OCR
    CELL_OUT_DIR = Wall_table_region

    img_files = [f for f in os.listdir(IMG_DIR)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        st.warning("처리할 이미지가 없습니다.")
        return

    total = len(img_files)
    progress = st.progress(0)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_image, fname,
                            IMG_DIR, BOXES_DIR, OCR_DIR,
                            DEBUG_DIR, CELL_OUT_DIR): fname
            for fname in img_files
        }
        with st.spinner("처리 중입니다, 잠시만 기다려주세요..."):
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                fname, vis, cnt = future.result()
                if vis is not None:
                    st.subheader(f"{fname} - Rects: {cnt}")
                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                             caption=fname)
                progress.progress((i+1)/total)
                time.sleep(0)








def apply_surya_ocr_Wall_Crop_SD():

    surya_output_folder = Wall_cell_crop_ocr

    os.makedirs(surya_output_folder, exist_ok=True)

    # ✅ 기존 OCR 결과가 있으면 생략
    existing_jsons = [f for f in os.listdir(surya_output_folder) if f.endswith(".json")]
    if existing_jsons:
        st.info(f"ℹ️ 이미 {len(existing_jsons)}개의 OCR 결과가 존재합니다. Surya OCR 생략.")
        return

    # ✅ 입력 이미지 확인
    image_files = [f for f in os.listdir(Wall_table_region) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        st.error("❌ plain text 이미지가 존재하지 않습니다.")
        return
    st.write("🔍OCR 실행 중...")
    progress_bar = st.progress(0)
    status_text = st.empty()


    for idx, image_file in enumerate(image_files):
        input_path = os.path.join(Wall_table_region, image_file)
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
# def detect_and_draw_lines_batch(
#     img_dir, save_dir=None,
#     min_line_length=80, max_line_gap=10,
#     hough_threshold=100, morph_kernel_scale=30,
#     resize_scale=0.7,
#     max_examples=5, return_imgs=False,
#     mode="contour",
#     block_size=15, C=-2
# ):
#     img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     out_imgs = []
#     for i, img_file in enumerate(img_files):
#         if (max_examples is not None) and (i >= max_examples):
#             break
#         img_path = os.path.join(img_dir, img_file)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         # 1. 리사이즈 (축소)
#         if resize_scale != 1.0:
#             img_small = cv2.resize(img, dsize=None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
#         else:
#             img_small = img.copy()
#         gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
#         binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
#         horizontalsize = max(1, binary.shape[1] // morph_kernel_scale)
#         verticalsize = max(1, binary.shape[0] // morph_kernel_scale)
#         hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
#         ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
#         horizontal = cv2.erode(binary, hor_kernel)
#         horizontal = cv2.dilate(horizontal, hor_kernel)
#         vertical = cv2.erode(binary, ver_kernel)
#         vertical = cv2.dilate(vertical, ver_kernel)

#         result_img = img_small.copy()

#         if mode == "hough":
#             # ---- HoughLinesP 방식 (이전과 동일) ----
#             lines_h = cv2.HoughLinesP(horizontal, 1, np.pi/180, hough_threshold,
#                                       minLineLength=min_line_length, maxLineGap=max_line_gap)
#             if lines_h is not None:
#                 for l in lines_h:
#                     x1, y1, x2, y2 = l[0]
#                     cv2.line(result_img, (x1, y1), (x2, y2), (255,0,0), 2)
#             lines_v = cv2.HoughLinesP(vertical, 1, np.pi/180, hough_threshold,
#                                       minLineLength=min_line_length, maxLineGap=max_line_gap)
#             if lines_v is not None:
#                 for l in lines_v:
#                     x1, y1, x2, y2 = l[0]
#                     cv2.line(result_img, (x1, y1), (x2, y2), (0,255,0), 2)
#         else:
#             # ---- Contour 기반 방식 ----
#             contours_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             for cnt in contours_h:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 if w > min_line_length and h < (binary.shape[0] // 10):
#                     cv2.line(result_img, (x, y + h // 2), (x + w, y + h // 2), (255, 0, 0), 2)
#             contours_v, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             for cnt in contours_v:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 if h > min_line_length and w < (binary.shape[1] // 10):
#                     cv2.line(result_img, (x + w // 2, y), (x + w // 2, y + h), (0, 255, 0), 2)

#         # 2. 교차점 검출 및 표시
#         mask = cv2.bitwise_and(horizontal, vertical)
#         pts = cv2.findNonZero(mask)
#         intersect_count = 0
#         if pts is not None:
#             for p in pts:
#                 x_i, y_i = p[0]
#                 cv2.circle(result_img, (x_i, y_i), 4, (0,0,255), -1) # 빨간색 교차점
#             intersect_count = len(pts)

#         # 저장 및 반환
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             out_path = os.path.join(save_dir, f"lines_{img_file}")
#             cv2.imwrite(out_path, result_img)
#         if return_imgs:
#             out_imgs.append((img_file, result_img, intersect_count))
#     return out_imgs if return_imgs else None











# def batch_ocr_json_to_excel(margin=5):
#     # 경로를 함수 안에서 지정 (원하는 폴더명으로 바꿔도 됨)
#     input_folder = 'D:\wall_drawing\Wall_cell_crop_ocr'
#     output_folder = './Excel_Results'
#     os.makedirs(output_folder, exist_ok=True)

#     files = [f for f in os.listdir(input_folder) if f.lower().endswith('.json')]
#     if not files:
#         print('입력 폴더에 json 파일이 없습니다.')
#         return
#     for fname in files:
#         input_json_path = os.path.join(input_folder, fname)
#         output_excel_path = os.path.join(
#             output_folder, os.path.splitext(fname)[0] + '.xlsx'
#         )
#         print(f"처리 중: {fname}")
#         try:
#             ocr_json_to_excel(input_json_path, output_excel_path, margin)
#         except Exception as e:
#             print(f"[에러] {fname} 처리 실패:", e)
#     print(f"\n전체 {len(files)}개 파일 처리 완료!")

# def ocr_json_to_excel(input_json_path, output_excel_path='output.xlsx', margin=5):
#     with open(input_json_path, encoding='utf-8') as f:
#         ocr_data = json.load(f)
#     ocr_boxes = []
#     for page in ocr_data.values():
#         for textline in page[0]['text_lines']:
#             text = textline['text'].strip()
#             bbox = textline['bbox']
#             ocr_boxes.append({'text': text, 'rect': bbox})
#     floor_keywords = ['F', 'B', 'PIT']
#     floor_boxes = [
#         box for box in ocr_boxes if any(kw in box['text'] for kw in floor_keywords)
#     ]
#     floor_boxes = sorted(floor_boxes, key=lambda b: b['rect'][1])
#     sorted_by_x = sorted(ocr_boxes, key=lambda b: b['rect'][0])
#     col_xs, col_names = [], []
#     col_num = 0
#     for box in sorted_by_x:
#         x1 = box['rect'][0]
#         if all(abs(x1 - cx) > 10 for cx in col_xs):
#             col_xs.append(x1)
#             col_names.append(chr(65+col_num))
#             col_num += 1
#     for box in ocr_boxes:
#         x1 = box['rect'][0]
#         idx = min(range(len(col_xs)), key=lambda i: abs(col_xs[i] - x1))
#         box['col'] = col_names[idx]
#     extracted_rows = []
#     for floor_box in floor_boxes:
#         fy1, fy2 = floor_box['rect'][1], floor_box['rect'][3]
#         row = {floor_box['col']: floor_box['text']}
#         for box in ocr_boxes:
#             if box == floor_box:
#                 continue
#             by1, by2 = box['rect'][1], box['rect'][3]
#             if (by2 > fy1 - margin) and (by1 < fy2 + margin):
#                 row[box['col']] = box['text']
#         extracted_rows.append(row)
#     all_cols = sorted(list(set(col_names)))
#     df = pd.DataFrame(extracted_rows)
#     df = df.reindex(columns=all_cols)
#     df.to_excel(output_excel_path, index=False)
#     print(f"[저장 완료] {os.path.abspath(output_excel_path)}")
#     return df





# def detect_and_draw_lines_batch(
#     img_dir, save_dir=None,
#     min_line_length=80, max_line_gap=10,
#     hough_threshold=100, morph_kernel_scale=30,
#     resize_scale=0.7,
#     max_examples=5, return_imgs=False,
#     mode="contour",
#     block_size=15, C=-2,
#     tol=2  # ↔ 교차점 허용 오차 (픽셀)
# ):
#     img_files = [f for f in os.listdir(img_dir)
#                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     out_imgs = []
#     for i, img_file in enumerate(img_files):
#         if max_examples is not None and i >= max_examples:
#             break
#         img_path = os.path.join(img_dir, img_file)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         # 1. 리사이즈 + 이진화
#         if resize_scale != 1.0:
#             img_small = cv2.resize(img, None, fx=resize_scale, fy=resize_scale,
#                                    interpolation=cv2.INTER_AREA)
#         else:
#             img_small = img.copy()
#         gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
#         binary = cv2.adaptiveThreshold(
#             ~gray, 255,
#             cv2.ADAPTIVE_THRESH_MEAN_C,
#             cv2.THRESH_BINARY,
#             int(block_size) | 1, C
#         )

#         # 2. 수평/수직 선 추출
#         hs = max(1, binary.shape[1] // morph_kernel_scale)
#         vs = max(1, binary.shape[0] // morph_kernel_scale)
#         hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hs, 1))
#         ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vs))
#         horizontal = cv2.dilate(cv2.erode(binary, hor_kernel), hor_kernel)
#         vertical   = cv2.dilate(cv2.erode(binary, ver_kernel), ver_kernel)

#         result_img = img_small.copy()

#         # 3. 선 그리기 (기존 Hough / contour)
#         if mode == "hough":
#             lines_h = cv2.HoughLinesP(horizontal, 1, np.pi/180, hough_threshold,
#                                       minLineLength=min_line_length, maxLineGap=max_line_gap)
#             if lines_h is not None:
#                 for l in lines_h:
#                     x1, y1, x2, y2 = l[0]
#                     cv2.line(result_img, (x1, y1), (x2, y2), (255,0,0), 2)
#             lines_v = cv2.HoughLinesP(vertical, 1, np.pi/180, hough_threshold,
#                                       minLineLength=min_line_length, maxLineGap=max_line_gap)
#             if lines_v is not None:
#                 for l in lines_v:
#                     x1, y1, x2, y2 = l[0]
#                     cv2.line(result_img, (x1, y1), (x2, y2), (0,255,0), 2)
#         else:
#             contours_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL,
#                                              cv2.CHAIN_APPROX_SIMPLE)
#             for cnt in contours_h:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 if w > min_line_length and h < (binary.shape[0] // 10):
#                     cv2.line(result_img, (x, y + h//2), (x + w, y + h//2),
#                              (255, 0, 0), 2)
#             contours_v, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL,
#                                              cv2.CHAIN_APPROX_SIMPLE)
#             for cnt in contours_v:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 if h > min_line_length and w < (binary.shape[1] // 10):
#                     cv2.line(result_img, (x + w//2, y), (x + w//2, y + h),
#                              (0, 255, 0), 2)

#         # 4. 교차점 검출: tol 픽셀 이내의 선도 연결해서 검출
#         #    4.1 수평선/수직선 마스크 팽창
#         kh = np.ones((1, 2*tol+1), np.uint8)
#         kv = np.ones((2*tol+1, 1), np.uint8)
#         hor_dil = cv2.dilate(horizontal, kh)
#         ver_dil = cv2.dilate(vertical, kv)
#         #    4.2 팽창된 마스크 AND
#         mask = cv2.bitwise_and(hor_dil, ver_dil)
#         pts = cv2.findNonZero(mask)

#         # 5. 교차점 시각화 및 리스트 축적
#         intersect_count = 0
#         intersections = []
#         if pts is not None:
#             intersect_count = len(pts)
#             for p in pts:
#                 x_i, y_i = p[0]
#                 cv2.circle(result_img, (x_i, y_i), 4, (0,0,255), -1)
#                 intersections.append((int(x_i), int(y_i)))

#         # 6. 결과 저장 (pkl + 이미지)
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             base = os.path.splitext(img_file)[0]
#             with open(os.path.join(save_dir, f"{base}_lines.pkl"), "wb") as f:
#                 pickle.dump({
#                     "horizontal": horizontal,
#                     "vertical":   vertical,
#                     "intersections": intersections
#                 }, f)
#             cv2.imwrite(os.path.join(save_dir, f"lines_{img_file}"), result_img)

#         if return_imgs:
#             out_imgs.append((img_file, result_img, intersect_count))

#     return out_imgs if return_imgs else None
def detect_and_draw_lines_batch(
    img_dir, save_dir=None,
    min_line_length=80, max_line_gap=10,
    hough_threshold=100, morph_kernel_scale=30,
    resize_scale=0.7,
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

        # 1. 리사이즈 + 이진화
        if resize_scale != 1.0:
            img_small = cv2.resize(img, None, fx=resize_scale, fy=resize_scale,
                                   interpolation=cv2.INTER_AREA)
        else:
            img_small = img.copy()
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
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

        result_img = img_small.copy()

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

        # 6. 결과 저장 (원본 해상도로 보정)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.splitext(img_file)[0]
            h0, w0 = img.shape[:2]

            if resize_scale != 1.0:
                horizontal_full = cv2.resize(horizontal, (w0, h0), interpolation=cv2.INTER_NEAREST)
                vertical_full = cv2.resize(vertical, (w0, h0), interpolation=cv2.INTER_NEAREST)
                intersections_full = [(int(x/resize_scale), int(y/resize_scale))
                                      for x, y in intersections]
            else:
                horizontal_full = horizontal
                vertical_full = vertical
                intersections_full = intersections

            with open(os.path.join(save_dir, f"{base}_lines.pkl"), "wb") as f:
                pickle.dump({
                    "horizontal": horizontal_full,
                    "vertical":   vertical_full,
                    "intersections": intersections_full
                }, f)

            cv2.imwrite(os.path.join(save_dir, f"lines_{img_file}"), result_img)

        if return_imgs:
            out_imgs.append((img_file, result_img, len(intersections)))

    return out_imgs if return_imgs else None


# ---------- (2) 표 셀 추출/엑셀화 ----------

def find_min_rectangles(intersections, horizontal, vertical, text_centers):
    rects = []
    pts_set = set(intersections)
    xs = sorted({x for x, y in intersections})
    ys = sorted({y for x, y in intersections})
    for tc in text_centers:
        cx, cy = tc['center']
        best = None
        min_area = float('inf')
        for i in range(len(xs)):
            for j in range(i+1, len(xs)):
                xl, xr = xs[i], xs[j]
                if not (xl <= cx <= xr):
                    continue
                for u in range(len(ys)):
                    for v in range(u+1, len(ys)):
                        yt, yb = ys[u], ys[v]
                        if not (yt <= cy <= yb):
                            continue
                        corners = {(xl, yt), (xl, yb), (xr, yt), (xr, yb)}
                        if corners.issubset(pts_set):
                            if (horizontal[yt, xl:xr+1].all() and
                                horizontal[yb, xl:xr+1].all() and
                                vertical[yt:yb+1, xl].all() and
                                vertical[yt:yb+1, xr].all()):
                                area = (xr - xl) * (yb - yt)
                                if area < min_area:
                                    min_area = area
                                    best = (xl, yt, xr, yb)
        if best:
            rects.append({'text': tc['text'], 'rect': best})
    return rects


# -------------------------------------------------------------
# 3) x축 클러스터링 열 인덱스 할당
# -------------------------------------------------------------
def assign_col_idx_by_cx(rects, margin=None):
    cxs = sorted([(r['rect'][0] + r['rect'][2]) / 2 for r in rects])
    if margin is None and len(cxs) > 1:
        diffs = [cxs[i+1] - cxs[i] for i in range(len(cxs)-1)]
        margin = (sum(diffs) / len(diffs)) / 2
    margin = margin or 10

    clusters = []
    for x in cxs:
        if not clusters or abs(x - clusters[-1]) > margin:
            clusters.append(x)
        else:
            clusters[-1] = (clusters[-1] + x) / 2

    for r in rects:
        cx = (r['rect'][0] + r['rect'][2]) / 2
        idx = int(np.argmin([abs(cx - c) for c in clusters]))
        r['col'] = idx

    return clusters


# -------------------------------------------------------------
# 4) rects 기반 표 추출/엑셀화
# -------------------------------------------------------------
def extract_table_dynamic(base, ocr_dir, line_dir):
    # (기존 extract_table_dynamic 의 1~6단계까지 동일하게 수행하고 DataFrame 반환)
    # 1) OCR centers
    with open(os.path.join(ocr_dir, f"{base}.json"), encoding='utf-8') as f:
        ocr = json.load(f)
    centers = []
    for page in ocr.values():
        for ln in page[0]['text_lines']:
            x1,y1,x2,y2 = ln['bbox']
            centers.append({'text':ln['text'].strip(),
                            'center':((x1+x2)/2,(y1+y2)/2)})
    # 2) load lines.pkl
    with open(os.path.join(line_dir, f"{base}_lines.pkl"), "rb") as f:
        data = pickle.load(f)
    inters, hor, ver = data['intersections'], data['horizontal'], data['vertical']
    # 3) rects
    rects = find_min_rectangles(inters, hor, ver, centers)
    # 4) col clustering
    clusters = assign_col_idx_by_cx(rects)
    # 5) floor rows
    floors = sorted([r for r in rects if any(k in r['text'] for k in ('F','B','PIT'))],
                    key=lambda r: r['rect'][1])
    # 6) build
    byh = sorted(rects, key=lambda r: r['rect'][3]-r['rect'][1])
    cols = [chr(65+i) for i in range(len(clusters))]
    rows = []
    for fr in floors:
        fy1,fy2 = fr['rect'][1], fr['rect'][3]
        row = {c:"" for c in cols}
        row[cols[fr['col']]] = fr['text']
        for r in byh:
            y1,y2 = r['rect'][1], r['rect'][3]
            if y1 <= fy1 and fy2 <= y2:
                cname = cols[r['col']]
                if not row[cname]:
                    row[cname] = r['text']
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)

def batch_extract_consolidated(img_dir, save_line_dir, ocr_dir, consolidated_path,
                               **line_kwargs):
    """
    1) img_dir → detect_and_draw_lines_batch → PKL 생성
    2) ocr_dir & save_line_dir → extract_table_dynamic → 개별 DF 추출
    3) 모든 DF 합쳐서 하나의 엑셀 파일에 저장
    """
    # 1) lines.pkl 생성
    detect_and_draw_lines_batch(
        img_dir=img_dir, save_dir=save_line_dir,
        max_examples=None, return_imgs=False, **line_kwargs
    )

    # 2) OCR→DF 반복 추출
    all_dfs = []
    for fn in os.listdir(ocr_dir):
        if not fn.lower().endswith(".json"):
            continue
        base = os.path.splitext(fn)[0]
        df = extract_table_dynamic(base, ocr_dir, save_line_dir)
        if not df.empty:
            df.insert(0, "파일명", base)  # 필요시 구분용 컬럼 추가
            all_dfs.append(df)

    if not all_dfs:
        print("추출된 표가 없습니다.")
        return

    # 3) 합치기 & 저장
    combined = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
    combined.to_excel(consolidated_path, index=False)
    print(f"[통합 엑셀 저장 완료] {consolidated_path}")


#######################################################################################################
########################################################################################################
#벽체 구조도면 끝 아래부터 구조계산서 벽체####################################################################
##########################################################################################################
##########################################################################################################





WALL_IMAGE_DIR = os.path.join(BASE_DIR, "raw_data_WALL_SCD")

# 상위 클래스 폴더들
MIDAS_DIR = os.path.join(BASE_DIR,"wall", "MIDAS")
BEST_DIR = os.path.join(BASE_DIR,"wall", "BeST")
TABLE_DIR = os.path.join(BASE_DIR,"wall", "Table")
MIDAS_INPUT_DIR = os.path.join(BASE_DIR,"wall", "MIDAS", "raw_data_classification_MIDAS")
MIDAS_OUTPUT_JSON_DIR = os.path.join(BASE_DIR,"wall", "MIDAS", "layout_results_json")
MIDAS_OUTPUT_VIS_DIR = os.path.join(BASE_DIR,"wall", "MIDAS", "layout_visualized")
MIDAS_OUTPUT_CROP_DIR = os.path.join(BASE_DIR,"wall", "MIDAS", "layout_crops")
MIDAS_OUTPUT_figure_DIR = os.path.join(BASE_DIR,"wall", "MIDAS", "layout_crops",'figure')
MIDAS_OUTPUT_plain_text_DIR = os.path.join(BASE_DIR,"wall", "MIDAS" ,"layout_crops","plain text")
MIDAS_OUTPUT_table_DIR = os.path.join(BASE_DIR,"wall", "MIDAS","layout_crops", "table")
Plain_text_ocr = os.path.join(MIDAS_OUTPUT_CROP_DIR,"plain_text_OCR")
Figure_ocr = os.path.join(MIDAS_OUTPUT_CROP_DIR,"figure_OCR")

Plain_text_elements_extraction = os.path.join(MIDAS_OUTPUT_CROP_DIR,"plain text elements")
BEST_INPUT_DIR = os.path.join(BASE_DIR,"wall", "BeST", "raw_data_classification_BeST")
BEST_OUTPUT_VIS_DIR = os.path.join(BASE_DIR,"wall", "BeST", "layout_visualized")
TABLE_INPUT_DIR = os.path.join(BASE_DIR,"wall", "TABLE", "raw_data_classification_Table")
TABLE_LATOUT = os.path.join(BASE_DIR,"wall", "TABLE", "layout_visualized")
BEST_OCR = os.path.join(BASE_DIR,"wall", "BeST", "BeST_OCR")
Table_crop = os.path.join(BASE_DIR,"wall", "TABLE", "crop")
Table_OCR = os.path.join(TABLE_DIR, "table_OCR")
BeST_plain_text = os.path.join(BEST_DIR,"wall","plain_text")
BeST_figure = os.path.join(BEST_DIR,"wall","figure")
BeST_table = os.path.join(BEST_DIR,"wall","table")




# 디렉토리 생성
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MIDAS_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)
os.makedirs(MIDAS_INPUT_DIR, exist_ok=True)
os.makedirs(MIDAS_OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(MIDAS_OUTPUT_VIS_DIR, exist_ok=True)
os.makedirs(MIDAS_OUTPUT_figure_DIR, exist_ok=True)
os.makedirs(MIDAS_OUTPUT_plain_text_DIR, exist_ok=True)
os.makedirs(MIDAS_OUTPUT_table_DIR, exist_ok=True)
os.makedirs(MIDAS_OUTPUT_CROP_DIR, exist_ok=True)
os.makedirs(Plain_text_ocr, exist_ok=True)
os.makedirs(Figure_ocr, exist_ok=True)
os.makedirs(Plain_text_elements_extraction, exist_ok=True)
os.makedirs(BEST_INPUT_DIR, exist_ok=True)
os.makedirs(BEST_OUTPUT_VIS_DIR, exist_ok=True)
os.makedirs(BEST_OCR, exist_ok=True)
os.makedirs(TABLE_INPUT_DIR, exist_ok=True)
os.makedirs(Table_crop, exist_ok=True)
os.makedirs(TABLE_LATOUT, exist_ok=True)
os.makedirs(Table_OCR, exist_ok=True)
os.makedirs(Table_crop, exist_ok=True)
os.makedirs(TABLE_LATOUT, exist_ok=True)
os.makedirs(Table_OCR, exist_ok=True)
os.makedirs(BeST_plain_text, exist_ok=True)
os.makedirs(BeST_table, exist_ok=True)
os.makedirs(BeST_figure, exist_ok=True)


# -------------------- 설정 --------------------


CONF_THRESH = 0.3
IOU_THRESH = 0.2
IMAGE_SIZE = 1024

# -------------------- 모델 로딩 --------------------
model_file = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
yolo_model = YOLOv10(model_file)

# -------------------- 기능 함수 --------------------
def convert_pdf_to_images(uploaded_pdf, output_dir=WALL_IMAGE_DIR):
    
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_bytes(uploaded_pdf.read(), dpi=300)
    image_paths = []

    for idx, image in enumerate(images):
        img_path = os.path.join(output_dir, f"page_{idx + 1}.png")
        image.save(img_path, "PNG")
        image_paths.append(img_path)

    return image_paths

def extract_yolo_features(results):
    boxes = results.boxes
    names = results.names
    labels = [names[int(b.cls.item())] for b in boxes]

    counts = Counter(labels)
    total = len(labels)
    widths = [b.xyxy[0][2] - b.xyxy[0][0] for b in boxes]
    heights = [b.xyxy[0][3] - b.xyxy[0][1] for b in boxes]

    feats = {}
    for t in ['text', 'table', 'figure', 'title', 'list']:
        feats[f"{t}_count"] = counts.get(t, 0)
        feats[f"{t}_ratio"] = counts.get(t, 0) / total if total else 0

    feats['box_count'] = total
    feats['avg_height'] = float(np.mean(heights)) if heights else 0
    feats['avg_width'] = float(np.mean(widths)) if widths else 0
    feats['std_height'] = float(np.std(heights)) if heights else 0
    feats['std_width'] = float(np.std(widths)) if widths else 0
    return feats

def predict_document_type(image_paths, conf_thresh):
    conf_folder = f"thresh_{str(conf_thresh).replace('.', '')}"
    model_dir = os.path.join("D:/wall_slabe_data/model", conf_folder)

    # 모델 자동 선택
    model_files = [f for f in os.listdir(model_dir) if f.startswith("best_model")]
    if not model_files:
        raise FileNotFoundError(f"❌ 모델 파일 없음: {model_dir}")

    model_file = os.path.join(model_dir, model_files[0])
    rf_model = joblib.load(model_file)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    predictions = []

    for img_path in image_paths:
        results = yolo_model.predict(img_path, imgsz=IMAGE_SIZE, conf=conf_thresh, iou=IOU_THRESH)[0]
        features = extract_yolo_features(results)

        df_feat = np.array([list(features.values())])
        label_idx = rf_model.predict(df_feat)[0]
        label = label_encoder.inverse_transform([label_idx])[0]

        predictions.append({
            "image_path": img_path,
            "label": label
        })

    return predictions



def save_images_by_prediction(results):
    class_counters = {}

    for item in results:
        src_path = item["image_path"]
        label = item["label"]

        # 새로운 구조: D:/Slabe_Wall/{label}/raw_data_classification_{label}/
        upper_dir = os.path.join(BASE_DIR, label)
        save_dir = os.path.join(upper_dir, f"raw_data_classification_{label}")
        os.makedirs(save_dir, exist_ok=True)

        # 파일명 구성
        original_name = os.path.splitext(os.path.basename(src_path))[0]
        ext = os.path.splitext(src_path)[1]

        class_counters[label] = class_counters.get(label, 0) + 1
        count = class_counters[label]

        new_filename = f"{original_name}_{label}_{count}{ext}"
        dst_path = os.path.join(save_dir, new_filename)

        shutil.copy(src_path, dst_path)

######################################################################################################
#MIDAS 애들 정보 추출

def analyze_and_crop_midas():
    image_files = [
        f for f in os.listdir(MIDAS_INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_file in image_files:
        img_path = os.path.join(MIDAS_INPUT_DIR, img_file)

        # YOLO 분석
        results = yolo_model.predict(
            img_path, imgsz=IMAGE_SIZE, conf=CONF_THRESH, iou=IOU_THRESH
        )[0]

        # 결과 저장할 JSON 리스트
        json_list = []

        # 이미지 로드
        image = Image.open(img_path)

        for idx, box in enumerate(results.boxes):
            label = results.names[int(box.cls.item())]
            xyxy = [round(x, 2) for x in box.xyxy[0].tolist()]
            json_list.append({"label": label, "box": xyxy})

            # 🎯 크롭 대상 클래스만 저장
            if label in ["plain text", "table", "figure"]:
                x1, y1, x2, y2 = map(int, xyxy)
                cropped = image.crop((x1, y1, x2, y2))

                class_folder = os.path.join(MIDAS_OUTPUT_CROP_DIR, label.replace(" ", "_"))
                crop_filename = f"{os.path.splitext(img_file)[0]}_{label.replace(' ', '_')}_{idx+1}.png"
                crop_path = os.path.join(class_folder, crop_filename)
                cropped.save(crop_path)

        # JSON 저장
        json_name = os.path.splitext(img_file)[0] + ".json"
        json_path = os.path.join(MIDAS_OUTPUT_JSON_DIR, json_name)
        with open(json_path, "w") as jf:
            json.dump(json_list, jf, indent=2)

        # 시각화 이미지 저장
        vis_np = results.plot()
        vis_img = Image.fromarray(vis_np)
        vis_img.save(os.path.join(MIDAS_OUTPUT_VIS_DIR, img_file))

    print(f"✅ MIDAS 분석 및 크롭 완료: 총 {len(image_files)}개 이미지 처리됨")



def run_midas_analysis():
    print("📁 모델 파일 경로:", model_file)
    print("📌 모델 라벨 목록:", yolo_model.names)

    image_files = [
        f for f in os.listdir(MIDAS_INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    total = len(image_files)

    progress_bar = st.progress(0)
    status_text = st.empty()

    label_names = yolo_model.names
    MIN_HEIGHT_THRESHOLD = 200

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(MIDAS_INPUT_DIR, img_file)
        image = Image.open(img_path).convert("RGB")
        base_name = os.path.splitext(img_file)[0]

        results = yolo_model.predict(
            img_path, imgsz=IMAGE_SIZE, conf=CONF_THRESH, iou=IOU_THRESH
        )[0]

        print(f"[{base_name}] 박스 개수: {len(results.boxes)}")
        if hasattr(results, "boxes"):
            label_set = set([label_names[int(box.cls.item())] for box in results.boxes])
            print(f"[{base_name}] 📌 Detected labels: {label_set}")
        if not hasattr(results, "boxes") or len(results.boxes) == 0:
            print(f"[{base_name}] ❗ No boxes detected")
            continue

        boxes = results.boxes
        cropped_items = []

        for i, box in enumerate(boxes):
            label = label_names[int(box.cls.item())]
            xyxy = [round(x, 2) for x in box.xyxy[0].tolist()]
            height = xyxy[3] - xyxy[1]

            if label == "plain text":
                print(f"[{base_name}] plain text height: {height}")
                if height < MIN_HEIGHT_THRESHOLD:
                    print(f"⛔ Skipping plain text due to small height: {height}")
                    continue

            cropped_items.append({"label": label, "box": xyxy})

        # figure 중 상단 1개만 남기기
        figures = [item for item in cropped_items if item["label"] == "figure"]
        if figures:
            top_figure = sorted(figures, key=lambda x: x["box"][1])[0]
            cropped_items = [item for item in cropped_items if item["label"] != "figure"]
            cropped_items.append(top_figure)

        plain_count = table_count = figure_count = 0

        for item in cropped_items:
            label = item["label"]
            x1, y1, x2, y2 = map(int, item["box"])
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image.width), min(y2, image.height)
            cropped_image = image.crop((x1, y1, x2, y2))
            height = y2 - y1

            if label == "plain text":
                print(f"[{base_name}] 📐 plain_text height 계산: {height}")
                if height < MIN_HEIGHT_THRESHOLD:
                    print(f"[{base_name}] ⛔ Skipping plain_text: height {height} < THRESH {MIN_HEIGHT_THRESHOLD}")
                    continue
                else:
                    print(f"[{base_name}] ✅ plain_text height 조건 통과")
                save_path = os.path.join(MIDAS_OUTPUT_plain_text_DIR, f"{base_name}_plain_{plain_count}.png")
                cropped_image.save(save_path)
                plain_count += 1

            elif label == "table":
                save_path = os.path.join(MIDAS_OUTPUT_table_DIR, f"{base_name}_table_{table_count}.png")
                cropped_image.save(save_path)
                table_count += 1

            elif label == "figure":
                save_path = os.path.join(MIDAS_OUTPUT_figure_DIR, f"{base_name}_figure_{figure_count}.png")
                cropped_image.save(save_path)
                figure_count += 1

        progress = (idx + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"🔄 {idx + 1} / {total} 처리 중: {img_file}")

    st.success("✅ 분석 및 크롭 완료!")


def apply_surya_ocr_to_plain_text():
    plain_text_folder = MIDAS_OUTPUT_plain_text_DIR
    surya_output_folder = os.path.join(MIDAS_OUTPUT_CROP_DIR, "plain_text_OCR")

    os.makedirs(surya_output_folder, exist_ok=True)

    # ✅ 기존 OCR 결과가 있으면 생략
    existing_jsons = [f for f in os.listdir(surya_output_folder) if f.endswith(".json")]
    if existing_jsons:
        st.info(f"ℹ️ 이미 {len(existing_jsons)}개의 OCR 결과가 존재합니다. Surya OCR 생략.")
        return

    # ✅ 입력 이미지 확인
    image_files = [f for f in os.listdir(plain_text_folder) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        st.error("❌ plain text 이미지가 존재하지 않습니다.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    st.write("🔍OCR 실행 중...")

    for idx, image_file in enumerate(image_files):
        input_path = os.path.join(plain_text_folder, image_file)
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


def apply_surya_ocr_to_figure():
    figure_folder = MIDAS_OUTPUT_figure_DIR
    surya_output_folder = os.path.join(MIDAS_OUTPUT_CROP_DIR, "figure_OCR")

    os.makedirs(surya_output_folder, exist_ok=True)

    # ✅ 기존 OCR 결과가 있으면 생략
    existing_jsons = [f for f in os.listdir(surya_output_folder) if f.endswith(".json")]
    if existing_jsons:
        st.info(f"ℹ️ 이미 {len(existing_jsons)}개의 OCR 결과가 존재합니다. Surya OCR 생략.")
        return

    # ✅ 입력 이미지 확인
    image_files = [f for f in os.listdir(figure_folder) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        st.error("❌ plain text 이미지가 존재하지 않습니다.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    st.write("🔍OCR 실행 중...")

    for idx, image_file in enumerate(image_files):
        input_path = os.path.join(figure_folder, image_file)
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
#####################################################

def render_keyword_based_ocr_extraction():
    image_list = sorted(glob.glob(os.path.join(MIDAS_OUTPUT_plain_text_DIR, "*.png")))
    json_list = sorted(glob.glob(os.path.join(MIDAS_OUTPUT_CROP_DIR, "plain_text_OCR", "*.json")))
    ocr_folder_path = os.path.join(MIDAS_OUTPUT_CROP_DIR, "plain_text_OCR")
    txt_save_dir = os.path.join(MIDAS_OUTPUT_CROP_DIR, "plain text elements")
    os.makedirs(txt_save_dir, exist_ok=True)

    if not image_list or not json_list:
        st.error("❌ 이미지 또는 OCR 결과가 존재하지 않습니다.")
        return

    def load_ocr_results(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        first_key = next(iter(data))
        text_lines = data[first_key][0]['text_lines']
        return [{"text": item["text"], "box": item["bbox"]} for item in text_lines]

    def get_center(box):
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        return x, y

    def draw_bounding_boxes_with_numbers(image_path, ocr_data, max_width=600):
        # 1) 이미지 로드 및 OCR 박스+번호 그리기
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        for idx, item in enumerate(ocr_data):
            box = item["box"]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0] + 5, box[1] + 5), str(idx), fill="blue", font=font)

        # 2) 최대 너비(max_width)보다 크면 비율 유지하며 축소
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            # Pillow 10 이상에서는 Resampling.LANCZOS 권장
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            image = image.resize((max_width, new_height), resample)

        return image



    def find_right_side_value(ocr_data, ref_box, y_tolerance=15):
        ref_x, ref_y = get_center(ref_box)
        candidates = []
        for item in ocr_data:
            x, y = get_center(item["box"])
            if abs(y - ref_y) < y_tolerance and x > ref_x:
                dist = abs(x - ref_x)
                candidates.append((dist, item["text"]))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1].lstrip(":：").strip()
    

    def normalize(s: str) -> str:
        return re.sub(r"\s+", "", s or "").lower()

    def find_reference_box(ocr_data, keyword):
        """키워드가 들어있는 OCR 아이템의 box를 반환"""
        kw_norm = normalize(keyword)
        best = None
        best_len = 10**9

        for item in ocr_data:
            text = item.get("text", "")
            box = item.get("box")
            if not text or box is None:
                continue

            tnorm = normalize(text)
            if kw_norm in tnorm:
                # 더 짧은 텍스트일수록 키워드 전용 라벨일 가능성이 커서 우선
                if len(tnorm) < best_len:
                    best = box
                    best_len = len(tnorm)

        return best

    def extract_horizontal_rebar_patterns(text):
        pattern = r"\b(?:D|HD|SD|UHD|SUHD)[\s\-]*\d{1,2}[\s\-]*@[\s\-]*\d{2,4}\b"
        return re.findall(pattern, text)

    def clean_text_after_keyword(full_text, keyword):
        text = re.sub(r"\([^)]*\)", "", full_text)  # 괄호 안 제거
        pattern = re.escape(keyword) + r"\s*[:：.]?\s*"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()

    def extract_grouped_by_page(keywords):
        grouped_files = defaultdict(list)
        for path in glob.glob(os.path.join(ocr_folder_path, "*.json")):
            name = os.path.basename(path)
            page_group = "_".join(name.split("_")[:2])
            grouped_files[page_group].append(path)

        for group, file_list in grouped_files.items():
            combined_info = {}
            horizontal_rebars = set()
            vertical_rebar_at = None

            for json_path in file_list:
                ocr_data = load_ocr_results(json_path)

                for keyword in keywords:
                    if keyword in combined_info:
                        continue

                    ref_box = find_reference_box(ocr_data, keyword)
                    value = None

                    if ref_box:
                        value = find_right_side_value(ocr_data, ref_box)

                    # 🔥 오른쪽 값이 없을 때 fallback으로 같은 박스 내부 텍스트 사용
                    if not value:
                        for item in ocr_data:
                            if keyword.lower() in item['text'].lower():
                                cleaned = clean_text_after_keyword(item['text'], keyword)
                                if re.search(r"[a-zA-Z0-9가-힣]", cleaned):
                                    value = cleaned
                                    break

                    combined_info[keyword] = value.strip() if value else "(값 없음)"

                    if "wall dim" in keyword.lower() and "*" not in combined_info[keyword]:
                        combined_info[keyword] = "(값 없음)"

                    if "vertical rebar" in keyword.lower():
                        vertical_rebar_at = re.search(r"@[\s\-]*\d{2,4}", value or "")

                for item in ocr_data:
                    text = item["text"]
                    matches = extract_horizontal_rebar_patterns(text)
                    for match in matches:
                        if vertical_rebar_at and vertical_rebar_at.group(0).replace(" ", "") in match.replace(" ", ""):
                            continue
                        horizontal_rebars.add(match.strip())

            lines = [f"{k}: {v}" for k, v in combined_info.items()]
            if horizontal_rebars:
                lines.append("Horizontal Rebar: " + ", ".join(sorted(horizontal_rebars)))

            save_path = os.path.join(txt_save_dir, f"{group}.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    # ───── Streamlit UI ─────
# ───── Streamlit UI ─────
    ocr_json_path = json_list[0]
    base = os.path.splitext(os.path.basename(ocr_json_path))[0]
    image_path = os.path.join(MIDAS_OUTPUT_plain_text_DIR, f"{base}.png")

    ocr_data = load_ocr_results(ocr_json_path)

    st.image(
        draw_bounding_boxes_with_numbers(image_path, ocr_data),
        caption=f"🖼️ {base}.png (OCR + 번호)",
        use_column_width=False,
        width=600
    )

    st.markdown("### ✏️ 추출할 키워드 5개를 입력하세요")

    default_values = [
        "Wall ID",
        "Story",
        "Material Data",
        "Wall DIM",
        "Vertical Rebar"
    ]
    keywords = []
    for i in range(5):
        value = st.text_input(f"🔹 키워드 {i+1}", value=default_values[i], key=f"kw_{i}")
        if value.strip():
            keywords.append(value.strip())

    if st.button("🚀 부재 단위로 OCR 자동 추출 및 txt 저장"):
        extract_grouped_by_page(keywords)
        st.success(f"✅ 모든 부재(page_xx)별 txt 저장 완료! → `{txt_save_dir}`")

##################################################################################################################
######################################################################################################################
#BeST처리

def apply_best_layout_analysis():
    """
    BeST_INPUT_DIR 내 이미지에 대해 DocLayout-YOLO 모델로 레이아웃 분석을 수행하고,
    검출된 'plain text', 'figure', 'table' 영역을 별도 폴더에 크롭하여 저장한다.
    시각화 결과는 BEST_OUTPUT_VIS_DIR에 저장됨
    """
    # 입력/출력 디렉토리
    input_dir = BEST_INPUT_DIR
    out_text  = BeST_plain_text
    out_fig   = BeST_figure
    out_tab   = BeST_table
    vis_dir   = BEST_OUTPUT_VIS_DIR

    # 출력 폴더 생성
    os.makedirs(out_text, exist_ok=True)
    os.makedirs(out_fig, exist_ok=True)
    os.makedirs(out_tab, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 이미지 파일 리스트
    input_dir = BEST_INPUT_DIR
    out_text  = BeST_plain_text
    out_fig   = BeST_figure
    out_tab   = BeST_table

    # 출력 폴더 생성
    os.makedirs(out_text, exist_ok=True)
    os.makedirs(out_fig, exist_ok=True)
    os.makedirs(out_tab, exist_ok=True)

    # 이미지 파일 리스트
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(image_files)
    progress = st.progress(0)
    status = st.empty()

    # 각 이미지에 대해 분석
    for idx, fname in enumerate(sorted(image_files)):
        path = os.path.join(input_dir, fname)
        im = Image.open(path).convert('RGB')

                # YOLO 모델 예측
        results = yolo_model.predict(path, imgsz=IMAGE_SIZE, conf=CONF_THRESH, iou=IOU_THRESH)[0]

        # 시각화 결과 저장
        vis_np = results.plot()
        vis_img = Image.fromarray(vis_np)
        vis_path = os.path.join(vis_dir, f"{os.path.splitext(fname)[0]}_vis.png")
        vis_img.save(vis_path)

        # 박스별로 크롭 저장
        counters = {'plain text': 0, 'figure': 0, 'table': 0}
        for box in results.boxes:
            label = results.names[int(box.cls.item())]
            if label not in counters:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = im.crop((x1, y1, x2, y2))

            # 저장 경로 결정
            if label == 'plain text':
                folder = out_text
            elif label == 'figure':
                folder = out_fig
            else:  # 'table'
                folder = out_tab

            os.makedirs(folder, exist_ok=True)
            count = counters[label]
            save_path = os.path.join(folder, f"{os.path.splitext(fname)[0]}_{label}_{count}.png")
            crop.save(save_path)
            counters[label] += 1

        # 진행 표시
        progress.progress((idx+1)/total)
        status.text(f"🔄 {idx+1}/{total} 처리: {fname}")

    st.success(f"✅ BeST 레이아웃 분석 및 크롭 완료: 총 {total}장 처리됨")


def apply_surya_ocr_to_BeST():
    figure_folder = BEST_INPUT_DIR
    surya_output_folder = os.path.join(BEST_DIR, "BeST_OCR")

    os.makedirs(surya_output_folder, exist_ok=True)

    # ✅ 기존 OCR 결과가 있으면 생략
    existing_jsons = [f for f in os.listdir(surya_output_folder) if f.endswith(".json")]
    if existing_jsons:
        st.info(f"ℹ️ 이미 {len(existing_jsons)}개의 OCR 결과가 존재합니다. Surya OCR 생략.")
        return

    # ✅ 입력 이미지 확인
    image_files = [f for f in os.listdir(figure_folder) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        st.error("❌ plain text 이미지가 존재하지 않습니다.")
        return
    st.write("🔍OCR 실행 중...")
    progress_bar = st.progress(0)
    status_text = st.empty()


    for idx, image_file in enumerate(image_files):
        input_path = os.path.join(figure_folder, image_file)
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

#---------------------------------------------------------------------------





















##########################################################################################################
###########################################################################################################
#여기서부터는 Midas_Table 형식





# def compute_iou(b1, b2):
#     x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
#     x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
#     inter_w = max(0, x2 - x1); inter_h = max(0, y2 - y1)
#     inter = inter_w * inter_h
#     area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
#     area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
#     union = area1 + area2 - inter
#     return inter/union if union>0 else 0

# # 겹치는 박스 제거 (IoU > 0 일 때 작은 것 제거)
# def filter_overlaps(boxes):
#     # boxes: list of [x1,y1,x2,y2]
#     # 넓이 순 내림차순 정렬
#     boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
#     keep = []
#     for b in boxes_sorted:
#         if all(compute_iou(b, k)==0 for k in keep):
#             keep.append(b)
#     return keep

# def analyze_table_docs(conf_thresh=0.3, top_margin=200):


#     image_files = [f for f in os.listdir(TABLE_INPUT_DIR)
#                    if f.lower().endswith((".png",".jpg",".jpeg"))]
#     total = len(image_files)
#     progress_bar = st.progress(0); status_text = st.empty()

#     for idx, img_file in enumerate(image_files):
#         base = os.path.splitext(img_file)[0]
#         img_path = os.path.join(TABLE_INPUT_DIR, img_file)
#         image = Image.open(img_path).convert("RGB")

#         # 1) 레이아웃 분석
#         results = yolo_model.predict(
#             img_path, imgsz=IMAGE_SIZE, conf=conf_thresh, iou=IOU_THRESH
#         )[0]

#         # 2) table 박스만 모아서 좌표 리스트 생성
#         raw_boxes = []
#         for box in results.boxes:
#             if results.names[int(box.cls.item())] == "table":
#                 raw_boxes.append(list(map(int, box.xyxy[0].tolist())))

#         # 3) 겹치는 건 넓이 제일 큰 것만 남김
#         filtered = filter_overlaps(raw_boxes)

#         # 4) 필터된 박스 크롭+마진 저장
#         for i, (x1,y1,x2,y2) in enumerate(filtered):
#             y1m = max(0, y1 - top_margin)
#             crop = image.crop((x1, y1m, x2, y2))
#             crop.save(os.path.join(Table_crop, f"{base}_table_{i}.png"))

#         # 5) 전체 박스 시각화
#         vis_np = results.plot()
#         Image.fromarray(vis_np).save(os.path.join(TABLE_LATOUT, img_file))

#         # 6) 진행 표시
#         progress = (idx+1)/total
#         progress_bar.progress(progress)
#         status_text.text(f"{idx+1}/{total} 처리 중: {img_file} (크롭된 테이블 {len(filtered)}개)")

#     progress_bar.empty()
#     status_text.success(f"TABLE {total}개 완료 → 시각화:{TABLE_LATOUT}, 크롭:{Table_crop}")


# #########



# # ✅ 3. Surya OCR 적용 (진행상황 + 캐시 확인 + 결과 이동)
# def apply_surya_ocr_table():
#     SURYA_RESULTS_FOLDER = r"D:\streamlit_app\app\results\surya"
#     table_folder = Table_crop
#     surya_output_folder = os.path.join(Table_OCR)
#     os.makedirs(surya_output_folder, exist_ok=True)

#     # ✅ 결과 JSON 파일이 이미 존재하면 재실행 생략
#     existing_jsons = [f for f in os.listdir(surya_output_folder) if f.endswith(".json")]
#     if existing_jsons:
#         st.info(f"ℹ️ 이미 {len(existing_jsons)}개의 OCR 결과가 존재합니다. Surya OCR 생략.")
#         return

#     image_files = [f for f in os.listdir(table_folder) if f.endswith(('.jpg', '.png'))]
#     if not image_files:
#         st.error("❌ No files to apply OCR")
#         return

#     progress_bar = st.progress(0)
#     status_text = st.empty()
#     st.write("🔍Running OCR...")

#     for idx, image_file in enumerate(image_files):
#         input_path = os.path.join(table_folder, image_file)
#         command = ["surya_ocr", input_path]
#         result = subprocess.run(command, capture_output=True, text=True)

#         stderr = result.stderr.strip()
#         if stderr and all(x not in stderr for x in ["Detecting bboxes", "Recognizing Text"]):
#             st.warning(f"⚠️ 오류: {image_file} → {stderr}")

#         progress = int((idx + 1) / len(image_files) * 100)
#         progress_bar.progress(progress)
#         status_text.write(f"🖼️ running ocr: {image_file} ({progress}%)")

#     progress_bar.empty()
#     status_text.write("✅ OCR complete! Result moving...")

#     moved, skipped = 0, 0
#     for folder_name in os.listdir(SURYA_RESULTS_FOLDER):
#         folder_path = os.path.join(SURYA_RESULTS_FOLDER, folder_name)
#         if os.path.isdir(folder_path):
#             json_file = os.path.join(folder_path, "results.json")
#             if os.path.exists(json_file):
#                 dst_file = os.path.join(surya_output_folder, f"{folder_name}.json")
#                 try:
#                     shutil.move(json_file, dst_file)
#                     moved += 1
#                 except Exception as e:
#                     st.error(f"❌ Move Error: {folder_name} → {e}")
#             else:
#                 skipped += 1

#     st.success(f"📁 OCR 결과 {moved}개 이동 완료 ({skipped}개는 누락됨)")


# #--------------------------------------------------------------
# def split_by_keywords(json_path, keywords, output_dir):
#     """
#     json_path: 처리할 OCR JSON 파일 경로
#     keywords: ['MEMB=', 'Wall mark :', ...] 형태의 분할 키워드 리스트
#     output_dir: 분할된 JSON을 저장할 폴더
#     """
#     with open(json_path, encoding='utf-8') as f:
#         data = json.load(f)
#     key = next(iter(data))
#     entry = data[key][0]
#     lines = entry['text_lines']

#     # boundaries: [(index, label_text), ...]
#     boundaries = []
#     for idx, item in enumerate(lines):
#         text = item['text']
#         for kw in keywords:
#             if kw in text:
#                 label = text.split(kw, 1)[1].strip()
#                 boundaries.append((idx, label))
#                 break

#     # 키워드가 하나도 안 걸리면 원본 그대로 저장
#     if not boundaries:
#         base = os.path.splitext(os.path.basename(json_path))[0]
#         dst = os.path.join(output_dir, f"{base}_full.json")
#         os.makedirs(output_dir, exist_ok=True)
#         with open(dst, 'w', encoding='utf-8') as wf:
#             json.dump(data, wf, ensure_ascii=False, indent=2)
#         return

#     # 각 구간별로 slice & save
#     for i, (start, label) in enumerate(boundaries):
#         end = boundaries[i+1][0] if i+1 < len(boundaries) else len(lines)
#         segment = lines[start:end]

#         new_entry = {k: v for k, v in entry.items() if k != 'text_lines'}
#         new_entry['text_lines'] = segment
#         out_data = { key: [ new_entry ] }

#         safe_label = re.sub(r'[^\w\-]', '_', label)
#         base = os.path.splitext(os.path.basename(json_path))[0]
#         out_name = f"{base}_{safe_label}.json"

#         os.makedirs(output_dir, exist_ok=True)
#         with open(os.path.join(output_dir, out_name), 'w', encoding='utf-8') as wf:
#             json.dump(out_data, wf, ensure_ascii=False, indent=2)







# def process_table_ocr_splits(keywords, input_dir, output_dir, margin=0):
#     input_dir     = os.path.join(TABLE_DIR, "table_OCR")
#     output_dir    = os.path.join(TABLE_DIR, "table_OCR_split")       
#     os.makedirs(output_dir, exist_ok=True)
#     file_count = 0

#     # Gather and sort JSON files by page number
#     file_paths = sorted(
#         glob.glob(os.path.join(input_dir, "*.json")),
#         key=lambda fp: int(re.search(r"page_(\d+)", os.path.basename(fp), re.IGNORECASE).group(1))
#     )

#     # Pattern for extra info (floor, rebar)
#     extra_pattern = re.compile(r"\b(?:B\d+|PIT|\d+F)\b|@\d+", re.IGNORECASE)

#     carryover_id = None
#     carryover_lines = []
#     prev_base = None

#     for json_path in tqdm(file_paths, desc="Splitting OCR files"):
#         base = os.path.splitext(os.path.basename(json_path))[0]
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         root = next(iter(data.values()))[0]
#         lines = root.get('text_lines', [])

#         # Detect boundary lines and identifiers
#         boundaries = []  # [(index, identifier), ...]
#         for idx_line, line in enumerate(lines):
#             text = line.get('text', '')
#             for kw in keywords:
#                 if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
#                     parts = re.split(r"[:=]", text, 1)
#                     if len(parts) > 1:
#                         ident = parts[1].strip().split()[0]
#                         boundaries.append((idx_line, ident))
#                     break

#         # Collect prefix_extra indices and lines before first boundary
#         prefix_indices = []
#         prefix_extra = []
#         if boundaries:
#             first_idx = boundaries[0][0]
#             for i in range(first_idx):
#                 if extra_pattern.search(lines[i].get('text', '')):
#                     prefix_indices.append(i)
#                     prefix_extra.append(lines[i])
#         else:
#             # No boundary: if carryover exists, collect extra lines for next flush
#             if carryover_id:
#                 carryover_lines.extend([l for l in lines if extra_pattern.search(l.get('text',''))])
#             prev_base = base
#             continue

#         # Flush carryover if identifier changed
#         first_ident = boundaries[0][1]
#         if carryover_id and first_ident != carryover_id:
#             flush_lines = carryover_lines + prefix_extra
#             safe_label = re.sub(r"[^\w\-]", "_", carryover_id)
#             out_name = f"{prev_base}_{safe_label}.json"
#             with open(os.path.join(output_dir, out_name), 'w', encoding='utf-8') as wf:
#                 json.dump({prev_base:[{'text_lines':flush_lines}]}, wf, ensure_ascii=False, indent=2)
#             file_count += 1
#             carryover_id, carryover_lines = None, []

#         # Build segments for this file
#         idxs = [b[0] for b in boundaries] + [len(lines)]
#         segments = []
#         for i, (_, ident) in enumerate(boundaries):
#             start, end = idxs[i], idxs[i+1]
#             # exclude prefix_extra indices from first segment
#             seg_lines = [lines[j] for j in range(start, end) if j not in prefix_indices]
#             if seg_lines:
#                 segments.append((ident, seg_lines))

#         # Save all but last segment
#         for ident, seg_lines in segments[:-1]:
#             safe_label = re.sub(r"[^\w\-]", "_", ident)
#             out_name = f"{base}_{safe_label}.json"
#             with open(os.path.join(output_dir, out_name), 'w', encoding='utf-8') as wf:
#                 json.dump({base:[{'text_lines':seg_lines}]}, wf, ensure_ascii=False, indent=2)
#             file_count += 1

#         # Last segment becomes carryover
#         if segments:
#             carryover_id, carryover_lines = segments[-1]
#         prev_base = base

#     # Flush any remaining carryover
#     if carryover_id and carryover_lines:
#         safe_label = re.sub(r"[^\w\-]", "_", carryover_id)
#         out_name = f"{prev_base}_{safe_label}.json"
#         with open(os.path.join(output_dir, out_name), 'w', encoding='utf-8') as wf:
#             json.dump({prev_base:[{'text_lines':carryover_lines}]}, wf, ensure_ascii=False, indent=2)
#         file_count += 1

#     return file_count


# def extract_floor_info(split_dir, csv_path):
#     floor_pattern = re.compile(r"\b(?:\d+F|B\d+|PIT)\b", re.IGNORECASE)
#     rows = []
#     for fn in sorted(os.listdir(split_dir)):
#         if not fn.lower().endswith('.json'):
#             continue
#         fp = os.path.join(split_dir, fn)
#         with open(fp,'r',encoding='utf-8') as f:
#             data = json.load(f)
#         key = next(iter(data))
#         for entry in data[key]:
#             for line in entry.get('text_lines', []):
#                 text = line.get('text','')
#                 for m in floor_pattern.findall(text):
#                     rows.append({'file':fn,'floor':m,'text':text})
#     with open(csv_path,'w',newline='',encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=['file','floor','text'])
#         writer.writeheader()
#         writer.writerows(rows)
#     return len(rows)

# # ─── 3) 줄 단위 테이블 추출 함수 ────────────────────────────────────────
# def extract_table_rows(json_path, y_tol=10):
#     """Extract rows from a single table JSON by clustering text_lines by y-center"""
#     with open(json_path, encoding='utf-8') as f:
#         data = json.load(f)
#     lines = next(iter(data.values()))[0]['text_lines']
#     # compute y-centers
#     items = []
#     for ln in lines:
#         x1,y1,x2,y2 = ln['bbox']
#         items.append({'y':(y1+y2)/2, 'x':x1, 'text':ln['text']})
#     if not items:
#         return []
#     # sort by y
#     items.sort(key=lambda i: i['y'])
#     rows = []
#     current = [items[0]]
#     for it in items[1:]:
#         avg_y = sum(i['y'] for i in current)/len(current)
#         if abs(it['y']-avg_y) <= y_tol:
#             current.append(it)
#         else:
#             rows.append(current)
#             current = [it]
#     rows.append(current)
#     # combine texts sorted by x
#     combined = []
#     for row in rows:
#         row.sort(key=lambda i: i['x'])
#         combined.append(' '.join(i['text'] for i in row))
#     return combined


# def smart_split(line: str) -> list:
#     """
#     Split a line by whitespace outside of parentheses,
#     so tokens like 'Vu(kN,LCB, iWAL, Lw)' stay intact.
#     """
#     tokens = []
#     buf = ''
#     depth = 0
#     for ch in line:
#         if ch == '(':
#             depth += 1
#             buf += ch
#         elif ch == ')':
#             depth -= 1
#             buf += ch
#         elif ch.isspace() and depth == 0:
#             if buf:
#                 tokens.append(buf)
#                 buf = ''
#         else:
#             buf += ch
#     if buf:
#         tokens.append(buf)
#     return tokens

# # ===================== 주요 함수 =====================
# def smart_split(line: str) -> list:
#     """
#     괄호 밖 공백만 분리하여 ( ) 내부는 하나의 토큰으로 유지
#     """
#     tokens, buf, depth = [], '', 0
#     for ch in line:
#         if ch == '(':
#             depth += 1; buf += ch
#         elif ch == ')':
#             depth -= 1; buf += ch
#         elif ch.isspace() and depth == 0:
#             if buf:
#                 tokens.append(buf)
#                 buf = ''
#         else:
#             buf += ch
#     if buf:
#         tokens.append(buf)
#     return tokens

# # ===================== 주요 함수 =====================
# def process_csv_to_excel(csv_path: str, excel_path: str) -> int:
#     """
#     CSV에서 테이블 블록(메타데이터+헤더+데이터)을 추출하여
#     스마트 분리 후 괄호와 점 처리 규칙 적용, 엑셀로 저장

#     Args:
#         csv_path: 입력 CSV 경로 (text 컬럼 혹은 마지막 컬럼 사용)
#         excel_path: 출력 Excel 경로

#     Returns:
#         작성된 시트 수
#     """
#     # CSV 로드 및 텍스트 라인 확보
#     df_raw = pd.read_csv(csv_path)
#     lines = df_raw['text'].astype(str).tolist() if 'text' in df_raw.columns else df_raw.iloc[:, -1].astype(str).tolist()

#     # 패턴 정의
#     header_pat = re.compile(r"STO\s+HTw", re.IGNORECASE)
#     eq_pat     = re.compile(r"\*\.\s*(.*?)\s*=\s*(.*)")
#     col_pat    = re.compile(r"\*\.\s*(.*?)\s*:\s*(.*)")
#     floor_pat  = re.compile(r'^(?:\d+F|B\d+|PIT)', re.IGNORECASE)

#     # 헤더 위치 찾기
#     header_idxs = [i for i, ln in enumerate(lines) if header_pat.search(ln)]
#     if not header_idxs:
#         raise ValueError("STO HTw 헤더를 찾지 못했습니다.")

#     # ExcelWriter 사용
#     with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
#         sheet_count = 0
#         for idx_num, h_idx in enumerate(header_idxs):
#             # 1) 헤더 토큰화
#             headers = smart_split(lines[h_idx].strip())

#                         # 2) 메타데이터 수집: 헤더명과 중복되지 않는 키만, 헤더 위 최대 5줄 검사
#             meta = {}
#             for lookback in range(max(0, h_idx-5), h_idx):
#                 line = lines[lookback].strip()
#                 m = eq_pat.match(line) or col_pat.match(line)
#                 if m:
#                     key, val = m.group(1).strip(), m.group(2).strip()
#                     if key not in headers:
#                         meta[key] = val

#             # 3) 데이터 블록 범위 설정
#             next_h = header_idxs[idx_num + 1] if idx_num + 1 < len(header_idxs) else len(lines)
#             data_rows = []

#             # 4) 데이터 행 처리
#             for ln in lines[h_idx + 1:next_h]:
#                 txt = ln.strip()
#                 if not txt or eq_pat.match(txt) or col_pat.match(txt) or header_pat.search(txt):
#                     continue
#                 parts = smart_split(txt)
#                 # (1) 단독 점 제거
#                 parts = [p for p in parts if p != '.']
#                 # (2) 괄호 컬럼 병합
#                 for i, col in enumerate(headers):
#                     if '(' in col and ')' in col and i < len(parts) - 1 and parts[i + 1].startswith('('):
#                         parts[i] = f"{parts[i]} {parts[i+1]}"
#                         del parts[i+1]
#                 # (3) 점+괄호 병합
#                 k = 0
#                 while k < len(parts) - 1:
#                     if parts[k].endswith('.') and parts[k+1].startswith('('):
#                         parts[k] = f"{parts[k]} {parts[k+1]}"
#                         del parts[k+1]
#                         continue
#                     k += 1
#                 # (4) 분리된 점 병합 (400 . D10@300)
#                 i2 = 1
#                 while i2 < len(parts) - 1:
#                     if parts[i2] == '.':
#                         prev_tok, next_tok = parts[i2 - 1], parts[i2 + 1]
#                         if not next_tok.startswith(('(', ',')):
#                             parts[i2 - 1:i2 + 2] = [f"{prev_tok}.", next_tok]
#                             i2 += 2
#                             continue
#                     i2 += 1
#                 # (5) mid-word 점 분할
#                 j = 0
#                 while j < len(parts) - 1:
#                     tok = parts[j]
#                     if j < len(headers) - 1 and re.search(r"\w\.\w", tok):
#                         left, right = tok.split('.', 1)
#                         if not right.startswith(('(', ',')):
#                             parts[j:j+1] = [f"{left}." if tok.endswith('.') else left, right]
#                             j += 2
#                             continue
#                     j += 1
#                 # (6) 칼럼 수 맞추기
#                 if len(parts) > len(headers):
#                     parts = parts[:len(headers) - 1] + [' '.join(parts[len(headers) - 1:])]
#                 elif len(parts) < len(headers):
#                     parts += [''] * (len(headers) - len(parts))
#                 # (7) 층수 없는 행 스킵
#                 if not floor_pat.match(parts[0]):
#                     continue
#                 data_rows.append(parts)

#             # 데이터가 없으면 스킵
#             if not data_rows:
#                 continue

#             # DataFrame 생성 및 메타데이터 결합
#             df = pd.DataFrame(data_rows, columns=headers)
#             for k, v in meta.items():
#                 df[k] = v

#             # 5) 시트명 설정: MEMB 우선, 없으면 Wall Mark
#             memb_key = next((k for k in meta if k.lower() == 'memb'), None)
#             wm_key = next((k for k in meta if k.lower() == 'wall mark'), None)
#             if memb_key:
#                 sheet_name = meta[memb_key].split()[0].lower()
#             elif wm_key:
#                 sheet_name = meta[wm_key].split()[0]
#             else:
#                 sheet_name = f"Table{idx_num+1}"

#             # 엑셀 시트 작성
#             df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
#             sheet_count += 1

#         return sheet_count



# def export_summary_csv(excel_path: str, summary_csv_path: str) -> int:
#     """
#     생성된 Excel 파일을 읽어 각 시트별로 요약 정보를 CSV로 저장
#     """
#     xls = pd.ExcelFile(excel_path)
#     rows = []
#     for sheet in xls.sheet_names:
#         df = pd.read_excel(xls, sheet_name=sheet)
#         if 'STO' in df.columns and 'hw' in df.columns and 'V-Rebar' in df.columns and 'H-Rebar' in df.columns and 'fy' in df.columns and 'fck' in df.columns:
#             for _, r in df.iterrows():
#                 rows.append({
#                     '부재명': sheet,
#                     '층수': r['STO'],
#                     '두께': r['hw'],
#                     '수직철근': r['V-Rebar'],
#                     '수평철근': r['H-Rebar'],
#                     'fy': r['fy'],
#                     'fck': r['fck'],
#                 })
#     pd.DataFrame(rows).to_csv(summary_csv_path, index=False)
#     return len(rows)





def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter_w = max(0, x2 - x1); inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter/union if union>0 else 0

# 겹치는 박스 제거 (IoU > 0 일 때 작은 것 제거)
def filter_overlaps(boxes):
    boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    for b in boxes_sorted:
        if all(compute_iou(b, k)==0 for k in keep):
            keep.append(b)
    return keep

# 새로 추가: 사분면(좌상, 좌하, 우상, 우하) 순 정렬
def sort_by_quadrant(boxes, img_width, img_height):
    mid_x, mid_y = img_width / 2, img_height / 2

    def quadrant(b):
        x1, y1, x2, y2 = b
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # 0: 좌상, 1: 좌하, 2: 우상, 3: 우하
        if cx < mid_x and cy < mid_y:
            return 0
        if cx < mid_x and cy >= mid_y:
            return 1
        if cx >= mid_x and cy < mid_y:
            return 2
        return 3

    return sorted(boxes, key=lambda b: (quadrant(b), b[1], b[0]))

def analyze_table_docs(conf_thresh=0.3, top_margin=200):
    image_files = [f for f in os.listdir(TABLE_INPUT_DIR)
                   if f.lower().endswith((".png",".jpg",".jpeg"))]
    total = len(image_files)
    progress_bar = st.progress(0); status_text = st.empty()

    for idx, img_file in enumerate(image_files):
        base = os.path.splitext(img_file)[0]
        img_path = os.path.join(TABLE_INPUT_DIR, img_file)
        image = Image.open(img_path).convert("RGB")

        # 1) 레이아웃 분석
        results = yolo_model.predict(
            img_path, imgsz=IMAGE_SIZE, conf=conf_thresh, iou=IOU_THRESH
        )[0]

        # 2) table 박스만 모아서 좌표 리스트 생성
        raw_boxes = []
        for box in results.boxes:
            if results.names[int(box.cls.item())] == "table":
                raw_boxes.append(list(map(int, box.xyxy[0].tolist())))

        # 3) 겹치는 건 넓이 제일 큰 것만 남김
        filtered = filter_overlaps(raw_boxes)

        # → 3.1) 사분면 순으로 재정렬
        w, h = image.width, image.height
        ordered = sort_by_quadrant(filtered, w, h)

        # 4) 🔥 파일명 일관성 개선 🔥
        page_num = idx + 1  # 1부터 시작하는 페이지 번호
        for i, (x1, y1, x2, y2) in enumerate(ordered, start=1):
            y1m = max(0, y1 - top_margin)
            crop = image.crop((x1, y1m, x2, y2))
            # 일관된 파일명: page_1_table_1.png
            filename = f"page_{page_num}_table_{i}.png"
            crop.save(os.path.join(Table_crop, filename))

        # 5) 전체 박스 시각화
        vis_np = results.plot()
        Image.fromarray(vis_np).save(os.path.join(TABLE_LATOUT, img_file))

        # 6) 진행 표시
        progress = (idx+1)/total
        progress_bar.progress(progress)
        status_text.text(f"{idx+1}/{total} 처리 중: {img_file} (크롭된 테이블 {len(ordered)}개)")

    progress_bar.empty()
    status_text.success(f"TABLE {total}개 완료 → 시각화:{TABLE_LATOUT}, 크롭:{Table_crop}")

#########



# ✅ 3. Surya OCR 적용 (진행상황 + 캐시 확인 + 결과 이동)
def apply_surya_ocr_table():
    table_folder = Table_crop
    surya_output_folder = os.path.join(Table_OCR)
    os.makedirs(surya_output_folder, exist_ok=True)

    # ✅ 결과 JSON 파일이 이미 존재하면 재실행 생략
    existing_jsons = [f for f in os.listdir(surya_output_folder) if f.endswith(".json")]
    if existing_jsons:
        st.info(f"ℹ️ 이미 {len(existing_jsons)}개의 OCR 결과가 존재합니다. Surya OCR 생략.")
        return

    image_files = [f for f in os.listdir(table_folder) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        st.error("❌ No files to apply OCR")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    st.write("🔍Running OCR...")

    for idx, image_file in enumerate(image_files):
        input_path = os.path.join(table_folder, image_file)
        command = ["surya_ocr", input_path]
        result = subprocess.run(command, capture_output=True, text=True)

        stderr = result.stderr.strip()
        if stderr and all(x not in stderr for x in ["Detecting bboxes", "Recognizing Text"]):
            st.warning(f"⚠️ 오류: {image_file} → {stderr}")

        progress = int((idx + 1) / len(image_files) * 100)
        progress_bar.progress(progress)
        status_text.write(f"🖼️ running ocr: {image_file} ({progress}%)")

    progress_bar.empty()
    status_text.write("✅ OCR complete! Result moving...")

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


#--------------------------------------------------------------
def split_by_keywords(json_path, keywords, output_dir):
    """
    json_path: 처리할 OCR JSON 파일 경로
    keywords: ['MEMB=', 'Wall mark :', ...] 형태의 분할 키워드 리스트
    output_dir: 분할된 JSON을 저장할 폴더
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    key = next(iter(data))
    entry = data[key][0]
    lines = entry['text_lines']

    # boundaries: [(index, label_text), ...]
    boundaries = []
    for idx, item in enumerate(lines):
        text = item['text']
        for kw in keywords:
            if kw in text:
                label = text.split(kw, 1)[1].strip()
                boundaries.append((idx, label))
                break

    # 키워드가 하나도 안 걸리면 원본 그대로 저장
    if not boundaries:
        base = os.path.splitext(os.path.basename(json_path))[0]
        dst = os.path.join(output_dir, f"{base}_full.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(dst, 'w', encoding='utf-8') as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)
        return

    # 각 구간별로 slice & save
    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i+1][0] if i+1 < len(boundaries) else len(lines)
        segment = lines[start:end]

        new_entry = {k: v for k, v in entry.items() if k != 'text_lines'}
        new_entry['text_lines'] = segment
        out_data = { key: [ new_entry ] }

        safe_label = re.sub(r'[^\w\-]', '_', label)
        base = os.path.splitext(os.path.basename(json_path))[0]
        out_name = f"{base}_{safe_label}.json"

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, out_name), 'w', encoding='utf-8') as wf:
            json.dump(out_data, wf, ensure_ascii=False, indent=2)


#--------------------------------------------------------------------------------------

def filter_ocr_by_keywords(keywords, input_dir=None, output_dir=None):
    """
    사용자 키워드가 포함된 OCR 파일들만 필터링해서 별도 폴더에 저장
    
    Args:
        keywords: 필터링할 키워드 리스트 (예: ['MEMB', 'Wall Mark'])
        input_dir: OCR JSON 파일들이 있는 폴더 경로
        output_dir: 필터링된 파일들을 저장할 폴더 경로
    
    Returns:
        tuple: (필터링된 파일 수, 전체 파일 수)
    """
    if input_dir is None:
        input_dir = os.path.join(TABLE_DIR, "table_OCR")
    if output_dir is None:
        output_dir = os.path.join(TABLE_DIR, "table_OCR_filtered")
    
    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 기존 필터링된 파일들 삭제 (새로운 필터링 시작)
    for existing_file in os.listdir(output_dir):
        if existing_file.endswith('.json'):
            os.remove(os.path.join(output_dir, existing_file))
    
    # JSON 파일들 검사
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    filtered_count = 0
    total_count = len(json_files)
    
    print(f"총 {total_count}개 파일 검사 시작...")
    print(f"필터링 키워드: {keywords}")
    
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        
        try:
            # JSON 파일 읽기
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # OCR 텍스트 추출
            all_text = ""
            for page_key, page_data in data.items():
                if isinstance(page_data, list) and len(page_data) > 0:
                    text_lines = page_data[0].get('text_lines', [])
                    for line in text_lines:
                        all_text += line.get('text', '') + " "
            
            # 키워드 포함 여부 확인
            has_keyword = False
            found_keywords = []
            
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", all_text, re.IGNORECASE):
                    has_keyword = True
                    found_keywords.append(keyword)
            
            # 키워드가 있으면 필터링된 폴더에 복사
            if has_keyword:
                dest_path = os.path.join(output_dir, json_file)
                shutil.copy2(json_path, dest_path)
                filtered_count += 1
                print(f"✓ {json_file} -> 키워드 발견: {found_keywords}")
            else:
                print(f"✗ {json_file} -> 키워드 없음, 제외")
                
        except Exception as e:
            print(f"오류 - {json_file}: {e}")
            continue
    
    print(f"\n필터링 완료:")
    print(f"전체 파일: {total_count}개")
    print(f"필터링된 파일: {filtered_count}개")
    print(f"제외된 파일: {total_count - filtered_count}개")
    print(f"결과 저장 위치: {output_dir}")
    
    return filtered_count, total_count





















def process_table_ocr_splits(keywords, input_dir=None, output_dir=None, margin=0):
    """
    연속된 테이블 번호와 콘텐츠 기반 연속성 분석을 통한 OCR 병합 함수
    
    병합 규칙:
    1. 동일 페이지: table_x → table_x+1 연속 번호
    2. 페이지 간: page_y 마지막 table → page_y+1 첫 table
    3. 덩어리 인식 시: 가장 하단/상단 부재 키워드 기준으로 연결점 찾기
    """
    
    if input_dir is None:
        input_dir = os.path.join(TABLE_DIR, "table_OCR_filtered")
    if output_dir is None:
        output_dir = os.path.join(TABLE_DIR, "table_OCR_split")
    
    os.makedirs(output_dir, exist_ok=True)
    
    def clean_math_tags(text):
        """MathML 태그 제거"""
        text = re.sub(r'<math[^>]*>(.*?)</math>', r'\1', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        return text.strip()
    
    def parse_filename(filename):
        """파일명에서 페이지, 테이블 번호 추출"""
        pattern = r"page_(\d+)_(?:Table_\d+_)?table_(\d+)"
        match = re.search(pattern, filename)
        if match:
            return {
                'page_num': int(match.group(1)),
                'table_num': int(match.group(2)),
                'filename': filename
            }
        return None
    
    def get_member_positions(json_path, keywords):
        """OCR에서 부재 키워드 위치 추출"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        root = next(iter(data.values()))[0]
        lines = root.get('text_lines', [])
        
        member_positions = []
        data_lines = []
        
        for line in lines:
            text = clean_math_tags(line.get('text', ''))
            y_center = (line['bbox'][1] + line['bbox'][3]) / 2
            
            # 부재 키워드 찾기
            member_found = None
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
                    parts = re.split(r"[:=]", text, 1)
                    if len(parts) > 1:
                        member_id = parts[1].strip().split()[0]
                        if member_id:
                            member_found = member_id
                            break
            
            if member_found:
                member_positions.append({
                    'member_id': member_found,
                    'y_position': y_center,
                    'text': text
                })
            else:
                # 테이블 데이터 라인
                if re.search(r'[A-Za-z0-9]', text) and len(text.strip()) > 3:
                    data_lines.append({
                        'y_position': y_center,
                        'text': text
                    })
        
        return member_positions, data_lines
    
    def check_connection(source_analysis, target_analysis):
        """두 테이블 간 연결 가능성 확인"""
        source_members, source_data = source_analysis
        target_members, target_data = target_analysis
        
        if not source_members or not target_members:
            return False, None, None
        
        # source에서 가장 하단 부재 키워드
        bottom_member = max(source_members, key=lambda x: x['y_position'])
        
        # target에서 가장 상단 부재 키워드  
        top_member = min(target_members, key=lambda x: x['y_position'])
        
        # 1. source 하단 부재 아래에 데이터 있는지
        data_after_bottom = [d for d in source_data 
                           if d['y_position'] > bottom_member['y_position']]
        
        # 2. source 하단 부재 아래에 다른 부재 키워드 없는지
        other_members_below = [m for m in source_members 
                             if m['y_position'] > bottom_member['y_position']]
        
        # 3. target 상단 부재 위에 데이터 있는지
        data_before_top = [d for d in target_data 
                         if d['y_position'] < top_member['y_position']]
        
        can_connect = (len(data_after_bottom) > 0 and 
                      len(other_members_below) == 0 and 
                      len(data_before_top) > 0)
        
        return can_connect, bottom_member['member_id'], top_member['member_id']
    
    # 1. 모든 JSON 파일 수집 및 파싱
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    file_infos = []
    
    print(f"발견된 JSON 파일: {len(json_files)}개")
    
    for json_file in json_files:
        parsed = parse_filename(json_file)
        if parsed:
            json_path = os.path.join(input_dir, json_file)
            member_positions, data_lines = get_member_positions(json_path, keywords)
            
            file_infos.append({
                **parsed,
                'json_path': json_path,
                'analysis': (member_positions, data_lines)
            })
            
            print(f"파일: {json_file}")
            print(f"  page_{parsed['page_num']}, table_{parsed['table_num']}")
            print(f"  부재: {[m['member_id'] for m in member_positions]}")
        else:
            print(f"파싱 실패: {json_file}")
    
    # 2. 페이지, 테이블 번호순으로 정렬
    file_infos.sort(key=lambda x: (x['page_num'], x['table_num']))
    
    # 3. 연속성 체인 구성
    chains = []
    
    for i, current_file in enumerate(file_infos):
        # 새로운 체인 시작
        if i == 0 or not chains:
            chains.append([current_file])
            continue
        
        # 이전 파일과 연결 시도
        prev_file = chains[-1][-1]  # 현재 체인의 마지막 파일
        
        can_connect = False
        connect_info = ""
        
        # 동일 페이지 내 연속 테이블
        if (current_file['page_num'] == prev_file['page_num'] and 
            current_file['table_num'] == prev_file['table_num'] + 1):
            
            can_connect, source_member, target_member = check_connection(
                prev_file['analysis'], current_file['analysis'])
            connect_info = f"페이지내 연결: {source_member} → {target_member}"
        
        # 페이지 간 연결 (다음 페이지 첫 테이블)
        elif (current_file['page_num'] == prev_file['page_num'] + 1 and 
              current_file['table_num'] == 1):
            
            can_connect, source_member, target_member = check_connection(
                prev_file['analysis'], current_file['analysis'])
            connect_info = f"페이지간 연결: {source_member} → {target_member}"
        
        if can_connect:
            chains[-1].append(current_file)
            print(f"연결 성공: {connect_info}")
        else:
            chains.append([current_file])
            print(f"새 체인 시작: {current_file['filename']}")
    
    # 4. 각 체인별로 병합된 JSON 생성
    file_count = 0
    
    for chain_idx, chain in enumerate(chains):
        if not chain:
            continue
        
        # 체인의 모든 부재 ID 수집
        all_members = set()
        for file_info in chain:
            members, _ = file_info['analysis']
            for member in members:
                all_members.add(member['member_id'])
        
        # 각 부재별로 분리하여 저장
        for member_id in all_members:
            all_lines = []
            first_metadata = {}
            member_files = []
            
            for file_info in chain:
                with open(file_info['json_path'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                root = next(iter(data.values()))[0]
                lines = root.get('text_lines', [])
                
                # 해당 부재가 있는 파일만 포함
                members, _ = file_info['analysis']
                has_member = any(m['member_id'] == member_id for m in members)
                
                if has_member:
                    member_files.append(file_info)
                    
                    if not first_metadata:
                        first_metadata = {k: v for k, v in root.items() if k != 'text_lines'}
                    
                    # 출처 정보 추가
                    for line in lines:
                        line['source_file'] = file_info['filename']
                        line['source_page'] = file_info['page_num']
                        line['source_table'] = file_info['table_num']
                    
                    all_lines.extend(lines)
            
            if not member_files:
                continue
            
            # 파일명 생성
            first_file = member_files[0]
            last_file = member_files[-1]
            safe_member = re.sub(r'[^\w\-]', '_', member_id)
            
            if len(member_files) == 1:
                out_name = f"page_{first_file['page_num']}_table_{first_file['table_num']}_{safe_member}.json"
            else:
                out_name = (f"page_{first_file['page_num']}_table_{first_file['table_num']}_"
                           f"to_page_{last_file['page_num']}_table_{last_file['table_num']}_{safe_member}.json")
            
            # 병합된 데이터 구조
            merged_data = {
                f"merged_{safe_member}": [{
                    **first_metadata,
                    'text_lines': all_lines,
                    'member_id': member_id,
                    'chain_info': {
                        'start': f"page_{first_file['page_num']}_table_{first_file['table_num']}",
                        'end': f"page_{last_file['page_num']}_table_{last_file['table_num']}",
                        'source_files': [f['filename'] for f in member_files],
                        'total_lines': len(all_lines),
                        'merge_type': 'sequential_continuity'
                    }
                }]
            }
            
            # 파일 저장
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, 'w', encoding='utf-8') as wf:
                json.dump(merged_data, wf, ensure_ascii=False, indent=2)
            
            file_count += 1
            print(f"병합 완료: {out_name}")
            print(f"  부재: {member_id}")
            print(f"  파일: {len(member_files)}개")
            print(f"  텍스트 라인: {len(all_lines)}개")
            print("-" * 40)
    
    return file_count


def extract_floor_info(split_dir, csv_path):
    floor_pattern = re.compile(r"\b(?:\d+F|B\d+|PIT)\b", re.IGNORECASE)
    rows = []
    for fn in sorted(os.listdir(split_dir)):
        if not fn.lower().endswith('.json'):
            continue
        fp = os.path.join(split_dir, fn)
        with open(fp,'r',encoding='utf-8') as f:
            data = json.load(f)
        key = next(iter(data))
        for entry in data[key]:
            for line in entry.get('text_lines', []):
                text = line.get('text','')
                for m in floor_pattern.findall(text):
                    rows.append({'file':fn,'floor':m,'text':text})
    with open(csv_path,'w',newline='',encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['file','floor','text'])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)

# ─── 3) 줄 단위 테이블 추출 함수 ────────────────────────────────────────
def clean_math_tags(text):
    """모든 수학 표기법을 정리"""
    # LaTeX 명령어들
    latex_patterns = [
        (r'\\mathbf\{([^}]+)\}', r'\1'),      # 굵은 글씨
        (r'\\mathrm\{([^}]+)\}', r'\1'),      # 로만체
        (r'\\mathit\{([^}]+)\}', r'\1'),      # 이탤릭
        (r'\\text\{([^}]+)\}', r'\1'),        # 일반 텍스트
        (r'\\textbf\{([^}]+)\}', r'\1'),      # 굵은 텍스트
        (r'\\[a-zA-Z]+\{([^}]+)\}', r'\1'),   # 기타 LaTeX 명령어
    ]
    
    for pattern, replacement in latex_patterns:
        text = re.sub(pattern, replacement, text)
    
    # MathML 태그
    text = re.sub(r'<math>(.*?)</math>', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_table_rows(json_path, y_tol=20):
    """Extract rows from a single table JSON by clustering text_lines by y-center"""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    lines = next(iter(data.values()))[0]['text_lines']
    
    # compute y-centers
    items = []
    for ln in lines:
        x1, y1, x2, y2 = ln['bbox']
        # MathML 태그 정리 추가
        cleaned_text = clean_math_tags(ln['text'])
        height = y2 - y1  # 텍스트 높이 계산
        items.append({
            'y': (y1 + y2) / 2, 
            'x': x1, 
            'text': cleaned_text,
            'height': height
        })
    
    if not items:
        return []
    
    # sort by y
    items.sort(key=lambda i: i['y'])
    rows = []
    current = [items[0]]
    
    for it in items[1:]:
        avg_y = sum(i['y'] for i in current) / len(current)
        avg_height = sum(i['height'] for i in current) / len(current)
        
        # 적응적 허용오차: 텍스트 높이의 50%와 기본값 중 큰 값 사용
        adaptive_tol = max(y_tol, avg_height * 0.5)
        
        if abs(it['y'] - avg_y) <= adaptive_tol:
            current.append(it)
        else:
            rows.append(current)
            current = [it]
    
    rows.append(current)
    
    # combine texts sorted by x
    combined = []
    for row in rows:
        row.sort(key=lambda i: i['x'])
        combined.append(' '.join(i['text'] for i in row))
    
    return combined


def smart_split(line: str) -> list:
    """
    Split a line by whitespace outside of parentheses,
    so tokens like 'Vu(kN,LCB, iWAL, Lw)' stay intact.
    """
    tokens = []
    buf = ''
    depth = 0
    for ch in line:
        if ch == '(':
            depth += 1
            buf += ch
        elif ch == ')':
            depth -= 1
            buf += ch
        elif ch.isspace() and depth == 0:
            if buf:
                tokens.append(buf)
                buf = ''
        else:
            buf += ch
    if buf:
        tokens.append(buf)
    return tokens

# ===================== 주요 함수 =====================


def clean_math_tags(text):
    """모든 수학 표기법을 정리"""
    # LaTeX 명령어들
    latex_patterns = [
        (r'\\mathbf\{([^}]+)\}', r'\1'),      # 굵은 글씨
        (r'\\mathrm\{([^}]+)\}', r'\1'),      # 로만체
        (r'\\mathit\{([^}]+)\}', r'\1'),      # 이탤릭
        (r'\\text\{([^}]+)\}', r'\1'),        # 일반 텍스트
        (r'\\textbf\{([^}]+)\}', r'\1'),      # 굵은 텍스트
        (r'\\[a-zA-Z]+\{([^}]+)\}', r'\1'),   # 기타 LaTeX 명령어
    ]
    
    for pattern, replacement in latex_patterns:
        text = re.sub(pattern, replacement, text)
    
    # MathML 태그
    text = re.sub(r'<math>(.*?)</math>', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def smart_split(line: str) -> list:
    """
    괄호 밖 공백만 분리하여 ( ) 내부는 하나의 토큰으로 유지
    """
    tokens, buf, depth = [], '', 0
    for ch in line:
        if ch == '(':
            depth += 1
            buf += ch
        elif ch == ')':
            depth -= 1
            buf += ch
        elif ch.isspace() and depth == 0:
            if buf:
                tokens.append(buf)
                buf = ''
        else:
            buf += ch
    if buf:
        tokens.append(buf)
    return tokens

def create_table_rows_csv():
    """OCR 분할 결과에서 테이블 행을 추출하여 table_rows.csv 생성"""
    split_dir = os.path.join(TABLE_DIR, "table_OCR_split")
    csv_path = os.path.join(TABLE_DIR, "table_rows.csv")
    
    all_rows = []
    
    for filename in os.listdir(split_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(split_dir, filename)
            rows = extract_table_rows(json_path)
            
            # 부재명 추출 (파일명에서)
            member_match = re.search(r'_([^_]+)\.json$', filename)
            member_id = member_match.group(1) if member_match else 'unknown'
            
            for row_text in rows:
                all_rows.append({
                    'file': filename,
                    'member_id': member_id,  # 🔥 부재명 컬럼 추가
                    'text': row_text
                })
    
    # CSV로 저장
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 🔥 한글 인코딩 수정
    
    return len(all_rows)


def preview_member_data_for_header_selection(csv_path: str, keywords: list) -> dict:
    """
    첫 번째 부재의 텍스트 라인들을 보여주고 사용자가 헤더 행을 선택할 수 있도록 함
    """
    import pandas as pd
    import re
    
    df_raw = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 모든 텍스트를 하나로 합치기
    all_text_lines = []
    for _, row in df_raw.iterrows():
        cleaned_text = clean_math_tags(str(row['text']))
        all_text_lines.append(cleaned_text)
    
    # 키워드가 포함된 라인 찾기
    member_boundaries = []
    for i, line in enumerate(all_text_lines):
        for keyword in keywords:
            if keyword.lower() in line.lower() and '=' in line:
                parts = line.split('=', 1)
                if len(parts) > 1:
                    member_name = parts[1].strip().split()[0]
                    member_boundaries.append({
                        'line_index': i,
                        'member_name': member_name,
                        'keyword': keyword
                    })
                    break
    
    if not member_boundaries:
        return {}
    
    # 첫 번째 부재의 텍스트 라인들 추출
    first_boundary = member_boundaries[0]
    start_line = first_boundary['line_index']
    end_line = member_boundaries[1]['line_index'] if len(member_boundaries) > 1 else len(all_text_lines)
    
    member_lines = all_text_lines[start_line:end_line]
    
    return {
        'member_name': first_boundary['member_name'],
        'lines': member_lines,
        'total_members': len(member_boundaries)
    }

def process_csv_to_excel_with_custom_header(csv_path: str, excel_path: str, keywords: list, header_pattern: str) -> int:
    """
    CSV에서 사용자 지정 키워드와 커스텀 헤더 패턴으로 부재를 구분하여 Excel 저장
    """
    import pandas as pd
    import re
    
    # CSV 로드
    df_raw = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 필수 컬럼 확인
    if not all(col in df_raw.columns for col in ['file', 'member_id', 'text']):
        raise ValueError("CSV에 필수 컬럼(file, member_id, text)이 없습니다.")
    
    # 모든 텍스트를 하나로 합치기 (순서 유지)
    all_text_lines = []
    for _, row in df_raw.iterrows():
        cleaned_text = clean_math_tags(str(row['text']))
        all_text_lines.append(cleaned_text)
    
    # 키워드가 포함된 라인 찾기
    member_boundaries = []
    
    print(f"전체 라인 수: {len(all_text_lines)}")
    print(f"찾을 키워드: {keywords}")
    print(f"헤더 패턴: {header_pattern}")
    
    for i, line in enumerate(all_text_lines):
        for keyword in keywords:
            if keyword.lower() in line.lower() and '=' in line:
                parts = line.split('=', 1)
                if len(parts) > 1:
                    member_name = parts[1].strip().split()[0]
                    member_boundaries.append({
                        'line_index': i,
                        'member_name': member_name,
                        'keyword': keyword,
                        'original_line': line
                    })
                    print(f"찾은 패턴 - 라인 {i}: {line.strip()}")
                    print(f"추출된 부재명: {member_name}")
                    break
    
    if not member_boundaries:
        print("Warning: 키워드 패턴을 찾지 못했습니다.")
        return 0
    
    # 커스텀 헤더 패턴으로 헤더 찾기
    header_words = header_pattern.split()[:2]
    header_pat = re.compile(r"\s+".join(re.escape(word) for word in header_words), re.IGNORECASE)
    
    eq_pat = re.compile(r"\*\.\s*(.*?)\s*=\s*(.*)")
    col_pat = re.compile(r"\*\.\s*(.*?)\s*:\s*(.*)")
    floor_pat = re.compile(r'^(?:\d+F|B\d+|PIT)', re.IGNORECASE)
    
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        sheet_count = 0
        
        for idx, boundary in enumerate(member_boundaries):
            start_line = boundary['line_index']
            end_line = member_boundaries[idx + 1]['line_index'] if idx + 1 < len(member_boundaries) else len(all_text_lines)
            member_name = boundary['member_name']
            
            member_lines = all_text_lines[start_line:end_line]
            
            print(f"처리 중: {member_name} (라인 {start_line}~{end_line-1}, 총 {len(member_lines)}줄)")
            
            # 커스텀 헤더 패턴으로 헤더 위치 찾기
            header_idxs = [i for i, ln in enumerate(member_lines) if header_pat.search(ln)]
            
            if not header_idxs:
                print(f"Warning: {member_name}에서 헤더 패턴 '{header_pattern}'을 찾지 못했습니다.")
                continue
            
            # 각 헤더에 대해 테이블 처리
            for table_idx, h_idx in enumerate(header_idxs):
                # 1) 헤더 토큰화 - session_state에 저장된 최종 헤더 사용

                if 'final_headers' in st.session_state:
                    headers = st.session_state.final_headers
                    print(f"사용자 정의 헤더 사용: {headers}")
                else:
                    headers = smart_split(member_lines[h_idx].strip())
                    print(f"자동 분할 헤더 사용: {headers}")
                
                # 2) 메타데이터 수집
                meta = {}
                for lookback in range(max(0, h_idx-5), h_idx):
                    line = member_lines[lookback].strip()
                    m = eq_pat.match(line) or col_pat.match(line)
                    if m:
                        key, val = m.group(1).strip(), m.group(2).strip()
                        if key not in headers:
                            meta[key] = val
                
                # 3) 데이터 블록 범위 설정
                next_h = header_idxs[table_idx + 1] if table_idx + 1 < len(header_idxs) else len(member_lines)
                data_rows = []
                
                # 4) 데이터 행 처리 - 개선된 분할 로직
                for ln in member_lines[h_idx + 1:next_h]:
                    txt = ln.strip()
                    if not txt or eq_pat.match(txt) or col_pat.match(txt) or header_pat.search(txt):
                        continue
                    
                    # 개선된 스마트 분할 사용
                    parts = improved_data_split(txt, headers)
                    
                    # 층수 없는 행 스킵
                    if not floor_pat.match(parts[0]):
                        continue
                        
                    data_rows.append(parts)
                
                # 데이터가 없으면 스킵
                if not data_rows:
                    continue
                
                # DataFrame 생성 및 메타데이터 결합
                df = pd.DataFrame(data_rows, columns=headers)
                for k, v in meta.items():
                    df[k] = v
                
                # 시트명 설정
                if len(header_idxs) > 1:
                    sheet_name = f"{member_name}_T{table_idx+1}"
                else:
                    sheet_name = str(member_name)
                
                sheet_name = re.sub(r'[^\w\-]', '_', sheet_name)[:31]
                
                # Excel 시트 작성
                try:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    sheet_count += 1
                    print(f"시트 생성 완료: {sheet_name} ({len(data_rows)}행)")
                except Exception as e:
                    print(f"시트 생성 실패: {sheet_name} - {e}")
                    continue
    
    return sheet_count


def improved_data_split(line: str, headers: list) -> list:
    """
    헤더에 맞춰 데이터를 스마트하게 분할
    빈 컬럼이 있어도 정확한 위치에 매칭
    괄호가 있는 헤더라도 괄호 없는 데이터를 처리
    """
    import re
    
    # 점 기반 스마트 토큰 분할
    raw_tokens = line.strip().split()
    smart_tokens = []
    
    for token in raw_tokens:
        if '.' in token and not token.endswith('.'):
            # 점이 중간에 있는 토큰을 점에서 분할
            parts = token.split('.')
            for i, part in enumerate(parts[:-1]):
                smart_tokens.append(part + '.')
            smart_tokens.append(parts[-1])
        else:
            smart_tokens.append(token)
    
    # 점 뒤 괄호 연결 처리
    final_tokens = []
    i = 0
    while i < len(smart_tokens):
        current = smart_tokens[i]
        
        # 점으로 끝나고 다음 토큰이 괄호로 시작하면 연결
        if (current.endswith('.') and 
            i + 1 < len(smart_tokens) and 
            smart_tokens[i + 1].startswith('(')):
            
            # 괄호가 완전히 닫힐 때까지 연결
            combined = current + ' ' + smart_tokens[i + 1]
            i += 2
            while i < len(smart_tokens) and combined.count('(') > combined.count(')'):
                combined += ' ' + smart_tokens[i]
                i += 1
            final_tokens.append(combined)
        else:
            final_tokens.append(current)
            i += 1
    
    parts = []
    tokens = final_tokens
    token_idx = 0
    
    for i, header in enumerate(headers):
        if token_idx >= len(tokens):
            parts.append("")
            continue
        
        if '(' in header and ')' in header:
            # 괄호가 있는 컬럼: 이미 합쳐진 토큰이거나 일반 괄호 패턴 처리
            current_token = tokens[token_idx]
            
            # 이미 점+괄호로 합쳐진 토큰인지 확인
            if '(' in current_token and ')' in current_token:
                parts.append(current_token)
                token_idx += 1
            else:
                # 기존 로직: 여러 토큰에 걸친 괄호 패턴 찾기
                remaining_text = ' '.join(tokens[token_idx:])
                pattern = r'^(\S+\.?)\s*(\([^)]*\))'
                match = re.search(pattern, remaining_text)
                
                if match:
                    # 괄호 패턴이 있는 경우
                    value = f"{match.group(1)} {match.group(2)}"
                    parts.append(value.strip())
                    
                    # 사용된 토큰 수 계산
                    used_tokens = len(match.group(0).split())
                    token_idx += used_tokens
                else:
                    # 괄호 패턴이 없어도 첫 번째 토큰을 가져감 (예: -23.)
                    parts.append(tokens[token_idx])
                    token_idx += 1
        else:
            # 괄호 없는 컬럼
            if token_idx < len(tokens):
                parts.append(tokens[token_idx])
                token_idx += 1
            else:
                parts.append("")
    
    # 남은 토큰들을 마지막 컬럼에 합치기
    if token_idx < len(tokens):
        remaining = ' '.join(tokens[token_idx:])
        if parts:
            parts[-1] += ' ' + remaining
    
    return parts




def export_summary_csv(excel_input: str, summary_csv: str) -> int:
    """
    Excel 파일의 모든 시트를 읽어서 하나의 CSV로 통합
    각 행에 시트명(부재명) 컬럼을 추가하여 구분
    
    Args:
        excel_input: 입력 Excel 파일 경로
        summary_csv: 출력 CSV 파일 경로
    
    Returns:
        통합된 총 행 수
    """
    import pandas as pd
    
    try:
        # Excel 파일의 모든 시트 읽기
        xls = pd.ExcelFile(excel_input)
        all_dataframes = []
        
        for sheet_name in xls.sheet_names:
            # 각 시트 읽기
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # 빈 DataFrame 건너뛰기
            if df.empty:
                continue
            
            # 부재명 컬럼 추가 (맨 앞에)
            df.insert(0, '부재명', sheet_name)
            
            all_dataframes.append(df)
        
        if not all_dataframes:
            print("통합할 데이터가 없습니다.")
            return 0
        
        # 모든 DataFrame 통합
        consolidated_df = pd.concat(all_dataframes, ignore_index=True)
        
        # CSV로 저장 (한글 인코딩 문제 해결)
        consolidated_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
        
        total_rows = len(consolidated_df)
        print(f"통합 완료: {len(all_dataframes)}개 시트, 총 {total_rows}행")
        print(f"저장 위치: {summary_csv}")
        
        return total_rows
        
    except Exception as e:
        print(f"통합 중 오류 발생: {e}")
        return 0



































# ==========================




st.set_page_config(page_title="PDF to PNG Converter", layout="wide")
st.title("PDF to PNG Converter")
tab1, tab2, tab3 = st.tabs(["SCD Wall_info extraction", "SDD Wall_info extraction", "Human Error detection"])




with tab1:

    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if uploaded_file:
        if st.button("공통 - Step 0: PDF → 이미지 변환"):
            with st.spinner("📄 PDF → 이미지 변환 중..."):
                pdf_bytes = uploaded_file.read()
                images = convert_from_bytes(pdf_bytes, dpi=500)

                image_paths = []
                os.makedirs(WALL_IMAGE_DIR, exist_ok=True)
                for idx, image in enumerate(images):
                    img_path = os.path.join(WALL_IMAGE_DIR, f"page_{idx + 1}.png")
                    image.save(img_path, "PNG")
                    image_paths.append(img_path)

            st.success(f"✅ 변환 완료! 총 {len(image_paths)} 페이지")


    conf_thresh = st.slider("🔧 CONF_THRESH (YOLO confidence threshold)", 0.1, 0.5, 0.3, 0.1)

    # Step 2: 문서 유형 예측
    if st.button("공통 - Step 1: 문서 유형 예측 및 저장"):
        image_paths = sorted(glob.glob(os.path.join(WALL_IMAGE_DIR, "page_*.png")))
        if not image_paths:
            st.warning("⚠️ 먼저 PDF를 이미지로 변환해주세요.")
        else:
            with st.spinner(f"🧠 문서 유형 예측 중... (CONF_THRESH={conf_thresh})"):
                results = predict_document_type(image_paths, conf_thresh)
                save_images_by_prediction(results)

            sampled_results = random.sample(results, min(5, len(results)))
            st.success("✅ 예측된 이미지를 유형별로 저장했습니다.")
            st.markdown("### 🔍 예측 결과 카드 (랜덤 5장)")

            for result in sampled_results:
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(Image.open(result["image_path"]), width=200)
                    with col2:
                        st.markdown(f"#### 🧾 예측된 문서 유형: `{result['label']}`")
                        st.markdown(f"**파일명:** `{os.path.basename(result['image_path'])}`")
                        st.markdown("---")


    with st.expander("MIDAS Step 1 : 이미지 레이아웃 추출"):
        if st.button("🚀 분석 및 크롭 시작"):
            run_midas_analysis()


        
    with st.expander("MIDAS Step 2 : 크롭된 이미지 OCR(plain text)"):
        if st.button("Aplying OCR"):
            apply_surya_ocr_to_plain_text()
            apply_surya_ocr_to_figure()
    


    with st.expander("MIDAS Step 3 : OCR 바운딩박스 기반 정보 추출"):
        render_keyword_based_ocr_extraction()
    ###################################################################################################
    #BeST

    with st.expander("BeST Step 1: 레이아웃 분석 & 크롭"):
        if st.button("🚀 분석 시작 (BeST)"):
            apply_best_layout_analysis()

    with st.expander("BeST Step 2: OCR 실행"):
        if st.button("🚀 BeST OCR"):
            apply_surya_ocr_to_BeST()




    #######################################################################################################
    ######################################################################################################
    #Table 형싣ㄱ


    with st.expander("Table Step 1 : 레이아웃 분석 실행"):
        if st.button("🚀 TABLE 레이아웃 분석 시작"):
            analyze_table_docs()
            st.success("✅ TABLE 분석 및 시각화 완료!")


    with st.expander("Table Step 2 : OCR"):
        if st.button("🚀 TABLE OCR 시작"):
            apply_surya_ocr_table()
            st.success("✅ DONE!")






# 기존 Table Step 2 다음에 추가
    with st.expander("Table Step 2.5 : OCR 결과 사전 필터링"):
        st.write("분석할 키워드가 포함된 OCR 파일들만 필터링하여 처리 시간을 단축합니다.")
        st.write("키워드 입력 시 구분어만 작성하세요. 예) MEMB, Wall Mark")
        
        # 키워드 입력
        filter_kw1 = st.text_input("첫 번째 필터링 키워드", value="MEMB", key="filter_kw1")
        filter_kw2 = st.text_input("두 번째 필터링 키워드", value="Wall Mark", key="filter_kw2")
        
        # 추가 키워드들 (선택사항)
        additional_keywords = st.text_area(
            "추가 키워드 (쉼표로 구분)", 
            placeholder="예: keyword1, keyword2, keyword3",
            key="additional_filter_keywords"
        )
        
        # 폴더 경로 설정
        filter_input_dir = st.text_input(
            "OCR JSON 폴더 경로", 
            value=os.path.join(TABLE_DIR, "table_OCR"),
            key="filter_input_dir"
        )
        filter_output_dir = st.text_input(
            "필터링된 파일 저장 폴더", 
            value=os.path.join(TABLE_DIR, "table_OCR_filtered"),
            key="filter_output_dir"
        )
        
        if st.button("🔍 키워드 기반 필터링 실행"):
            # 키워드 리스트 구성
            filter_keywords = []
            if filter_kw1.strip():
                filter_keywords.append(filter_kw1.strip())
            if filter_kw2.strip():
                filter_keywords.append(filter_kw2.strip())
            
            # 추가 키워드 처리
            if additional_keywords.strip():
                additional_list = [kw.strip() for kw in additional_keywords.split(',') if kw.strip()]
                filter_keywords.extend(additional_list)
            
            if not filter_keywords:
                st.error("최소 하나의 키워드를 입력해주세요.")
            else:
                with st.spinner(f"키워드 '{', '.join(filter_keywords)}' 기반 필터링 중..."):
                    try:
                        filtered_count, total_count = filter_ocr_by_keywords(
                            keywords=filter_keywords,
                            input_dir=filter_input_dir,
                            output_dir=filter_output_dir
                        )
                        
                        st.success(f"""
                        ✅ 필터링 완료!
                        - 전체 파일: {total_count}개
                        - 필터링된 파일: {filtered_count}개
                        - 제외된 파일: {total_count - filtered_count}개
                        - 저장 위치: {filter_output_dir}
                        """)
                        
                        if filtered_count == 0:
                            st.warning("⚠️ 키워드가 포함된 파일이 없습니다. 키워드를 다시 확인해주세요.")
                        
                    except Exception as e:
                        st.error(f"필터링 중 오류 발생: {e}")








    with st.expander("Table Step 3 : OCR Splitting (키워드 저장)"):
        st.write("테이블 OCR JSON을 사용자 키워드로 분할 저장합니다.")
        st.write("키워드 입력 시 구분어만 작성하세요. 예) MEMB, Wall Mark")
        
        kw1 = st.text_input("첫 번째 키워드", value="MEMB")
        kw2 = st.text_input("두 번째 키워드", value="Wall Mark")
        margin = st.number_input("분할 마진(px)", min_value=0, max_value=100, value=5, step=1)
        
        input_dir = os.path.join(TABLE_DIR, "table_OCR_filtered")
        output_dir = os.path.join(TABLE_DIR, "table_OCR_split")
        
        if st.button("분할 실행 및 키워드 저장"):
            # 키워드를 session_state에 저장
            keywords = [kw.strip() for kw in [kw1, kw2] if kw.strip()]
            st.session_state.split_keywords = keywords
            
            # 분할 실행
            n = process_table_ocr_splits(keywords, input_dir, output_dir, margin)
            st.success(f"✅ {n}개의 분할 파일 생성 완료 → {output_dir}")
            st.success(f"키워드 저장 완료: {keywords}")




    with st.expander("Table Step 4.5: 테이블 행 CSV 생성"):
        if st.button("table_rows.csv 생성"):
            n = create_table_rows_csv()
            st.success(f"{n}개 행이 포함된 table_rows.csv 생성 완료")




    with st.expander("Table Step 5 : 테이블 형식으로 정보 저장 (키워드 기반)", expanded=True):
        # Step 3에서 입력받은 키워드 사용 (session_state 활용)
        if 'split_keywords' not in st.session_state:
            st.session_state.split_keywords = ['MEMB', 'Wall Mark']
        
        st.write("Step 3에서 설정한 키워드를 사용하여 부재별 시트를 생성합니다.")
        st.write(f"현재 키워드: {st.session_state.split_keywords}")
        
        # 키워드 수정 옵션
        if st.checkbox("키워드 수정하기"):
            kw1 = st.text_input("첫 번째 키워드", value=st.session_state.split_keywords[0])
            kw2 = st.text_input("두 번째 키워드", value=st.session_state.split_keywords[1] if len(st.session_state.split_keywords) > 1 else "")
            additional_keywords = st.text_area("추가 키워드 (쉼표로 구분)", placeholder="keyword3, keyword4")
            
            if st.button("키워드 업데이트"):
                keywords = [kw.strip() for kw in [kw1, kw2] if kw.strip()]
                if additional_keywords.strip():
                    keywords.extend([kw.strip() for kw in additional_keywords.split(',') if kw.strip()])
                st.session_state.split_keywords = keywords
                st.success(f"키워드 업데이트 완료: {keywords}")
        
        # 파일 경로 설정
        csv_path = st.text_input("CSV File Path", value=os.path.join(TABLE_DIR, "table_rows.csv"))
        excel_path = st.text_input("Output Excel Path", value=os.path.join(TABLE_DIR, "output.xlsx"))
        
        # 미리보기 데이터 상태 초기화
        if 'preview_data' not in st.session_state:
            st.session_state.preview_data = None
        
        # 부재 데이터 미리보기 및 헤더 선택
        if st.button("부재 데이터 미리보기 및 헤더 선택"):
            # 파일 존재 확인
            if not os.path.exists(csv_path):
                st.error(f"CSV 파일이 존재하지 않습니다: {csv_path}")
                st.info("먼저 Table Step 4.5에서 table_rows.csv를 생성해주세요.")
            else:
                try:
                    st.session_state.preview_data = preview_member_data_for_header_selection(csv_path, st.session_state.split_keywords)
                except Exception as e:
                    st.error(f"데이터 미리보기 오류: {e}")
        
        # 미리보기 데이터가 있으면 표시
        if st.session_state.preview_data:
            preview_data = st.session_state.preview_data
            st.subheader(f"부재 '{preview_data['member_name']}' 데이터 미리보기")
            st.info(f"총 {preview_data['total_members']}개 부재 발견")
            
            # 라인들을 표시하고 선택할 수 있게 함
            st.write("아래에서 테이블 헤더가 될 행을 선택하세요:")
            
            header_options = []
            for i, line in enumerate(preview_data['lines'][:15]):  # 처음 15줄만 표시
                if line.strip():  # 빈 줄이 아닌 경우만
                    header_options.append(f"라인 {i}: {line[:100]}...")  # 처음 100자만
            
            if header_options:
                selected_header_idx = st.selectbox(
                    "헤더 행 선택:",
                    range(len(header_options)),
                    format_func=lambda x: header_options[x],
                    key="header_selector"
                )
                
                # 선택된 헤더 라인 분석
                selected_line = preview_data['lines'][selected_header_idx]
                st.write("선택된 헤더:")
                st.code(selected_line)
                
                # 헤더 분리해서 보여주기
                initial_headers = smart_split(selected_line.strip())
                st.write("자동 분리된 컬럼들:")
                
                # 편집 가능한 헤더 입력 필드들
                st.write("컬럼 편집 (병합하려면 해당 컬럼을 빈 칸으로 두고 이전 컬럼에 합쳐서 입력):")
                
                if 'edited_headers' not in st.session_state:
                    st.session_state.edited_headers = initial_headers.copy()
                
                # 헤더가 바뀌면 편집된 헤더도 초기화
                if len(st.session_state.edited_headers) != len(initial_headers):
                    st.session_state.edited_headers = initial_headers.copy()
                
                edited_headers = []
                cols = st.columns(3)  # 3열로 배치
                for i, header in enumerate(st.session_state.edited_headers):
                    with cols[i % 3]:
                        edited_header = st.text_input(
                            f"컬럼 {i+1}", 
                            value=header, 
                            key=f"header_edit_{i}",
                            help="빈 칸으로 두면 이전 컬럼과 병합됩니다"
                        )
                        edited_headers.append(edited_header)
                
                # 빈 칸 제거 및 병합 처리
                final_headers = []
                temp_header = ""
                for header in edited_headers:
                    if header.strip():  # 빈 칸이 아니면
                        if temp_header:  # 이전에 누적된 것이 있으면
                            temp_header += " " + header.strip()
                        else:
                            temp_header = header.strip()
                        final_headers.append(temp_header)
                        temp_header = ""
                    else:  # 빈 칸이면 이전 헤더에 누적
                        if final_headers:  # 이전 헤더가 있으면
                            continue  # 다음 반복에서 병합
                
                st.write("최종 헤더:")
                for i, header in enumerate(final_headers):
                    st.write(f"{i+1}. `{header}`")
                
                st.write(f"총 {len(final_headers)}개 컬럼")
                
                # 헤더 재설정 버튼
                if st.button("원래대로 되돌리기"):
                    st.session_state.edited_headers = initial_headers.copy()
                    st.experimental_rerun()
                
                # session_state에 헤더 패턴 저장
                if st.button("이 헤더로 설정"):
                    if len(final_headers) >= 2:
                        first_words = final_headers[:2]
                        header_pattern = " ".join(first_words)
                        st.session_state.custom_header_pattern = header_pattern
                        st.session_state.final_headers = final_headers  # 최종 헤더도 저장
                        st.success(f"헤더 패턴 설정 완료: '{header_pattern}'")
                        st.success(f"총 {len(final_headers)}개 컬럼으로 설정됨")
                    else:
                        st.error("최소 2개 이상의 컬럼이 필요합니다.")
            else:
                st.warning("표시할 데이터가 없습니다.")
        
        st.markdown("---")
        
        # 설정된 헤더 패턴 표시
        if 'custom_header_pattern' in st.session_state:
            st.write(f"설정된 헤더 패턴: `{st.session_state.custom_header_pattern}`")
            if 'final_headers' in st.session_state:
                st.write("설정된 컬럼들:")
                header_display = " | ".join(st.session_state.final_headers)
                st.code(header_display)
        
        # Excel 생성 버튼
        if st.button("키워드 기반 Excel 생성"):
            # 파일 존재 확인
            if not os.path.exists(csv_path):
                st.error(f"CSV 파일이 존재하지 않습니다: {csv_path}")
                st.info("먼저 Table Step 4.5에서 table_rows.csv를 생성해주세요.")
            elif 'custom_header_pattern' not in st.session_state:
                st.error("먼저 헤더를 선택해주세요.")
            else:
                try:
                    n = process_csv_to_excel_with_custom_header(
                        csv_path=csv_path, 
                        excel_path=excel_path, 
                        keywords=st.session_state.split_keywords,
                        header_pattern=st.session_state.custom_header_pattern
                    )
                    st.success(f"{n} sheets exported to {excel_path}")
                except Exception as e:
                    st.error(f"Error: {e}")




    with st.expander("Table Step 6 : 최종 정리 파일 저장"):
        excel_input = st.text_input("Input Excel File Path", value=os.path.join(TABLE_DIR, "output.xlsx"))
        summary_csv = st.text_input("Summary CSV Path", value=os.path.join(TABLE_DIR,"summary.csv"))
        if st.button("Export Summary CSV"):
            try:
                m = export_summary_csv(excel_input, summary_csv)
                st.success(f"Summary CSV with {m} rows saved to {summary_csv}")
            except Exception as e:
                st.error(f"Error: {e}")















with tab2: 
    with st.expander("Step 0: PDF→이미지", expanded=True):
        uploaded = st.file_uploader("PDF 업로드", type=["pdf"])
        if uploaded:
            if st.button("변환 실행"):
                try:
                    pdf_bytes = uploaded.read()
                    paths = convert_pdf_to_png(pdf_bytes)
                    st.success(f"{len(paths)}페이지 변환 완료: '{Wall_rawdata_SD}' 확인하세요")
                except Exception as e:
                    st.error(f"변환 중 오류 발생: {e}")
        else:
            st.info("PDF 파일을 업로드하세요.")


    with st.expander("바운딩박스"):
        # "바운딩박스 시작" 버튼을 누르면 UI가 나타나게
        if "show_box_ui" not in st.session_state:
            st.session_state.show_box_ui = False

        if st.button("바운딩박스 시작"):
            st.session_state.show_box_ui = True

        if st.session_state.show_box_ui:
            draw_and_save_bounding_boxes()
            # "완료하기" 버튼을 넣어서 누르면 UI 닫힘
            if st.button("완료하기"):
                st.session_state.show_box_ui = False




    with st.expander("테이블 셀 자동 크롭 흐름"):
        # 단계 1: OCR
        if st.button("1. 전체 OCR 실행"):
            apply_surya_ocr_Wall_SD()


    with st.expander("OCR 동일선상 보기"):
        if st.button("전체 처리 및 예시 보기"):
            img, path = process_all_and_show_one()
            if img is not None:
                st.image(img, caption=path, use_column_width=True)
            else:
                st.warning("이미지/박스/OCR 셋 다 존재하는 파일이 폴더에 없음")

    with st.expander("동일 선상 인식 및 교차점 인식"):
        if st.button("텍스트 인식 및 교차점 인식"):
            visualize_lines_batch()




    with st.expander("🔽 배치 처리 및 엑셀 저장"):
        st.write("이미지 폴더에서 테이블을 검출하고 각 부재명별 시트로 output.xlsx 를 생성합니다")
        # ② 버튼 생성
        if st.button("▶️ 실행"):
            with st.spinner("처리 중..."):
                process_batch()
            st.success("완료! output.xlsx 파일을 확인하세요")



    with st.expander("크롭된 이미지 OCR"):
        # 단계 1: OCR
        if st.button("OCR 작동"):
            apply_surya_ocr_Wall_Crop_SD()


    # with st.expander("수평/수직선 + 교차점 인식", expanded=True):
    #     img_dir = st.text_input("입력 이미지 폴더 경로", value=r"D:\wall_drawing\Wall_table_region")
    #     save_dir = st.text_input("결과 저장 폴더명 (비우면 저장 X)", value="Wall_table_region/lines_detected")
    #     mode = st.radio("선 검출 방식", options=["contour", "hough"], index=0)
    #     min_line_length = st.slider("min_line_length", 20, 300, 80, 5)
    #     max_line_gap = st.slider("max_line_gap", 2, 40, 10, 2)
    #     hough_threshold = st.slider("Hough threshold", 30, 300, 100, 5)
    #     morph_kernel_scale = st.slider("Morph kernel 비율", 10, 60, 30, 2)
    #     resize_scale = st.slider("이미지 축소 비율 (1=원본)", 0.3, 1.0, 0.7, 0.05)
    #     block_size = st.slider("adaptiveThreshold blockSize(홀수)", 7, 31, 15, 2)
    #     C = st.slider("adaptiveThreshold C", -10, 10, -2, 1)

    #     run_btn = st.button("예시 5개 보기")
    #     save_btn = st.button("전체 저장")

    #     if run_btn:
    #         imgs = detect_and_draw_lines_batch(
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
    #             C=C
    #         )
    #         if not imgs:
    #             st.warning("이미지가 없습니다")
    #         else:
    #             for fname, img, inter_cnt in imgs:
    #                 st.subheader(f"{fname} (교차점: {inter_cnt}개)")
    #                 st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="검출 결과", use_column_width=True)
    #     if save_btn:
    #         detect_and_draw_lines_batch(
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
    #             C=C
    #         )
    #         st.success(f"전체 이미지 결과 저장 완료! → '{save_dir}' 폴더 확인")


    # with st.expander("OCR → 엑셀 일괄 변환"):
    #     if st.button("일괄 변환 실행"):
    #         batch_ocr_json_to_excel()
    #         st.success("폴더 내 모든 OCR json → 엑셀 변환 완료!")


    with st.expander("수평/수직선 + 교차점 인식", expanded=True):
        img_dir = st.text_input("입력 이미지 폴더 경로", value=os.path.join(Wall_table_region))
        save_dir = st.text_input("결과 저장 폴더명 (비우면 저장 X)", value=os.path.join(Wall_table_region,"lines_detected"))
        mode = st.radio("선 검출 방식", options=["contour", "hough"], index=0)
        min_line_length = st.slider("min_line_length", 20, 300, 80, 5)
        max_line_gap = st.slider("max_line_gap", 2, 40, 10, 2)
        hough_threshold = st.slider("Hough threshold", 30, 300, 100, 5)
        morph_kernel_scale = st.slider("Morph kernel 비율", 10, 60, 30, 2)
        resize_scale = st.slider("이미지 축소 비율 (1=원본)", 0.3, 1.0, 0.7, 0.05)
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
                resize_scale=resize_scale,
                max_examples=5,
                return_imgs=True,
                mode=mode,
                block_size=block_size | 1,
                C=C,
                tol=tol
            )
            if not imgs:
                st.warning("이미지가 없습니다")
            else:
                for fname, img, inter_cnt in imgs:
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
                resize_scale=resize_scale,
                max_examples=None,
                return_imgs=False,
                mode=mode,
                block_size=block_size | 1,
                C=C,
                tol=tol
            )
            st.success(f"전체 이미지 결과 저장 완료! → '{save_dir}' 폴더 확인")

    # ---------- (3) 표 추출 엑셀화 expander 추가 ----------

    # with st.expander("2) OCR+교차점 기반 표 엑셀 추출", expanded=True):
    #     ocr_dir = st.text_input("OCR json 폴더", value=r"D:\wall_drawing\Wall_cell_crop_ocr")
    #     line_dir = st.text_input("라인 pkl 폴더", value=r"D:\wall_drawing\Wall_table_region\lines_detected")
    #     out_dir  = st.text_input("엑셀 저장 폴더", value=r"D:\wall_drawing\Wall_table_region\Excel_Results")
    #     if st.button("표 추출 실행"):
    #         batch_extract_dynamic(ocr_dir, line_dir, out_dir)
    #         st.success(f"표 추출 완료 → {out_dir}")

    with st.expander("3) 통합 엑셀 파일 생성", expanded=True):
        img_dir_cons = st.text_input("입력 이미지 폴더", value= Wall_table_region)
        save_line_dir_cons = st.text_input("라인 pkl 폴더", value=os.path.join(Wall_table_region, "lines_detected"))
        ocr_dir_cons = st.text_input("OCR json 폴더", value=Wall_cell_crop_ocr)
        consolidated_path = st.text_input("통합 엑셀 파일 경로", value=os.path.join(Wall_table_region, "All_In_One.xlsx"))
        if st.button("통합 엑셀 추출"):
            batch_extract_consolidated(
                img_dir=img_dir_cons,
                save_line_dir=save_line_dir_cons,
                ocr_dir=ocr_dir_cons,
                consolidated_path=consolidated_path,
                mode=mode,
                min_line_length=min_line_length,
                max_line_gap=max_line_gap,
                hough_threshold=hough_threshold,
                morph_kernel_scale=morph_kernel_scale,
                resize_scale=resize_scale,
                block_size=block_size|1,
                C=C
            )
            st.success(f"통합 엑셀 파일 생성 완료 → {consolidated_path}")

