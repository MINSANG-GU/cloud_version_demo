import streamlit as st
import os
import json

# 기본 경로 설정
DEFAULT_BASE_DIR = r"C:\SCD_Project"
DEFAULT_OCR_RESULTS_FOLDER = r"C:\SCD_Project\ocr_results"
CONFIG_FILE = "app_config.json"

def save_config_to_file():
    """설정 저장"""
    config = {
        'base_dir': st.session_state.base_dir,
        'ocr_results_folder': st.session_state.ocr_results_folder
    }
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 저장 실패해도 무시

def load_config_from_file():
    """설정 로드"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        st.session_state.base_dir = config.get('base_dir', DEFAULT_BASE_DIR)
        st.session_state.ocr_results_folder = config.get('ocr_results_folder', DEFAULT_OCR_RESULTS_FOLDER)
    except:
        st.session_state.base_dir = DEFAULT_BASE_DIR
        st.session_state.ocr_results_folder = DEFAULT_OCR_RESULTS_FOLDER

def init_session_state():
    """Session state 초기화"""
    if 'base_dir' not in st.session_state:
        load_config_from_file()
    if 'ocr_results_folder' not in st.session_state:
        st.session_state.ocr_results_folder = DEFAULT_OCR_RESULTS_FOLDER

def get_base_dir():
    """현재 설정된 base directory 반환"""
    init_session_state()
    return st.session_state.base_dir

def get_ocr_results_folder():
    """현재 설정된 OCR 결과 폴더 반환"""
    init_session_state()
    return st.session_state.ocr_results_folder

def apply_path_changes(base_dir, ocr_results_folder):
    try:
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(ocr_results_folder, exist_ok=True)
        
        st.session_state.base_dir = base_dir
        st.session_state.ocr_results_folder = ocr_results_folder
        
        save_config_to_file()
        st.success("✅ 경로가 적용되었습니다!")
        # st.experimental_rerun() 제거해도 됨
        
    except Exception as e:
        st.error(f"❌ 경로 설정 실패: {str(e)}")

def reset_to_default():
    """기본 경로로 리셋"""
    st.session_state.base_dir = DEFAULT_BASE_DIR
    st.session_state.ocr_results_folder = DEFAULT_OCR_RESULTS_FOLDER
    save_config_to_file()
    st.success("🔄 기본 경로로 초기화!")
    st.experimental_rerun()

def setup_sidebar_config():
    """사이드바에 경로 설정 추가"""
    init_session_state()
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔧 경로 설정")
        
        # 설정 영역을 expander로 축소 가능하게
        with st.expander("📁 폴더 경로", expanded=False):
            # Base Directory
            new_base_dir = st.text_input(
                "Base Directory (프로젝트 기본 폴더)",
                value=st.session_state.base_dir,
                help="전체 프로젝트의 기본 폴더",
                key="base_dir_input"
            )
            
            # OCR 결과 폴더
            new_ocr_folder = st.text_input(
                "OCR 결과 폴더", 
                value=st.session_state.ocr_results_folder,
                help="OCR 처리 결과가 저장될 폴더",
                key="ocr_folder_input"
            )
            
            # 적용/리셋 버튼
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 적용", key="apply_config"):
                    apply_path_changes(new_base_dir, new_ocr_folder)
            
            with col2:
                if st.button("🔄 기본값", key="reset_config"):
                    reset_to_default()
        
        # 현재 경로 상태 간단 표시
        st.markdown("**📂 현재 설정**")
        base_exists = os.path.exists(st.session_state.base_dir)
        ocr_exists = os.path.exists(st.session_state.ocr_results_folder)
        
        st.markdown(f"Base: {'🟢' if base_exists else '🔴'} `{os.path.basename(st.session_state.base_dir)}`")
        st.markdown(f"OCR: {'🟢' if ocr_exists else '🔴'} `{os.path.basename(st.session_state.ocr_results_folder)}`")

# ==========================
# 편의 함수들 (기존 코드와의 호환성을 위해)
# ==========================

def get_raw_data_folder():
    """raw_data 폴더 경로 (base_dir 하위)"""
    base = get_base_dir()
    return os.path.join(base, "raw_data")

def get_output_folder():
    """output 폴더 경로 (base_dir 하위)"""
    base = get_base_dir()
    return os.path.join(base, "output")