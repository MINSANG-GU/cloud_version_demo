import streamlit as st
from .config import setup_sidebar_config, get_raw_data_folder, get_output_folder

# 페이지 기본 설정
st.set_page_config(
    page_title="📑 스크립트 실행 플랫폼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바에 경로 설정 추가
setup_sidebar_config()

# 메인 타이틀 및 설명
st.title("📑 SCD & SDD Extraction Dashboard")
st.markdown(
    """
    환영합니다!  
    이 플랫폼에서는 `pages` 폴더에 있는 스크립트(예: Beam Column, Wall 등)를  
    사이드바에서 직접 선택하여 실행할 수 있습니다.  
    각 메뉴를 클릭해 필요한 기능을 사용하세요.
    """
)

# 현재 경로 설정 표시 (선택사항이지만 있으면 좋아요!)
st.markdown("### 📂 현재 작업 환경")
col1, col2 = st.columns(2)

with col1:
    st.info(f"**📁 결과 저장 폴더 **\n`{get_raw_data_folder()}`")

with col2:

    st.info(f"**📤 OCR_결과 폴더 **\n`{get_output_folder()}`")
