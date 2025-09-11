import streamlit as st
import sys
import psutil
import os
import traceback

def get_memory_usage():
    """현재 메모리 사용량을 GB 단위로 반환"""
    return psutil.virtual_memory().used / (1024**3)

def main():
    st.title("🔍 Surya OCR 스트림릿 클라우드 호환성 테스트")
    st.write("datalab-to/surya 리포지토리가 스트림릿 클라우드에서 작동하는지 테스트합니다.")
    
    # 시스템 정보 표시
    st.header("📊 시스템 환경")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        st.metric("총 메모리", f"{psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    with col2:
        st.metric("사용가능 메모리", f"{psutil.virtual_memory().available / (1024**3):.2f} GB")
        st.metric("CPU 코어", psutil.cpu_count())
    
    with col3:
        current_memory = get_memory_usage()
        st.metric("현재 사용중", f"{current_memory:.2f} GB")
        
        # 메모리 위험도 표시
        if current_memory > 0.7:
            st.error("⚠️ 메모리 위험")
        elif current_memory > 0.5:
            st.warning("⚠️ 메모리 주의")
        else:
            st.success("✅ 메모리 안전")

    st.divider()
    
    # 테스트 진행
    st.header("🧪 단계별 호환성 테스트")
    
    # 1단계: 기본 의존성 테스트
    st.subheader("1️⃣ 기본 의존성 테스트")
    if st.button("기본 라이브러리 테스트", key="basic"):
        memory_start = get_memory_usage()
        st.write(f"🔄 시작 메모리: {memory_start:.2f} GB")
        
        try:
            # 기본 라이브러리들
            import PIL
            import numpy as np
            import torch
            
            st.success("✅ 기본 라이브러리 임포트 성공")
            st.write(f"- PIL: {PIL.__version__}")
            st.write(f"- NumPy: {np.__version__}")
            st.write(f"- PyTorch: {torch.__version__}")
            st.write(f"- CUDA 사용가능: {torch.cuda.is_available()}")
            
            memory_end = get_memory_usage()
            st.write(f"📊 현재 메모리: {memory_end:.2f} GB (+{memory_end-memory_start:.2f} GB)")
            
        except Exception as e:
            st.error(f"❌ 기본 라이브러리 실패: {str(e)}")
            st.code(traceback.format_exc())

    # 2단계: Surya 모듈 임포트 테스트
    st.subheader("2️⃣ Surya 모듈 임포트 테스트")
    if st.button("Surya 모듈 임포트", key="surya_import"):
        memory_start = get_memory_usage()
        st.write(f"🔄 시작 메모리: {memory_start:.2f} GB")
        
        try:
            # Surya 모듈들 임포트
            from surya.ocr import run_ocr
            from surya.input.load import load_from_folder, load_from_file
            from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            from surya.postprocessing.text import draw_text_on_image
            
            st.success("✅ Surya 모듈 임포트 성공!")
            st.write("📦 성공적으로 임포트된 모듈:")
            st.write("- OCR 실행 함수")
            st.write("- 입력 로딩 함수")
            st.write("- Detection 모델 & 프로세서")  
            st.write("- Recognition 모델 & 프로세서")
            st.write("- 후처리 함수")
            
            memory_end = get_memory_usage()
            st.write(f"📊 현재 메모리: {memory_end:.2f} GB (+{memory_end-memory_start:.2f} GB)")
            
        except Exception as e:
            st.error(f"❌ Surya 모듈 임포트 실패: {str(e)}")
            st.code(traceback.format_exc())

    # 3단계: 모델 로딩 테스트 (위험)
    st.subheader("3️⃣ 모델 로딩 테스트 ⚠️")
    st.warning("⚠️ **위험한 테스트**: 메모리 오버플로우로 앱이 크래시될 수 있습니다!")
    
    danger_check = st.checkbox("위험을 감수하고 모델 로딩 테스트 진행")
    
    if danger_check and st.button("🚨 모델 로딩 테스트", key="model_load"):
        memory_start = get_memory_usage()
        st.write(f"🔄 시작 메모리: {memory_start:.2f} GB")
        
        if memory_start > 0.6:
            st.error("❌ 메모리 사용량이 너무 높습니다. 테스트를 중단합니다.")
            st.stop()
        
        try:
            with st.spinner("모델 로딩 중... (30초 이상 소요될 수 있습니다)"):
                from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
                
                # Detection 모델 먼저 로딩
                det_processor = load_det_processor()
                st.write("✅ Detection Processor 로딩 완료")
                
                memory_mid1 = get_memory_usage()
                st.write(f"📊 Processor 후 메모리: {memory_mid1:.2f} GB")
                
                if memory_mid1 > 0.75:
                    st.error("❌ 메모리 한계에 도달. Recognition 모델 로딩을 건너뜁니다.")
                else:
                    det_model = load_det_model()
                    st.write("✅ Detection Model 로딩 완료")
                    
                    memory_mid2 = get_memory_usage()
                    st.write(f"📊 Detection Model 후 메모리: {memory_mid2:.2f} GB")
                    
                    if memory_mid2 < 0.7:
                        from surya.model.recognition.model import load_model as load_rec_model
                        from surya.model.recognition.processor import load_processor as load_rec_processor
                        
                        rec_processor = load_rec_processor()
                        st.write("✅ Recognition Processor 로딩 완료")
                        
                        rec_model = load_rec_model()
                        st.write("✅ Recognition Model 로딩 완료")
                        
                        memory_final = get_memory_usage()
                        st.write(f"📊 최종 메모리: {memory_final:.2f} GB")
                        
                        st.success("🎉 **모든 모델 로딩 성공!** Surya OCR이 스트림릿 클라우드에서 완전히 작동합니다!")
                    else:
                        st.warning("⚠️ 메모리 부족으로 Recognition 모델 로딩을 건너뛰었습니다.")
                        
        except Exception as e:
            st.error(f"❌ 모델 로딩 실패: {str(e)}")
            st.code(traceback.format_exc())

    # 결과 해석 가이드
    st.divider()
    st.header("📋 결과 해석 가이드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ 성공 기준")
        st.write("- **1단계 성공**: 기본 ML 라이브러리 사용 가능")
        st.write("- **2단계 성공**: Surya 라이브러리 사용 가능") 
        st.write("- **3단계 성공**: 실제 OCR 작업 수행 가능")
        
    with col2:
        st.subheader("📊 메모리 기준")
        st.write("- **< 0.5GB**: 안전 영역")
        st.write("- **0.5-0.7GB**: 주의 영역")
        st.write("- **> 0.7GB**: 위험 영역")
        st.write("- **> 0.8GB**: 크래시 가능")

    st.info("""
    💡 **테스트 순서**:
    1. 먼저 '기본 라이브러리 테스트' 실행
    2. 성공하면 'Surya 모듈 임포트' 실행  
    3. 메모리가 충분하면 '모델 로딩 테스트' 실행 (선택)
    
    ⚠️ **주의사항**: 
    - 각 단계에서 메모리 사용량을 꼭 확인하세요
    - 0.7GB 초과 시 다음 단계를 진행하지 마세요
    - 앱이 크래시되면 스트림릿에서 재부팅하세요
    """)

if __name__ == "__main__":
    main()
