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

    # 2단계: 패키지 설치 확인
    st.subheader("2️⃣ Surya 패키지 설치 확인")
    if st.button("설치된 패키지 확인", key="package_check"):
        try:
            import pkg_resources
            installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
            
            # Surya 관련 패키지 찾기
            surya_packages = {name: version for name, version in installed_packages.items() 
                            if 'surya' in name.lower()}
            
            st.write("🔍 Surya 관련 설치된 패키지:")
            if surya_packages:
                for pkg, ver in surya_packages.items():
                    st.success(f"✅ {pkg}: {ver}")
            else:
                st.error("❌ Surya 관련 패키지를 찾을 수 없습니다")
            
            # 전체 패키지 개수
            st.write(f"📦 총 설치된 패키지: {len(installed_packages)}개")
            
            # surya 모듈 직접 확인
            st.write("\n🔍 surya 모듈 구조 확인:")
            try:
                import surya
                st.success(f"✅ surya 패키지 임포트 성공")
                st.write(f"📍 surya 위치: {surya.__file__}")
                
                # surya 하위 모듈들 확인
                import os
                surya_dir = os.path.dirname(surya.__file__)
                submodules = [f for f in os.listdir(surya_dir) 
                            if os.path.isdir(os.path.join(surya_dir, f)) and not f.startswith('__')]
                st.write(f"📁 surya 하위 모듈들: {submodules}")
                
                # __all__ 속성 확인
                if hasattr(surya, '__all__'):
                    st.write(f"🔧 surya.__all__: {surya.__all__}")
                
            except ImportError as e:
                st.error(f"❌ surya 패키지 임포트 실패: {e}")
                
        except Exception as e:
            st.error(f"❌ 패키지 확인 실패: {str(e)}")
            st.code(traceback.format_exc())

    # 3단계: 개별 모듈 임포트 테스트
    st.subheader("3️⃣ 개별 Surya 모듈 임포트 테스트")
    if st.button("개별 모듈 테스트", key="individual_import"):
        memory_start = get_memory_usage()
        st.write(f"🔄 시작 메모리: {memory_start:.2f} GB")
        
        modules_to_test = [
            ("surya", "기본 패키지"),
            ("surya.ocr", "OCR 모듈"),
            ("surya.input", "Input 모듈"),
            ("surya.input.load", "Load 모듈"),
            ("surya.model", "Model 모듈"),
            ("surya.model.detection", "Detection 모듈"),
            ("surya.model.recognition", "Recognition 모듈"),
            ("surya.postprocessing", "Postprocessing 모듈")
        ]
        
        success_count = 0
        for module_name, description in modules_to_test:
            try:
                __import__(module_name)
                st.success(f"✅ {description} ({module_name})")
                success_count += 1
            except ImportError as e:
                st.error(f"❌ {description} ({module_name}): {e}")
            except Exception as e:
                st.warning(f"⚠️ {description} ({module_name}): {e}")
        
        st.write(f"📊 성공률: {success_count}/{len(modules_to_test)}")
        
        memory_end = get_memory_usage()
        st.write(f"📊 현재 메모리: {memory_end:.2f} GB (+{memory_end-memory_start:.2f} GB)")

    # 4단계: 함수별 임포트 테스트  
    st.subheader("4️⃣ 구체적 함수 임포트 테스트")
    if st.button("함수 임포트 테스트", key="function_import"):
        memory_start = get_memory_usage()
        st.write(f"🔄 시작 메모리: {memory_start:.2f} GB")
        
        functions_to_test = [
            ("surya.ocr", "run_ocr", "OCR 실행 함수"),
            ("surya.input.load", "load_from_folder", "폴더 로딩"),
            ("surya.input.load", "load_from_file", "파일 로딩"),
            ("surya.model.detection.model", "load_model", "Detection 모델"),
            ("surya.model.detection.model", "load_processor", "Detection 프로세서"),
            ("surya.model.recognition.model", "load_model", "Recognition 모델"),
            ("surya.model.recognition.processor", "load_processor", "Recognition 프로세서")
        ]
        
        success_functions = []
        for module_name, func_name, description in functions_to_test:
            try:
                module = __import__(module_name, fromlist=[func_name])
                func = getattr(module, func_name)
                st.success(f"✅ {description}: {module_name}.{func_name}")
                success_functions.append(f"{module_name}.{func_name}")
            except ImportError as e:
                st.error(f"❌ {description}: 모듈 {module_name} 임포트 실패 - {e}")
            except AttributeError as e:
                st.error(f"❌ {description}: 함수 {func_name} 없음 - {e}")
            except Exception as e:
                st.warning(f"⚠️ {description}: 기타 오류 - {e}")
        
        st.write(f"📊 성공한 함수들:")
        for func in success_functions:
            st.write(f"  - {func}")
            
        memory_end = get_memory_usage()
        st.write(f"📊 현재 메모리: {memory_end:.2f} GB (+{memory_end-memory_start:.2f} GB)")

    # 5단계: 모델 로딩 테스트 (위험)
    st.subheader("5️⃣ 모델 로딩 테스트 ⚠️")
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
        st.write("- **2단계 성공**: Surya 패키지 제대로 설치됨") 
        st.write("- **3단계 성공**: Surya 모듈들 사용 가능")
        st.write("- **4단계 성공**: Surya 함수들 사용 가능")
        st.write("- **5단계 성공**: 실제 OCR 작업 수행 가능")
        
    with col2:
        st.subheader("📊 메모리 기준")
        st.write("- **< 0.5GB**: 안전 영역")
        st.write("- **0.5-0.7GB**: 주의 영역")
        st.write("- **> 0.7GB**: 위험 영역")
        st.write("- **> 0.8GB**: 크래시 가능")

    st.info("""
    💡 **테스트 순서**:
    1. 먼저 '기본 라이브러리 테스트' 실행
    2. '설치된 패키지 확인'으로 surya-ocr 설치 상태 점검
    3. '개별 모듈 테스트'로 어느 모듈에서 문제 생기는지 확인
    4. '신버전 구조 테스트'로 0.9.0 버전의 실제 함수들 확인
    5. '모델 함수 탐색'으로 사용 가능한 모든 함수 목록 확인
    6. 메모리가 충분하면 '신버전 모델 로딩 테스트' 실행 (선택)
    
    ⚠️ **주의사항**: 
    - 각 단계에서 메모리 사용량을 꼭 확인하세요
    - 0.7GB 초과 시 다음 단계를 진행하지 마세요
    - 앱이 크래시되면 스트림릿에서 재부팅하세요
    """)

if __name__ == "__main__":
    main()
