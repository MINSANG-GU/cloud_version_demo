import streamlit as st
import sys
import psutil
import os
import traceback

def get_memory_usage():
    """현재 메모리 사용량을 GB 단위로 반환"""
    return psutil.virtual_memory().used / (1024**3)

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
        ("surya.detection", "Detection 모듈"),
        ("surya.recognition", "Recognition 모듈"),
        ("surya.layout", "Layout 모듈"),
        ("surya.table_rec", "Table Recognition 모듈"),
        ("surya.input", "Input 모듈"),
        ("surya.input.load", "Load 모듈"),
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

# 4단계: Predictor 클래스 테스트
st.subheader("4️⃣ Predictor 클래스 임포트 테스트")
if st.button("Predictor 클래스 테스트", key="predictor_test"):
    memory_start = get_memory_usage()
    st.write(f"🔄 시작 메모리: {memory_start:.2f} GB")
    
    # 실제 존재하는 Predictor 클래스들로 테스트
    predictors_to_test = [
        ("surya.input.load", "load_from_folder", "폴더 로딩 함수"),
        ("surya.input.load", "load_from_file", "파일 로딩 함수"),
        ("surya.detection", "DetectionPredictor", "텍스트 감지 Predictor"),
        ("surya.recognition", "RecognitionPredictor", "텍스트 인식 Predictor"),
        ("surya.layout", "LayoutPredictor", "레이아웃 감지 Predictor"),
        ("surya.detection", "DetectionModelLoader", "Detection 모델 로더"),
        ("surya.recognition", "RecognitionModelLoader", "Recognition 모델 로더"),
        ("surya.layout", "LayoutModelLoader", "Layout 모델 로더"),
    ]
    
    success_predictors = []
    for module_name, class_name, description in predictors_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            predictor_class = getattr(module, class_name)
            st.success(f"✅ {description}: {module_name}.{class_name}")
            success_predictors.append(f"{module_name}.{class_name}")
        except ImportError as e:
            st.error(f"❌ {description}: 모듈 {module_name} 임포트 실패 - {e}")
        except AttributeError as e:
            st.error(f"❌ {description}: 클래스 {class_name} 없음 - {e}")
        except Exception as e:
            st.warning(f"⚠️ {description}: 기타 오류 - {e}")
    
    st.write(f"📊 성공한 클래스들:")
    for predictor in success_predictors:
        st.write(f"  - {predictor}")
        
    memory_end = get_memory_usage()
    st.write(f"📊 현재 메모리: {memory_end:.2f} GB (+{memory_end-memory_start:.2f} GB)")

# 5단계: 모델 로딩 함수 찾기
st.subheader("5️⃣ 모델 로딩 함수 찾기")
if st.button("모델 함수 탐색", key="model_functions"):
    memory_start = get_memory_usage()
    st.write(f"🔄 시작 메모리: {memory_start:.2f} GB")
    
    # 각 모듈에서 사용 가능한 함수들 탐색
    modules_to_explore = [
        ("surya.detection", "Detection 모듈"),
        ("surya.recognition", "Recognition 모듈"),
        ("surya.layout", "Layout 모듈"),
        ("surya.table_rec", "Table Recognition 모듈"),
    ]
    
    for module_name, description in modules_to_explore:
        try:
            module = __import__(module_name)
            submodule = getattr(module, module_name.split('.')[1])
            
            st.write(f"\n🔍 **{description}** 사용 가능한 함수들:")
            functions = [name for name in dir(submodule) if not name.startswith('_')]
            
            for func_name in functions:
                try:
                    func = getattr(submodule, func_name)
                    if callable(func):
                        st.write(f"  - {func_name}()")
                except:
                    pass
                    
        except Exception as e:
            st.warning(f"⚠️ {description} 탐색 실패: {e}")
    
    memory_end = get_memory_usage()
    st.write(f"📊 현재 메모리: {memory_end:.2f} GB (+{memory_end-memory_start:.2f} GB)")

# 6단계: 실제 Predictor 객체 생성 테스트 (위험)
st.subheader("6️⃣ 실제 Predictor 객체 생성 테스트 ⚠️")
st.warning("⚠️ **위험한 테스트**: 메모리 오버플로우로 앱이 크래시될 수 있습니다!")

danger_check = st.checkbox("위험을 감수하고 Predictor 객체 생성 테스트 진행")

if danger_check and st.button("🚨 Predictor 객체 생성 테스트", key="predictor_creation"):
    memory_start = get_memory_usage()
    st.write(f"🔄 시작 메모리: {memory_start:.2f} GB")
    
    if memory_start > 0.6:
        st.error("❌ 메모리 사용량이 너무 높습니다. 테스트를 중단합니다.")
        st.stop()
    
    try:
        with st.spinner("Predictor 객체들 생성 중... (30초 이상 소요될 수 있습니다)"):
            
            # Predictor 객체 생성 시도
            test_attempts = [
                ("Detection Predictor", "surya.detection", "DetectionPredictor"),
                ("Recognition Predictor", "surya.recognition", "RecognitionPredictor"),
                ("Layout Predictor", "surya.layout", "LayoutPredictor"),
            ]
            
            created_predictors = []
            for predictor_name, module_name, class_name in test_attempts:
                try:
                    # 모듈 임포트
                    module = __import__(module_name, fromlist=[class_name])
                    predictor_class = getattr(module, class_name)
                    
                    # 객체 생성 시도
                    predictor = predictor_class()
                    st.success(f"✅ {predictor_name} 객체 생성 성공")
                    created_predictors.append(predictor_name)
                    
                    memory_current = get_memory_usage()
                    st.write(f"📊 {predictor_name} 후 메모리: {memory_current:.2f} GB")
                    
                    if memory_current > 0.75:
                        st.warning(f"⚠️ 메모리 한계 근접. 추가 Predictor 생성을 중단합니다.")
                        break
                        
                except Exception as e:
                    st.error(f"❌ {predictor_name} 생성 실패: {e}")
                    # 세부 에러 정보 표시
                    if "CUDA" in str(e) or "GPU" in str(e):
                        st.info("💡 GPU 관련 에러일 수 있습니다. CPU 모드로 시도해보세요.")
                    elif "memory" in str(e).lower():
                        st.warning("⚠️ 메모리 부족 에러입니다.")
            
            memory_final = get_memory_usage()
            st.write(f"📊 최종 메모리: {memory_final:.2f} GB")
            
            if created_predictors:
                st.success(f"🎉 **성공한 Predictors**: {', '.join(created_predictors)}")
                st.success("✅ Surya OCR 0.9.0이 스트림릿 클라우드에서 완전히 작동합니다!")
                st.balloons()
            else:
                st.error("❌ 모든 Predictor 객체 생성에 실패했습니다.")
                    
    except Exception as e:
        st.error(f"❌ Predictor 생성 테스트 실패: {str(e)}")
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
    st.write("- **4단계 성공**: Predictor 클래스들 임포트 가능")
    st.write("- **5단계 성공**: 모델 함수들 탐색 완료")
    st.write("- **6단계 성공**: 실제 OCR Predictor 객체 생성 가능")
    
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
3. '개별 모듈 테스트'로 실제 존재하는 모듈들 확인
4. 'Predictor 클래스 테스트'로 0.9.0 버전의 실제 클래스들 확인
5. '모델 함수 탐색'으로 사용 가능한 모든 함수 목록 확인
6. 메모리가 충분하면 'Predictor 객체 생성 테스트' 실행 (선택)

⚠️ **주의사항**: 
- 각 단계에서 메모리 사용량을 꼭 확인하세요
- 0.7GB 초과 시 다음 단계를 진행하지 마세요
- 앱이 크래시되면 스트림릿에서 재부팅하세요

🎯 **0.9.0 버전 사용법**:
```python
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.layout import LayoutPredictor

# Predictor 객체 생성
detector = DetectionPredictor()
recognizer = RecognitionPredictor()
layout_analyzer = LayoutPredictor()
```
""")
