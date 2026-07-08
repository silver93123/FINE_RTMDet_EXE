"""
Cython 빌드 스크립트
.py 소스를 .so 바이너리로 컴파일합니다.

실행 (최초 1회):
    conda activate vision_env
    pip install cython
    python build_cython.py build_ext --inplace

완료 후 생성 파일 예시:
    scripts/Run_binpicking_TCP_UI.cpython-310-x86_64-linux-gnu.so
    src/camera/__init__.cpython-310-x86_64-linux-gnu.so
    src/camera/base.cpython-310-x86_64-linux-gnu.so
    src/camera/lucid_helios.cpython-310-x86_64-linux-gnu.so
    src/detection/__init__.cpython-310-x86_64-linux-gnu.so
    src/detection/rtmdet_inferencer.cpython-310-x86_64-linux-gnu.so
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# =============================================================================
# scripts/__init__.py 자동 생성
# scripts/ 를 Python 패키지로 인식시키기 위해 필요
# (없으면 'from scripts.Run_binpicking_TCP_UI import main' 실행 시 오류)
# =============================================================================
scripts_init = "scripts/__init__.py"
if not os.path.exists(scripts_init):
    open(scripts_init, "w").close()
    print(f"[CREATE] {scripts_init}")

# =============================================================================
# 컴파일 대상 파일 목록
# =============================================================================
TARGETS = [
    "scripts/__init__.py",
    "scripts/Run_binpicking_TCP_UI.py",
    "src/camera/__init__.py",
    "src/camera/base.py",
    "src/camera/lucid_helios.py",
    "src/detection/__init__.py",
    "src/detection/rtmdet_inferencer.py",
]

# =============================================================================
# Extension 생성
# =============================================================================
extensions = []
for target in TARGETS:
    if not os.path.exists(target):
        print(f"[SKIP] 파일 없음: {target}")
        continue

    # 경로 → 모듈명 변환
    # scripts/__init__.py               → scripts
    # scripts/Run_binpicking_TCP_UI.py  → scripts.Run_binpicking_TCP_UI
    # src/camera/__init__.py            → src.camera
    # src/camera/base.py                → src.camera.base
    mod_name = target.replace("/", ".").replace(".py", "")
    if mod_name.endswith(".__init__"):
        mod_name = mod_name[:-9]

    ext = Extension(
        name=mod_name,
        sources=[target],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2"],
    )
    extensions.append(ext)
    print(f"[ADD] {target}  →  {mod_name}")

# =============================================================================
# 빌드
# =============================================================================
setup(
    name="binpicking",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
        build_dir="build_cython",   # 중간 .c 파일 (빌드 후 삭제 가능)
    ),
)
