#!/usr/bin/env bash
set -euo pipefail

# Run ZenFlow B with CUDA 12 and a user-owned GCC/G++ 12 Conda toolchain.
# Activate the Conda environment before invoking this script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_server_conda.sh --check-only
  bash scripts/run_server_conda.sh --install-deps --clean-jittor-cache --check-only
  bash scripts/run_server_conda.sh [--clean-jittor-cache] [run.py arguments]

Examples:
  bash scripts/run_server_conda.sh --check-only
  bash scripts/run_server_conda.sh --install-deps --clean-jittor-cache --check-only
  bash scripts/run_server_conda.sh
  bash scripts/run_server_conda.sh --chunk-rows 2500
  bash scripts/run_server_conda.sh --clean-jittor-cache

Environment overrides:
  nvcc_path=/usr/bin/nvcc
  CUDA_HOME=/path/to/cuda
  PYTHON_BIN=/path/to/conda/env/bin/python
EOF
}

CHECK_ONLY=0
CLEAN_CACHE=0
INSTALL_DEPS=0
PIPELINE_ARGS=()
while (($#)); do
    case "$1" in
        --check-only)
            CHECK_ONLY=1
            ;;
        --clean-jittor-cache)
            CLEAN_CACHE=1
            ;;
        --install-deps)
            INSTALL_DEPS=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PIPELINE_ARGS+=("$1")
            ;;
    esac
    shift
done

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: no active Conda environment." >&2
    echo "Run: conda activate zenflow" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
CC12="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
CXX12="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

for executable in "${PYTHON_BIN}" "${CC12}" "${CXX12}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "ERROR: required executable not found: ${executable}" >&2
        echo "Install the toolchain with:" >&2
        echo "  conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12 -y" >&2
        exit 2
    fi
done

PYTHON_IMPLEMENTATION="$("${PYTHON_BIN}" -c 'import sys; print(sys.implementation.name)')"
PYTHON_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
HAS_RTLD_DEEPBIND="$("${PYTHON_BIN}" -c 'import os; print(int(hasattr(os, "RTLD_DEEPBIND")))')"
if [[ "${PYTHON_IMPLEMENTATION}" != "cpython" || "${HAS_RTLD_DEEPBIND}" != "1" ]]; then
    echo "ERROR: Jittor requires Linux CPython; the active interpreter is ${PYTHON_IMPLEMENTATION}." >&2
    echo "  executable=${PYTHON_BIN}" >&2
    echo "  prefix=${PYTHON_PREFIX}" >&2
    echo "Create a clean CPython environment as documented in README.md." >&2
    exit 2
fi
if [[ "$(readlink -f "${PYTHON_PREFIX}")" != "$(readlink -f "${CONDA_PREFIX}")" ]]; then
    echo "ERROR: Python does not belong to the active Conda environment." >&2
    echo "  CONDA_PREFIX=${CONDA_PREFIX}" >&2
    echo "  sys.prefix=${PYTHON_PREFIX}" >&2
    exit 2
fi

GCC_MAJOR="$(${CC12} -dumpfullversion -dumpversion | cut -d. -f1)"
GXX_MAJOR="$(${CXX12} -dumpfullversion -dumpversion | cut -d. -f1)"
if [[ "${GCC_MAJOR}" != "12" || "${GXX_MAJOR}" != "12" ]]; then
    echo "ERROR: the active Conda compilers must both be version 12." >&2
    "${CC12}" --version >&2 || true
    "${CXX12}" --version >&2 || true
    exit 2
fi

NVCC_BIN=""
if [[ -n "${nvcc_path:-}" && -x "${nvcc_path}" ]]; then
    NVCC_BIN="${nvcc_path}"
elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    NVCC_BIN="${CUDA_HOME}/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
else
    for candidate in /usr/local/cuda-12.0 /usr/local/cuda; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            NVCC_BIN="${candidate}/bin/nvcc"
            break
        fi
    done
fi
if [[ -z "${NVCC_BIN}" || ! -x "${NVCC_BIN}" ]]; then
    echo "ERROR: CUDA nvcc was not found." >&2
    echo "Set its path, for example:" >&2
    echo "  nvcc_path=/usr/bin/nvcc bash scripts/run_server_conda.sh --check-only" >&2
    exit 2
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    NVCC_REAL="$(readlink -f "${NVCC_BIN}")"
    CUDA_HOME="$(cd -- "$(dirname -- "${NVCC_REAL}")/.." && pwd)"
fi

export CUDA_HOME
export CC="${CC12}"
export CXX="${CXX12}"
export gcc_path="${CC12}"
export cc_path="${CXX12}"
export CUDAHOSTCXX="${CXX12}"
export CUDACXX="${NVCC_BIN}"
export nvcc_path="${NVCC_BIN}"
export PATH="${CONDA_PREFIX}/bin:$(dirname -- "${NVCC_BIN}"):${PATH}"
# Jittor's downloaded oneDNN is linked against GNU OpenMP.  With a Conda
# compiler, libgomp lives in the environment rather than a system library dir.
export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -d "${CUDA_HOME}/lib64" ]]; then
    export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi
export PYTHONPATH="${PROJECT_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

NVCC_RELEASE="$(${nvcc_path} --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -n 1)"
if [[ "${NVCC_RELEASE}" != 12.* ]]; then
    echo "ERROR: expected CUDA 12.x nvcc, found release ${NVCC_RELEASE:-unknown}." >&2
    exit 2
fi

echo "Environment ready:"
echo "  CONDA_PREFIX=${CONDA_PREFIX}"
echo "  python=$(${PYTHON_BIN} --version 2>&1) (${PYTHON_IMPLEMENTATION})"
echo "  CC=${CC} ($(${CC} -dumpfullversion -dumpversion))"
echo "  CXX=${CXX} ($(${CXX} -dumpfullversion -dumpversion))"
echo "  nvcc_path=${nvcc_path} (CUDA ${NVCC_RELEASE})"
echo "  cc_path=${cc_path}"

LIBGOMP_PATH="$(find "${CONDA_PREFIX}" -name 'libgomp.so.1' -print -quit 2>/dev/null || true)"
if [[ -z "${LIBGOMP_PATH}" ]]; then
    echo "ERROR: libgomp.so.1 is missing from the Conda environment." >&2
    echo "Install it with:" >&2
    echo "  conda install -c conda-forge 'libgomp>=12,<13' -y" >&2
    exit 2
fi
echo "  libgomp=${LIBGOMP_PATH}"

if ((CLEAN_CACHE)); then
    CACHE_ROOT="${HOME}/.cache/jittor"
    if [[ -e "${CACHE_ROOT}" ]]; then
        CACHE_BACKUP="${HOME}/.cache/jittor.compiler-backup.$(date +%Y%m%d-%H%M%S)"
        mv -- "${CACHE_ROOT}" "${CACHE_BACKUP}"
        echo "Moved old Jittor cache to ${CACHE_BACKUP}"
    else
        echo "No existing Jittor cache to move."
    fi
fi

if ((INSTALL_DEPS)); then
    echo "Installing Python dependencies inside ${CONDA_PREFIX}..."
    "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
    "${PYTHON_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
    # The server may provide CUDA through /usr/bin and /usr/include without
    # installing the cuDNN development package.  Install only cuDNN here;
    # --no-deps avoids downloading another CUDA/cuBLAS stack over the system
    # CUDA 12 toolkit.
    "${PYTHON_BIN}" -m pip install --no-deps \
        'nvidia-cudnn-cu12==8.9.7.29'
fi

# Debian/Ubuntu CUDA packages commonly place nvcc in /usr/bin, headers in
# /usr/include and libraries in /usr/lib/x86_64-linux-gnu.  Jittor can use that
# layout, but it does not search a pip-installed cuDNN package.  If system
# cuDNN is absent, create a user-owned CUDA view whose target directory exposes
# the wheel's cuDNN headers/libraries while retaining the system CUDA toolkit.
CUDNN_HEADER=""
CUDNN_LIB_DIR=""
for candidate in \
    "${CUDA_HOME}/include/cudnn.h" \
    "/usr/include/cudnn.h" \
    "${CONDA_PREFIX}/include/cudnn.h"; do
    if [[ -f "${candidate}" ]]; then
        CUDNN_HEADER="${candidate}"
        break
    fi
done

if [[ -z "${CUDNN_HEADER}" ]]; then
    PYTHON_SITE="$(${PYTHON_BIN} -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    CUDNN_WHEEL_ROOT="${PYTHON_SITE}/nvidia/cudnn"
    if [[ -f "${CUDNN_WHEEL_ROOT}/include/cudnn.h" ]]; then
        CUDNN_HEADER="${CUDNN_WHEEL_ROOT}/include/cudnn.h"
        CUDNN_LIB_DIR="${CUDNN_WHEEL_ROOT}/lib"

        CUDA_SYSTEM_INCLUDE="/usr/include"
        CUDA_SYSTEM_LIB="/usr/lib/x86_64-linux-gnu"
        if [[ ! -f "${CUDA_SYSTEM_INCLUDE}/cuda.h" ]]; then
            echo "ERROR: system CUDA header not found: ${CUDA_SYSTEM_INCLUDE}/cuda.h" >&2
            exit 2
        fi
        if [[ ! -f "${CUDA_SYSTEM_LIB}/libcudart.so" ]]; then
            echo "ERROR: system CUDA runtime not found: ${CUDA_SYSTEM_LIB}/libcudart.so" >&2
            exit 2
        fi

        CUDA_VIEW="${CONDA_PREFIX}/jittor-cuda-system-v2"
        CUDA_VIEW_TARGET="${CUDA_VIEW}/targets/x86_64-linux"
        mkdir -p "${CUDA_VIEW}/bin" "${CUDA_VIEW}/include" \
            "${CUDA_VIEW_TARGET}/include" "${CUDA_VIEW_TARGET}/lib"
        ln -sfn "${CUDA_SYSTEM_LIB}" "${CUDA_VIEW}/lib64"

        # Jittor discovers cudnn.h in targets/.../include, but its generated
        # compile command only carries CUDA_HOME/include.  Build a merged
        # include view containing both system CUDA and wheel cuDNN headers.
        while IFS= read -r -d '' system_header; do
            ln -sfn "${system_header}" \
                "${CUDA_VIEW}/include/$(basename -- "${system_header}")"
        done < <(find "${CUDA_SYSTEM_INCLUDE}" -mindepth 1 -maxdepth 1 \
            -print0)

        # A wrapper must be a regular executable rather than a symlink because
        # Jittor resolves nvcc symlinks before deriving the CUDA root.
        printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "${NVCC_BIN}" \
            > "${CUDA_VIEW}/bin/nvcc"
        chmod 755 "${CUDA_VIEW}/bin/nvcc"

        while IFS= read -r header; do
            ln -sfn "${header}" "${CUDA_VIEW}/include/$(basename -- "${header}")"
            ln -sfn "${header}" "${CUDA_VIEW_TARGET}/include/$(basename -- "${header}")"
        done < <(find "${CUDNN_WHEEL_ROOT}/include" -maxdepth 1 \
            -type f -name 'cudnn*.h' -print)

        while IFS= read -r library; do
            library_name="$(basename -- "${library}")"
            ln -sfn "${library}" "${CUDA_VIEW_TARGET}/lib/${library_name}"
            unversioned_name="${library_name%%.so.*}.so"
            ln -sfn "${library}" "${CUDA_VIEW_TARGET}/lib/${unversioned_name}"
        done < <(find "${CUDNN_LIB_DIR}" -maxdepth 1 \
            -type f -name 'libcudnn*.so*' -print)

        CUDA_HOME="${CUDA_VIEW}"
        NVCC_BIN="${CUDA_VIEW}/bin/nvcc"
        export CUDA_HOME
        export nvcc_path="${NVCC_BIN}"
        export CUDACXX="${NVCC_BIN}"
        export PATH="${CUDA_VIEW}/bin:${PATH}"
        export LIBRARY_PATH="${CUDA_VIEW_TARGET}/lib:${LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${CUDA_VIEW_TARGET}/lib:${LD_LIBRARY_PATH}"
        echo "  CUDA view=${CUDA_VIEW}"
        echo "  cuDNN=${CUDNN_HEADER}"
    else
        echo "ERROR: CUDA 12 was found, but the cuDNN development files are missing." >&2
        echo "Install the supported user-space package with:" >&2
        echo "  ${PYTHON_BIN} -m pip install --no-deps nvidia-cudnn-cu12==8.9.7.29" >&2
        echo "Then rerun this script with --clean-jittor-cache --check-only." >&2
        exit 2
    fi
else
    echo "  cuDNN=${CUDNN_HEADER}"
fi

echo "Checking Jittor CUDA compilation (the first run can take several minutes)..."
"${PYTHON_BIN}" - <<'PY'
import os

print("Jittor cc_path:", os.environ["cc_path"])
print("Jittor nvcc_path:", os.environ["nvcc_path"])

import jittor as jt

expected_cc = os.path.realpath(os.environ["cc_path"])
expected_nvcc = os.path.realpath(os.environ["nvcc_path"])
actual_cc = os.path.realpath(jt.flags.cc_path)
actual_nvcc = os.path.realpath(jt.flags.nvcc_path)
assert actual_cc == expected_cc, (actual_cc, expected_cc)
assert actual_nvcc == expected_nvcc, (actual_nvcc, expected_nvcc)
jt.flags.use_cuda = 1
x = jt.ones((64, 64))
y = (x @ x).mean()
value = float(y.item())
assert abs(value - 64.0) < 1e-4, value
assert jt.has_cuda, "Jittor did not enable CUDA"
print("Jittor version:", jt.__version__)
print("Jittor has_cuda:", jt.has_cuda)
print("Jittor actual cc_path:", jt.flags.cc_path)
print("Jittor actual nvcc_path:", jt.flags.nvcc_path)
print("CUDA smoke result:", value)

from jittor_geometric.nn.models.craft import CRAFT
print("JittorGeometric CRAFT import: OK")
PY

if ((CHECK_ONLY)); then
    echo "Environment and Jittor CUDA checks passed; training was not started."
    exit 0
fi

cd "${PROJECT_ROOT}"
echo "Starting the complete Dataset3 + Dataset4 B-board pipeline..."
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/run.py" \
    --target b --workers 8 --chunk-rows 5000 "${PIPELINE_ARGS[@]}"
