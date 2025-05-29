FROM python:3.11-slim

# Install system dependencies including BLAS libraries and OpenMP for snnpy
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    gcc \
    g++ \
    make \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libgomp1 \
    libomp-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Needed for SNN to work.
RUN pip install --upgrade pip wheel setuptools

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY clemont/ ./clemont/
COPY example.py .

# Install main Clemont dependencies
RUN pip install numpy pandas scikit-learn faiss-cpu psutil

# Install SNN dependencies
RUN pip install pybind11

# Install SNN.
RUN pip install -v snnpy

# WORKAROUND: Fix snnpy packaging bug
# The snnpy package (in v0.0.9) seems to have a setup.py bug where the compiled 
# C++ extension 'snnomp.cpython-311-aarch64-linux-gnu.so' gets installed to the root 
# of site-packages instead of inside the snnpy/ package directory. This causes
# "ModuleNotFoundError: No module named 'snnpy.snnomp'" on import
#
# The workaround finds the misplaced .so file and moves it to the correct location
# inside the snnpy package directory where Python can find it.
RUN python -c "import os, glob, shutil; so_files = glob.glob('/usr/local/lib/python3.11/site-packages/snnomp*.so'); [shutil.move(so_files[0], '/usr/local/lib/python3.11/site-packages/snnpy/' + os.path.basename(so_files[0])) if so_files else None]; print('Moved' if so_files else 'No files to move')"

# Verify snnpy installation
RUN python -c "import snnpy; print('snnpy basic import works')"

# Install dd using approach from the official script
RUN pip install dd
RUN pip uninstall -y dd
RUN pip download --no-deps dd --no-binary dd
RUN tar -xzf dd-*.tar.gz && \
    DD_DIR=$(ls -d dd-*/ | head -1) && \
    cd "$DD_DIR" && \
    DD_FETCH=1 DD_CUDD=1 DD_CUDD_ZDD=1 pip install . -vvv --use-pep517 --no-build-isolation

# Verify dd
RUN cd /app && python -c "import dd.cudd; print('dd.cudd imported successfully')"

# Install clemont
RUN pip install -e .

# Verify that available backends work
RUN python -c "import clemont; print('Available backends:', clemont.list_available_backends())"
RUN python -c "from clemont.backends.bdd import BDD; print('BDD backend ready!')"
RUN python -c "from clemont.backends.snn import Snn; print('SNN backend ready!')"

# Set up workspace directory for mounting user code
WORKDIR /workspace

# Default command
CMD ["bash"]
