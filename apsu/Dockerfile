# Base image with Python and PyTorch support, including CUDA for GPU acceleration
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Create a directory for the software and copy the application code
COPY ./software/ /app/software/
COPY ./apsu/ /app/apsu/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies using pip
# Pinning versions for reproducibility as per REQUIREMENTS_v4.md
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0 \
    reservoirpy==0.4.16 \
    cma==3.3.0 \
    matplotlib==3.8.4 \
    pandas==2.2.2 \
    jupyterlab==4.2.1

# Expose ports if necessary (e.g., for Jupyter)
EXPOSE 8888

# Default command to run when the container starts
# For example, starting a bash shell or running the main experiment script
CMD ["/bin/bash"] 