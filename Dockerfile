# Use the official Jupyter base image
FROM quay.io/jupyter/base-notebook

# Set working directory
WORKDIR /home/jovyan/work

# Copy the entire repository
COPY . .

# Install Python dependencies and the local package in editable mode
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Copy the notebook file
COPY tattoo.ipynb .

# Expose port for Jupyter
EXPOSE 8888

# Set environment variables
ENV JUPYTER_ENABLE_LAB=yes

# Start Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 