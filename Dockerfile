# Use the official Jupyter base image
FROM quay.io/jupyter/base-notebook

# Set working directory
WORKDIR /home/jovyan/work

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the notebook file
COPY tattoo.ipynb .

# Expose port for Jupyter
EXPOSE 8888

# Set environment variables
ENV JUPYTER_ENABLE_LAB=yes

# Start Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 