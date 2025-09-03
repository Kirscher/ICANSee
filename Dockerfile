FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN apt-get update && apt-get install -y git
RUN pip install yq

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY train.py .
COPY evaluate.py .
COPY metrics.py .
COPY utils.py .
COPY dataset.py .
COPY run_baselines.sh .
COPY pretrained . /app/pretrained/

COPY configs/ /app/configs/


# Run training for each baseline
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["./run_baselines.sh"]


