FROM tensorflow/tensorflow:latest-gpu-jupyter
COPY . /app
RUN make /app
CMD python /app/src/main.py
