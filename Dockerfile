FROM tensorflow/tensorflow:latest-jupyter AS tf
COPY . /app
RUN make -C /app
CMD python /app/src/main.py
