FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the ANN once at build time so app.py has model.h5 / scaler.pkl / classes.json ready
RUN python train_model.py

EXPOSE 7860

CMD ["python", "app.py"]
