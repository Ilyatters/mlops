import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
from prometheus_client import start_http_server, Gauge

# === Конфигурация (можно переопределить через env) ===
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
ARTIFACT_DIR = "artifacts"
np.random.seed(42)

# === Prometheus метрики ===
loss_gauge = Gauge('model_loss', 'Current training loss')
weight_gauge = Gauge('model_weight', 'Current model weight')
bias_gauge = Gauge('model_bias', 'Current model bias')
start_http_server(8000)  # будет отдавать метрики на http://localhost:8000

# === Данные ===
X = np.random.rand(100, 1) * 10
y = 3 * X + 5 + np.random.randn(100, 1) * 2

# === Гиперпараметры ===
learning_rate = 0.02
epochs = 50

# === Модель ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[1]),
    tf.keras.layers.Dense(1, use_bias=True)
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss='mse')

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# === Подключаемся к MLflow server ===
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Linear_Regression_Experiment")

# === Callback, который логирует метрики в MLflow и Prometheus по эпохам ===
class MLFlowPromCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.w_history = []
        self.b_history = []
        self.loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        loss = float(logs.get('loss'))
        # читаем веса
        w, b = model.get_weights()
        w_val = float(w[0][0])
        b_val = float(b[0])

        # MLflow
        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("weight", w_val, step=epoch)
        mlflow.log_metric("bias", b_val, step=epoch)

        # Prometheus
        loss_gauge.set(loss)
        weight_gauge.set(w_val)
        bias_gauge.set(b_val)

        # локально сохранить историю для графиков
        self.loss_history.append(loss)
        self.w_history.append(w_val)
        self.b_history.append(b_val)

# === Запуск обучения внутри MLflow run ===
with mlflow.start_run():
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)

    cb = MLFlowPromCallback()
    history = model.fit(X, y, epochs=epochs, verbose=1, callbacks=[cb])

    # Создаём графики из callback-истории
    plt.figure(figsize=(6,4))
    plt.plot(cb.loss_history)
    plt.title('Loss Dynamics')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    loss_path = os.path.join(ARTIFACT_DIR, "loss_plot.png")
    plt.savefig(loss_path)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(cb.w_history)
    plt.title('Weight (w) Dynamics')
    plt.xlabel('Epoch')
    plt.grid(True)
    w_path = os.path.join(ARTIFACT_DIR, "weight_plot.png")
    plt.savefig(w_path)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(cb.b_history)
    plt.title('Bias (b) Dynamics')
    plt.xlabel('Epoch')
    plt.grid(True)
    b_path = os.path.join(ARTIFACT_DIR, "bias_plot.png")
    plt.savefig(b_path)
    plt.close()

    mlflow.log_artifact(loss_path)
    mlflow.log_artifact(w_path)
    mlflow.log_artifact(b_path)

    # Сигнатура + example -> безопасно сохраняем модель
    signature = infer_signature(X, model.predict(X))
    input_example = X[:3].tolist()  # короткий пример (json-serializable)

    # Сохраняем keras-модель в MLflow с сигнатурой
    mlflow.keras.log_model(model, artifact_path="model", signature=signature, input_example=input_example)

    # Логируем финальные веса как params
    final_w, final_b = model.get_weights()
    mlflow.log_param("final_weight", float(final_w[0][0]))
    mlflow.log_param("final_bias", float(final_b[0]))

    print("Training finished. Metrics available on Prometheus (http://localhost:9090) and MLflow UI (http://localhost:5000).")

    # Оставляем сервис отдавать метрики ещё немного, чтобы Prometheus успел их просканить
    time.sleep(5)
