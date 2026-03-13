import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib

# 使用无界面后端，避免在没有显示器的环境中报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf


# 固定随机种子，保证实验结果可复现
np.random.seed(42)
tf.random.set_seed(42)
tf.get_logger().setLevel("ERROR")


class EpochLogger(tf.keras.callbacks.Callback):
    """每 10 轮输出一次训练过程指标。"""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_epoch = epoch + 1

        if current_epoch % 10 == 0:
            train_mse = logs.get("mean_squared_error", 0.0)
            val_mse = logs.get("val_mean_squared_error", 0.0)
            print(
                f"第 {current_epoch:03d} 轮 - "
                f"训练集 MSE: {train_mse:.6f}, "
                f"验证集 MSE: {val_mse:.6f}"
            )


def target_function(x: np.ndarray) -> np.ndarray:
    """目标函数：y = sin(x) + 2cos(2x)。"""
    return np.sin(x) + 2.0 * np.cos(2.0 * x)


def resolve_interval() -> tuple[float, float]:
    """
    如果后续需要改为其他区间，只需修改下面两个值即可。
    """
    requested_left = -2.0 * np.pi
    requested_right = 2.0 * np.pi

    return requested_left, requested_right


def generate_dataset(num_samples: int, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    """在指定区间内均匀随机采样，并生成对应标签。"""
    x = np.random.uniform(x_min, x_max, size=(num_samples, 1)).astype(np.float32)
    y = target_function(x).astype(np.float32)
    return x, y


def build_model() -> tf.keras.Sequential:
    """构建包含至少两个隐藏层的 ReLU 神经网络。"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )
    return model


def main() -> None:
    # 解析实验区间
    x_min, x_max = resolve_interval()

    # 按题目要求生成训练集和测试集
    x_train, y_train = generate_dataset(num_samples=2000, x_min=x_min, x_max=x_max)
    x_test, y_test = generate_dataset(num_samples=500, x_min=x_min, x_max=x_max)

    # 构建并训练模型
    model = build_model()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=300,
        batch_size=64,
        verbose=0,
        callbacks=[EpochLogger()],
    )

    # 在测试集上评估最终 MSE
    test_loss, test_mse = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集最终 MSE 损失值: {test_mse:.6f}")

    # 为了便于绘图展示，将测试样本按 x 从小到大排序
    sorted_indices = np.argsort(x_test[:, 0])
    x_test_sorted = x_test[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = model.predict(x_test_sorted, verbose=0)

    # 绘制真实函数与模型预测曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x_test_sorted, y_test_sorted, label="Ground Truth", linewidth=2)
    plt.plot(x_test_sorted, y_pred_sorted, label="Prediction", linewidth=2, linestyle="--")
    plt.title("ReLU Network Fitting for y = sin(x) + 2cos(2x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("hw_fit.png", dpi=200)
    print("对比图已保存为 hw_fit.png")

    # 输出训练过程中的最优验证集 MSE，便于分析拟合效果
    best_val_mse = min(history.history["val_mean_squared_error"])
    print(f"训练过程中最优验证集 MSE: {best_val_mse:.6f}")


if __name__ == "__main__":
    main()
