import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.losses import Loss
import numpy as np

def make_lstm(input_shape, lr=1e-3):
    x_in = L.Input(shape=input_shape)
    x = L.LSTM(64, return_sequences=True)(x_in)
    x = L.Dropout(0.2)(x)
    x = L.LSTM(32)(x)
    x = L.Dropout(0.2)(x)
    x = L.Dense(32, activation="relu")(x)
    y = L.Dense(1)(x)
    model = tf.keras.Model(x_in, y, name="lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model

def make_gru(input_shape, lr=1e-3):
    x_in = L.Input(shape=input_shape)
    x = L.GRU(64, return_sequences=True)(x_in)
    x = L.Dropout(0.2)(x)
    x = L.GRU(32)(x)
    x = L.Dropout(0.2)(x)
    x = L.Dense(32, activation="relu")(x)
    y = L.Dense(1)(x)
    model = tf.keras.Model(x_in, y, name="gru")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model

def make_cnn(input_shape, lr=1e-3):
    x_in = L.Input(shape=input_shape)
    x = L.Conv1D(64, 3, padding="causal", activation="relu")(x_in)
    x = L.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.2)(x)
    x = L.Dense(64, activation="relu")(x)
    y = L.Dense(1)(x)
    model = tf.keras.Model(x_in, y, name="cnn1d")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model

def transformer_block(x, num_heads=4, key_dim=32, mlp_dim=64, rate=0.1):
    attn = L.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = L.LayerNormalization()(x + attn)
    mlp = L.Dense(mlp_dim, activation="relu")(x)
    mlp = L.Dropout(rate)(mlp)
    mlp = L.Dense(x.shape[-1])(mlp)
    x = L.LayerNormalization()(x + mlp)
    return x

def make_trans(input_shape, lr=1e-3, depth=2):
    x_in = L.Input(shape=input_shape)
    x = x_in
    for _ in range(depth):
        x = transformer_block(x, num_heads=4, key_dim=32, mlp_dim=64, rate=0.1)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.2)(x)
    x = L.Dense(64, activation="relu")(x)
    y = L.Dense(1)(x)
    model = tf.keras.Model(x_in, y, name="transformer")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model

# Custom Loss Functions
def pinball_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss

def huber_multi_loss(delta=1.0):
    base = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)
    def loss(y_true, y_pred):
        # y_* shape: [batch, H]
        l = base(y_true, y_pred)
        return tf.reduce_mean(l)
    return loss

def bce_multi_loss():
    base = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    def loss(y_true, y_pred):
        l = base(y_true, y_pred)
        return tf.reduce_mean(l)
    return loss

def combined_multihorizon_loss(loss_type="huber", delta=1.0, lam=0.3):
    reg_loss_fn = huber_multi_loss(delta=delta) if loss_type == "huber" else None
    if loss_type == "pinball":
        # Use median quantile as default
        reg_loss_fn = pinball_loss(0.5)
    cls_loss_fn = bce_multi_loss()

    def loss(y_true, y_pred):
        # y_true: {"reg": [B,H], "cls": [B,H]}
        # y_pred: {"reg": [B,H], "cls": [B,H]}
        reg_l = reg_loss_fn(y_true["reg"], y_pred["reg"]) if reg_loss_fn is not None else 0.0
        cls_l = cls_loss_fn(y_true["cls"], y_pred["cls"]) if y_true.get("cls") is not None else 0.0
        return reg_l + lam * cls_l

    return loss

# Squeeze-and-Excitation Block
def squeeze_excitation_block(x, ratio=16):
    """Lightweight channel attention mechanism"""
    channels = x.shape[-1]
    se = L.GlobalAveragePooling1D()(x)
    se = L.Dense(channels // ratio, activation='relu')(se)
    se = L.Dense(channels, activation='sigmoid')(se)
    se = L.Reshape((1, channels))(se)
    return L.Multiply()([x, se])

# TCN Residual Block
def tcn_residual_block(x, filters=64, kernel_size=5, dilation_rate=1, dropout_rate=0.1):
    """TCN residual block with causal convolution, layer norm, ReLU, and dropout"""
    # Main path
    conv = L.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        use_bias=False
    )(x)
    conv = L.LayerNormalization()(conv)
    conv = L.ReLU()(conv)
    conv = L.Dropout(dropout_rate)(conv)
    
    # Residual connection
    if x.shape[-1] != filters:
        # Match dimensions with 1x1 conv
        x = L.Conv1D(filters, 1, padding='same')(x)
    
    return L.Add()([x, conv])

# TCN-Residual Model
def make_tcn_residual(input_shape, filters=64, kernel_size=5, dilations=[1, 2, 4, 8], 
                     dropout_rate=0.1, use_se=True, use_multitask=True, lr=1e-3,
                     horizons=None, multihorizon=False, loss_type="huber", lambda_cls=0.3):
    """
    TCN-Residual architecture as specified:
    Input L×d
     └─> Causal 1D Conv (k=5, f=64, dilation=1) + LayerNorm + ReLU + Dropout(0.1)
         Residual add
     └─> Causal 1D Conv (k=5, f=64, dilation=2) + LN + ReLU + Dropout(0.1) + Residual
     └─> Causal 1D Conv (k=5, f=64, dilation=4) + LN + ReLU + Dropout(0.1) + Residual
     └─> Causal 1D Conv (k=5, f=64, dilation=8) + LN + ReLU + Dropout(0.1) + Residual
     └─> (tuỳ) Squeeze-Excitation (channel attention nhẹ)
     └─> GlobalAvgPool (theo thời gian)
     └─> Head Hồi quy: Dense(64) → Dense(1)  =>  ŷ_reg = r̂_{t+1}
     └─> (tuỳ) Head Phân loại: Dense(64) → Dense(1, sigmoid) =>  p̂(up)
    """
    x_in = L.Input(shape=input_shape)
    x = x_in
    
    # TCN residual blocks with increasing dilations
    for dilation in dilations:
        x = tcn_residual_block(
            x, filters=filters, kernel_size=kernel_size, 
            dilation_rate=dilation, dropout_rate=dropout_rate
        )
    
    # Optional Squeeze-Excitation
    if use_se:
        x = squeeze_excitation_block(x)
    
    # Global Average Pooling
    x = L.GlobalAveragePooling1D()(x)
    
    if use_multitask and multihorizon:
        H = len(horizons) if horizons is not None else 3
        # Regression head with MC Dropout
        reg_head = L.Dense(64, activation='relu', name='reg_dense')(x)
        reg_head = L.Dropout(0.1, name='reg_dropout')(reg_head)
        reg_out = L.Dense(H, name='reg')(reg_head)

        # Classification head with MC Dropout
        cls_head = L.Dense(64, activation='relu', name='cls_dense')(x)
        cls_head = L.Dropout(0.1, name='cls_dropout')(cls_head)
        cls_out = L.Dense(H, activation='sigmoid', name='cls')(cls_head)

        outputs = {"reg": reg_out, "cls": cls_out}

        model = tf.keras.Model(x_in, outputs, name="tcn_residual_multihorizon")

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=float(1e-4)),
            loss=combined_multihorizon_loss(loss_type=loss_type, delta=1.0, lam=lambda_cls),
            metrics={'reg': ['mae']}
        )
    elif use_multitask:
        # Original multi-task single-horizon
        reg_head = L.Dense(64, activation='relu', name='reg_dense')(x)
        reg_out = L.Dense(1, name='regression_output')(reg_head)

        cls_head = L.Dense(64, activation='relu', name='cls_dense')(x)
        cls_out = L.Dense(1, activation='sigmoid', name='classification_output')(cls_head)

        outputs = L.Concatenate(name='combined_output')([reg_out, cls_out])

        model = tf.keras.Model(x_in, outputs, name="tcn_residual_multitask")

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=float(1e-4)),
            loss=huber_multi_loss(delta=1.0),
            metrics=['mae']
        )
    else:
        # Single regression head
        x = L.Dense(64, activation='relu')(x)
        y = L.Dense(1)(x)
        
        model = tf.keras.Model(x_in, y, name="tcn_residual")
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=float(1e-4)),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae']
        )
    
    return model

# Learning Rate Scheduler with Warmup + Cosine Decay
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_epochs, total_epochs, steps_per_epoch):
        super().__init__()
        self.initial_learning_rate = tf.constant(initial_learning_rate, dtype=tf.float32)
        self.warmup_steps = tf.constant(warmup_epochs * steps_per_epoch, dtype=tf.float32)
        self.total_steps = tf.constant(total_epochs * steps_per_epoch, dtype=tf.float32)
        self.cosine_decay = CosineDecay(
            initial_learning_rate, 
            total_epochs * steps_per_epoch - warmup_epochs * steps_per_epoch
        )
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
    
    @tf.function
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Sử dụng tf.cond để xử lý symbolic tensor
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.initial_learning_rate * step / self.warmup_steps,
            lambda: self.cosine_decay(step - self.warmup_steps)
        )
    
    def get_config(self):
        return {
            'initial_learning_rate': float(self.initial_learning_rate),
            'warmup_epochs': self.warmup_epochs,
            'steps_per_epoch': self.steps_per_epoch
        }
def make_model(name: str, input_shape, lr=1e-3, **kwargs):
    name = name.lower()
    if name == "lstm":
        return make_lstm(input_shape, lr)
    if name == "gru":
        return make_gru(input_shape, lr)
    if name in ["cnn", "cnn1d"]:
        return make_cnn(input_shape, lr)
    if name in ["trans", "transformer"]:
        return make_trans(input_shape, lr)
    if name in ["tcn", "tcn_residual"]:
        return make_tcn_residual(input_shape, lr=lr, **kwargs)
    raise ValueError(f"Unknown model: {name}")