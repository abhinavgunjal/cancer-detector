import numpy as np
import tensorflow as tf

# =========================
# 🔥 GENERATOR
# =========================
def build_generator(noise_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(noise_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='linear')  # match feature size
    ])
    return model


# =========================
# 🔥 DISCRIMINATOR
# =========================
def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


# =========================
# 🔥 TRAIN GAN
# =========================
def train_gan(X_minority, epochs=300, batch_size=16):

    noise_dim = 10   # random noise size
    data_dim = X_minority.shape[1]  # number of features

    generator = build_generator(noise_dim, data_dim)
    discriminator = build_discriminator(data_dim)

    # Freeze discriminator for GAN training
    discriminator.trainable = False

    gan_input = tf.keras.Input(shape=(noise_dim,))
    fake_data = generator(gan_input)
    gan_output = discriminator(fake_data)

    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    for epoch in range(epochs):

        # ===== Generate fake samples =====
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_samples = generator.predict(noise, verbose=0)

        # ===== Get real samples =====
        idx = np.random.randint(0, X_minority.shape[0], batch_size)
        real_samples = X_minority[idx]

        # ===== Combine =====
        X_combined = np.vstack([real_samples, fake_samples])
        y_combined = np.hstack([np.ones(batch_size), np.zeros(batch_size)])

        # ===== Train discriminator =====
        discriminator.trainable = True
        discriminator.train_on_batch(X_combined, y_combined)

        # ===== Train generator =====
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        y_gan = np.ones(batch_size)

        discriminator.trainable = False
        gan.train_on_batch(noise, y_gan)

        # ===== Logging =====
        if epoch % 50 == 0:
            print(f"GAN Epoch {epoch}")

    return generator


# =========================
# 🔥 GENERATE SYNTHETIC DATA
# =========================
def generate_samples(generator, n_samples):

    noise_dim = 10

    # Safety fix
    if n_samples <= 0:
        n_samples = 1

    noise = np.random.normal(0, 1, (n_samples, noise_dim))
    synthetic_data = generator.predict(noise, verbose=0)

    return synthetic_data