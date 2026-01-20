import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class Encoder(layers.Layer):
    def __init__(self, latent_dim, original_dim, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.flatten = keras.layers.Flatten()
        self.dense = layers.Dense(units=latent_dim, activity_regularizer=regularizers.l1())

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense(x)
        x = x / (tf.norm(x, axis=-1, keepdims=True) + 1e-9)
        return x

class Decoder(layers.Layer):
    def __init__(self, original_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.dense = layers.Dense(units=300)
        self.dense1 = layers.Dense(units=300, activation="sigmoid")
        self.dense2 = layers.Dense(units=np.prod(original_dim))

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dense1(x)
        reconstructed = self.dense2(x)
        reshaped = tf.reshape(reconstructed, tuple((-1,) + self.original_dim))
        return reshaped

class AutoEncoder(keras.Model):
    def __init__(
            self,
            original_dim,
            latent_dim=6,
            name="autoencoder",
            **kwargs
    ):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, original_dim=original_dim)
        self.decoder = Decoder(original_dim)

    def call(self, inputs):
        concepts = self.encoder(inputs)
        reconstructed = self.decoder(concepts)
        return reconstructed


class LSTM_Encoder(layers.Layer):
    def __init__(self, latent_dim, name="lstm_encoder", **kwargs):
        super(LSTM_Encoder, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        
        # LSTM encoder with return_sequences=False to get last hidden state only
        self.lstm = layers.LSTM(
            units=latent_dim,
            return_sequences=False,
            return_state=False,
            activity_regularizer=regularizers.l1()
        )
        
    def call(self, inputs):
        # Input shape: (batch_size, timesteps, features)
        x = self.lstm(inputs)  # Output shape: (batch_size, latent_dim)
        
        # Optional: Normalize the latent representation
        x = x / (tf.norm(x, axis=-1, keepdims=True) + 1e-9)
        return x


class LSTM_Decoder(layers.Layer):
    def __init__(self, original_dim, latent_dim, name="lstm_decoder", **kwargs):
        super(LSTM_Decoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim  # Should be (timesteps, features) for reshaping
        
        # RepeatVector to create sequence from latent vector
        self.repeat = layers.RepeatVector(self.original_dim[0])
        
        # LSTM decoder with return_sequences=True for sequence output
        self.lstm = layers.LSTM(
            units=latent_dim,
            return_sequences=True,
            return_state=False,
            activation='relu'
        )
        
        # TimeDistributed Dense layers for per-timestep reconstruction
        self.time_dense1 = layers.TimeDistributed(layers.Dense(units=300))
        self.time_dense2 = layers.TimeDistributed(
            layers.Dense(units=300, activation="sigmoid")
        )
        self.time_dense3 = layers.TimeDistributed(
            layers.Dense(units=self.original_dim[1])
        )
        
    def call(self, inputs):
        # Input shape: (batch_size, latent_dim)
        
        # Repeat the latent vector to create a sequence
        x = self.repeat(inputs)  # Shape: (batch_size, timesteps, latent_dim)
        
        # Process through LSTM decoder
        x = self.lstm(x)  # Shape: (batch_size, timesteps, lstm_units)
        
        # Apply Dense layers to each timestep
        x = self.time_dense1(x)
        x = self.time_dense2(x)
        reconstructed = self.time_dense3(x)  # Shape: (batch_size, timesteps, features)
        
        return reconstructed
    

class LSTM_AutoEncoder(keras.Model):
    def __init__(
        self,
        original_dim,  # Should be (timesteps, features)
        latent_dim=32,  # Increased for LSTM (typical range: 32-256)
        name="lstm_autoencoder",
        **kwargs
    ):
        super(LSTM_AutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        
        # For LSTM, original_dim should be a tuple: (timesteps, features)
        if not isinstance(original_dim, tuple) or len(original_dim) != 2:
            raise ValueError("original_dim must be a tuple: (timesteps, features)")
            
        self.encoder = LSTM_Encoder(latent_dim=latent_dim)
        self.decoder = LSTM_Decoder(
            original_dim=original_dim,
            latent_dim=latent_dim
        )
        
    def call(self, inputs):
        # Input shape should be: (batch_size, timesteps, features)
        concepts = self.encoder(inputs)
        reconstructed = self.decoder(concepts)
        return reconstructed
    
    def encode(self, inputs):
        """Convenience method to get latent representation"""
        return self.encoder(inputs)
    
    def decode(self, latent_vectors):
        """Convenience method to decode from latent space"""
        return self.decoder(latent_vectors)