import tensorflow as tf

# Load the model from the H5 file
model = tf.keras.models.load_model('FRUpdated\model_78.h5')

# Get the model's weights and biases
model_weights = model.get_weights()

# You can access individual layers' weights and biases like this:
for layer in model.layers:
    layer_weights = layer.get_weights()
    if layer_weights:
        print(f"Layer {layer.name} weights shape: {layer_weights[0].shape}")
        print(f"Layer {layer.name} biases shape: {layer_weights[1].shape}")
