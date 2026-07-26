from tensorflow.keras.models import load_model

model = load_model("model/xray_model.keras")

print("Inputs :", model.inputs)
print("Outputs:", model.outputs)

print("\nLayer Names:")
for layer in model.layers:
    print(layer.name)