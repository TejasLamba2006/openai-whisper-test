import onnxruntime as ort

print("Available providers:")
print(ort.get_available_providers())

sess = ort.InferenceSession(
    "model.onnx",
    providers=["VSINPUExecutionProvider"]
)

print("Actual providers:")
print(sess.get_providers())
