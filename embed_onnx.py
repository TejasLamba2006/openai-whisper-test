import onnx

model = onnx.load(
    "whisper-onnx-exporter/onnx-models/tiny/encoder.onnx",
    load_external_data=True
)

onnx.save_model(
    model,
    "whisper-onnx-exporter/onnx-models/tiny/encoder_embedded.onnx",
    save_as_external_data=False
)