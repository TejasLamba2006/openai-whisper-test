import onnx

model = onnx.load(
    "whisper-onnx-exporter/onnx-models/tiny/decoder.onnx",
    load_external_data=True
)

onnx.save_model(
    model,
    "whisper-onnx-exporter/onnx-models/tiny/decoder_embedded.onnx",
    save_as_external_data=False
)