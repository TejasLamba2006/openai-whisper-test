import onnxruntime as ort

so = ort.SessionOptions()
so.log_severity_level = 0

sess = ort.InferenceSession(
    "whisper-onnx-exporter/onnx-models/tiny/encoder_embedded.onnx",
    sess_options=so,
    providers=["VSINPUExecutionProvider"]
)

print(sess.get_providers())
