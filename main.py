import onnxruntime as ort

so = ort.SessionOptions()
so.log_severity_level = 0

sess = ort.InferenceSession(
    "encoder.onnx",
    sess_options=so,
    providers=["VSINPUExecutionProvider"]
)
