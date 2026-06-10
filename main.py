#!/usr/bin/env python3

from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="openai/whisper-tiny",
    output="whisper_tiny_onnx"
)