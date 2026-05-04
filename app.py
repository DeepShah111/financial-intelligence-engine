# Entry point for HuggingFace Spaces
# Spaces expects app.py — this simply imports and runs gradio_app.py
from gradio_app import demo

demo.queue()
demo.launch()