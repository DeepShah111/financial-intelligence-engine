# Entry point for Render: import the Gradio app and bind to Render's port.
import os
from gradio_app import demo

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)