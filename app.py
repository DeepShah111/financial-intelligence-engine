import os
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
from gradio_app import demo

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, show_api=False)