import os
from gradio_app import demo

if __name__ == "__main__":
    # Hugging Face Spaces usa a porta 7860 por padrão
    demo.launch(server_name="0.0.0.0", server_port=7860)
