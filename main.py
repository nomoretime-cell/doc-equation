from equation.models import ModelInfo, load_nougat_model
from equation.nougat import inner_process
from equation.settings import settings
import base64
import io

from pyfunvice import (
    app_service,
    start_app,
)


model: ModelInfo = None


def post_fork_func():
    global model
    model = load_nougat_model()


@app_service(path="/api/v1/parser/ppl/equation", inparam_type="flat")
async def process(content_base64: str, page_idx: int, block_idx: int):
    image_byte = base64.b64decode(content_base64)
    image = io.BytesIO(image_byte)
    text = inner_process([image], [2048], model, 1)
    return {"text": text}


if __name__ == "__main__":
    start_app(workers=settings.WORKER_NUM, port=8004, post_fork_func=post_fork_func)
