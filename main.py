import datetime
import logging
import os
import threading
from equation.models import ModelInfo, load_nougat_model
from equation.nougat import inner_process
from equation.settings import settings
import base64
import io

from pyfunvice import (
    app_service,
    start_app,
    app_service_get,
)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(thread)d] [%(levelname)s] %(message)s"
)


model: ModelInfo = None


def post_fork_func():
    global model
    model = load_nougat_model()


@app_service(path="/api/v1/parser/ppl/equation", inparam_type="flat")
async def process(
    requestId: str,
    content_base64: str,
    type_block_idx: int,
    type_block_num: int,
    block_idx: int,
):
    logging.info(
        "POST request"
        + f" [P{os.getpid()}][T{threading.current_thread().ident}][{requestId}] "
        + f"type_block_idx: {type_block_idx}"
        + f", type_block_num: {type_block_num}"
        + f", block_idx: {block_idx}"
    )
    image_byte = base64.b64decode(content_base64)
    image = io.BytesIO(image_byte)
    text = inner_process([image], [2048], model, 1)
    logging.info(
        "Return result"
        + f" [P{os.getpid()}][T{threading.current_thread().ident}][{requestId}] "
        + f"type_block_idx: {type_block_idx}"
        + f", type_block_num: {type_block_num}"
        + f", block_idx: {block_idx}"
        + f", text: {text}"
    )
    return {"text": text}


@app_service_get(path="/health")
async def health(data: dict) -> dict:
    time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"timestamp": time_string}


if __name__ == "__main__":
    start_app(workers=settings.WORKER_NUM, port=8000, post_fork_func=post_fork_func)
