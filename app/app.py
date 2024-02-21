import io
from app.algorithm.kohonen import Kohonen
import numpy as np
from PIL import Image
from fastapi import FastAPI, Response

app = FastAPI()


@app.get(
    "/kohonen", responses={200: {"content": {"image/png": {}}}}, response_class=Response
)
async def kohonen():
    """
    Kohonen Self-Organizing Map (SOM) algorithm
    """
    try:
        kohonen_som = Kohonen(
            None, random=True, input_size=20, width=100, height=100, max_iterations=1000
        )

        kohonen_som.fit()

        image_input_layer = Image.fromarray(
            kohonen_som.get_input_layer().vectors.astype(np.uint8)
        ).resize((400, 20), resample=Image.Resampling.NEAREST)

        image_node_map = Image.fromarray(
            kohonen_som.get_output_layer().nodes.astype(np.uint8)
        ).resize((400, 400))

        concatenated_image = Image.new(
            "RGB",
            (image_input_layer.width, image_input_layer.height + image_node_map.height),
        )
        concatenated_image.paste(image_input_layer, (0, 0))
        concatenated_image.paste(image_node_map, (0, image_input_layer.height))

        with io.BytesIO() as buf:
            concatenated_image.save(buf, format="PNG")
            im_concatenated_bytes = buf.getvalue()

        headers = {
            "Content-Disposition": 'inline; filename="kohonen_som_input_output.png"'
        }
        return Response(im_concatenated_bytes, headers=headers, media_type="image/png")

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return Response(error_message, status_code=500)
