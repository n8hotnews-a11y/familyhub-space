import gradio as gr
from utils import url_to_pil, pil_to_base64, to_comic_style, to_line_art, layout_photobook, edu_illustration
from typing import Dict, Any

def process(payload: Dict[str, Any]):
    # payload example: {"mode":"comic","image_urls":[...],"options":{...}}
    mode = payload.get("mode", "comic")
    image_urls = payload.get("image_urls", []) or []
    options = payload.get("options", {})

    pages = []
    try:
        # load images
        pil_images = [ url_to_pil(u) for u in image_urls ]
    except Exception as e:
        return {"status":"error","message":f"cannot load images: {str(e)}","pages":[]}

    if mode == "comic":
        # convert each image to comic page
        for i, im in enumerate(pil_images):
            page = to_comic_style(im)
            pages.append({"page_index": i, "image_b64": pil_to_base64(page)})
    elif mode == "coloringbook":
        for i, im in enumerate(pil_images):
            page = to_line_art(im)
            pages.append({"page_index": i, "image_b64": pil_to_base64(page, fmt="PNG")})
    elif mode == "photobook":
        photopages = layout_photobook(pil_images, layout=options.get("layout","single"))
        for i, p in enumerate(photopages):
            pages.append({"page_index": i, "image_b64": pil_to_base64(p)})
    elif mode == "edubook":
        topic = options.get("topic","")
        for i, im in enumerate(pil_images):
            p = edu_illustration(im, topic=topic)
            pages.append({"page_index": i, "image_b64": pil_to_base64(p)})
    else:
        return {"status":"error","message":"unknown mode","pages": []}

    return {"status":"ok", "pages": pages, "meta": {"mode": mode, "count": len(pages)}}

# Expose a JSON-to-JSON Gradio interface:
iface = gr.Interface(fn=process, inputs=gr.JSON(label="payload"), outputs=gr.JSON(label="response"), allow_flagging="never",
                     title="AI Comic Space - JSON API")

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
