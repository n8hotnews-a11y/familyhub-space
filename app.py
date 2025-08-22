import gradio as gr
from typing import Dict, Any
from utils import url_to_pil, pil_to_b64, analyze_simple, comic_filter, line_art, photobook_layout


def process(payload: Dict[str, Any]):
mode = payload.get("mode", "comic")
image_urls = payload.get("image_urls", []) or []
options = payload.get("options", {})
if not image_urls:
return {"status":"error","message":"no image_urls","pages":[]}
try:
pil_imgs = [url_to_pil(u) for u in image_urls]
except Exception as e:
return {"status":"error","message":f"load error: {e}","pages":[]}


pages = []
if mode == "analyze":
analyses = [analyze_simple(im) for im in pil_imgs]
return {"status":"ok","analyses":analyses,"pages":[]}


if mode == "comic":
for i, im in enumerate(pil_imgs):
out = comic_filter(im)
pages.append({"page_index": i, "image_b64": pil_to_b64(out)})
elif mode == "coloringbook":
for i, im in enumerate(pil_imgs):
out = line_art(im)
pages.append({"page_index": i, "image_b64": pil_to_b64(out)})
elif mode == "photobook":
layout = options.get("layout","single")
outs = photobook_layout(pil_imgs, layout=layout)
for i, im in enumerate(outs):
pages.append({"page_index": i, "image_b64": pil_to_b64(im)})
elif mode == "edubook":
topic = options.get("topic","")
# demo: reuse comic filter; later replace with real pipeline conditioned on topic
for i, im in enumerate(pil_imgs):
out = comic_filter(im)
pages.append({"page_index": i, "image_b64": pil_to_b64(out)})
else:
return {"status":"error","message":"unknown mode","pages":[]}


return {"status":"ok","pages": pages, "meta": {"mode": mode, "count": len(pages)}}


iface = gr.Interface(
fn=process,
inputs=gr.JSON(label="payload"),
outputs=gr.JSON(label="response"),
allow_flagging="never",
title="AI Multi‑Book Space (JSON API)"
)


if __name__ == "__main__":
iface.launch()
