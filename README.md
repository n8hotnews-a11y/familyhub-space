# AI Multi-Book Space (Gradio JSON API)
## Run locally
pip install -r requirements.txt python app.py
## Request format
json
{
"mode": "comic|photobook|coloringbook|edubook|analyze",
"image_urls": ["https://..."],
"options": {"layout":"single","topic":"ocean"}
}
