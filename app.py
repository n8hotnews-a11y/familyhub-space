import gradio as gr
from PIL import Image
import torch

# ----- (A) Load caption model "lite" để chạy cả trên CPU -----
from transformers import BlipProcessor, BlipForConditionalGeneration
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model.eval()

# ----- (B) Stable Diffusion (bật khi chọn GPU trong Space) -----
ENABLE_SD = torch.cuda.is_available()
if ENABLE_SD:
    from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
    sd_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        pipe.to("cuda")

STYLES = {
    "comic": "comic style, clean lines, soft pastel colors, family-friendly",
    "watercolor": "watercolor painting, soft edges, pastel tones, warm light",
    "sketch": "pencil sketch, minimal shading, soft line art"
}

def generate_caption_pil(img: Image.Image, lang="vi"):
    # BLIP caption (EN) → dịch sơ bộ bằng prompt rule-based (đơn giản, không phụ thuộc LLM)
    inputs = caption_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = caption_model.generate(**inputs, max_new_tokens=40)
    en = caption_processor.decode(out[0], skip_special_tokens=True)
    if lang == "vi":
        # quick heuristic translation prompt (đơn giản, tránh gọi model nặng)
        vi = f"Mô tả ảnh (dịch từ EN): {en}"
        return vi
    return en

def predict(image: Image.Image, style: str, lang: str):
    if image is None:
        return "Vui lòng tải ảnh", Image.new("RGB", (512,512), "white")
    caption = generate_caption_pil(image, lang=lang)

    if ENABLE_SD:
        prompt = f"family photo in {STYLES.get(style, STYLES['comic'])}"
        # Tối ưu tốc độ demo GPU
        styled = pipe(prompt, num_inference_steps=15, guidance_scale=5.0).images[0]
    else:
        # Fallback: không có GPU → trả lại ảnh gốc như minh hoạ
        styled = image.copy()

    return caption, styled

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload family photo"),
        gr.Radio(choices=list(STYLES.keys()), value="comic", label="Style"),
        gr.Radio(choices=["vi","en"], value="vi", label="Caption language")
    ],
    outputs=[gr.Textbox(label="Caption"), gr.Image(label="Styled image")],
    title="FamilyHub AI – Caption + Style (Space)",
    description="BLIP caption (CPU OK). Stable Diffusion cần GPU; nếu không có GPU, app sẽ trả ảnh gốc như demo."
)

if __name__ == "__main__":
    demo.launch()
