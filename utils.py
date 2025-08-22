from PIL import Image, ImageFilter, ImageOps
import io, base64, requests, numpy as np, cv2

def url_to_pil(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")

def pil_to_base64(img, fmt="PNG"):
    buff = io.BytesIO()
    img.save(buff, format=fmt)
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

def to_comic_style(img: Image.Image):
    # simple cartoon-like filter (placeholder)
    # convert to OpenCV, bilateral filter + edge detection + color quantization
    cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    # edge
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,9)
    # color
    color = cv2.bilateralFilter(cv_img, 9, 300, 300)
    # combine
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(cartoon)
    return pil

def to_line_art(img: Image.Image):
    # produce a black-white line-art for coloring book
    gray = ImageOps.grayscale(img)
    inverted = ImageOps.invert(gray)
    blurred = inverted.filter(ImageFilter.GaussianBlur(radius=5))
    sketch = Image.blend(gray, blurred, alpha=0.5)
    # increase contrast threshold
    return sketch.convert("L").point(lambda x: 0 if x<128 else 255, mode='1').convert("RGBA")

def layout_photobook(imgs, layout="single"):
    # naive: just return images resized to same page size
    pages = []
    for img in imgs:
        pages.append(img.resize((1024,1024)))
    return pages

def edu_illustration(img, topic=None):
    # placeholder: return comic style and append caption text later on client
    return to_comic_style(img)
