from PIL import Image, ImageFilter, ImageOps
import io, base64, requests, numpy as np, cv2


TIMEOUT = 20


def url_to_pil(url: str) -> Image.Image:
r = requests.get(url, timeout=TIMEOUT)
r.raise_for_status()
img = Image.open(io.BytesIO(r.content))
return img.convert("RGBA")


def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
buff = io.BytesIO()
img.save(buff, format=fmt)
return f"data:image/{fmt.lower()};base64," + base64.b64encode(buff.getvalue()).decode("utf-8")


# ---- Analysis helpers ----


def analyze_simple(img: Image.Image) -> dict:
arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
# face-ish detection via OpenCV Haar (optional minimal)
gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
faces = []
try:
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.2, 5)
except Exception:
faces = []
face_count = 0 if faces is None else len(faces)
# dominant color (very rough)
small = cv2.resize(arr, (32, 32))
avg_bgr = small.mean(axis=(0,1)).tolist()
avg_rgb = [int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])]
caption = f"Ảnh có {face_count} khuôn mặt, tông màu gần RGB{tuple(avg_rgb)}"
return {"faces": face_count, "avg_rgb": avg_rgb, "caption": caption}


# ---- Render helpers ----


def comic_filter(img: Image.Image) -> Image.Image:
cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,9)
color = cv2.bilateralFilter(cv_img, 9, 300, 300)
cartoon = cv2.bitwise_and(color, color, mask=edges)
cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
return Image.fromarray(cartoon)


def line_art(img: Image.Image) -> Image.Image:
gray = ImageOps.grayscale(img)
inverted = ImageOps.invert(gray)
blurred = inverted.filter(ImageFilter.GaussianBlur(radius=5))
sketch = Image.blend(gray, blurred, alpha=0.5)
return sketch.convert("L").point(lambda x: 0 if x<128 else 255, mode='1').convert("RGBA")


def photobook_layout(imgs, layout="single"):
pages = []
for im in imgs:
pages.append(im.convert("RGBA").resize((1024,1024)))
return pages
