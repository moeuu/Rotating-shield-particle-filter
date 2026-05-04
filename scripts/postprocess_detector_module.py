from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
LATEX_FIG_DIR = (
    REPO_ROOT.parent
    / "latex"
    / "projects"
    / "ieee-ra-l-letter"
    / "sections"
    / "03_system_model"
    / "figures"
)
RAW_PNG = Path("/tmp/ral_detector_raw.png")
FINAL_PNG = Path("/tmp/ral_detector_final.png")
FINAL_PDF = LATEX_FIG_DIR / "Detector.pdf"


def load_font(size):
    """Load a Times-compatible figure font available on the workstation."""
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def label(draw, text, box_xy, target_xy, font, outline, fill=(255, 255, 255)):
    """Draw a labeled callout box with a leader line."""
    x, y = box_xy
    tx, ty = target_xy
    pad_x, pad_y = 24, 18
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 2 * pad_x
    h = bbox[3] - bbox[1] + 2 * pad_y
    rect = [x, y, x + w, y + h]
    draw.line((x + w / 2, y + h / 2, tx, ty), fill=outline, width=5)
    draw.ellipse((tx - 7, ty - 7, tx + 7, ty + 7), fill=outline)
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline=outline, width=4)
    draw.text(
        (x + pad_x - bbox[0], y + pad_y - bbox[1]),
        text,
        font=font,
        fill=(0, 0, 0),
    )


def crop_vertical_whitespace(img, margin_top=24, margin_bottom=28):
    """Crop vertical white margins while preserving the original figure width."""
    background = Image.new("RGB", img.size, (255, 255, 255))
    bbox = ImageChops.difference(img, background).getbbox()
    if bbox is None:
        return img
    top = max(0, bbox[1] - margin_top)
    bottom = min(img.height, bbox[3] + margin_bottom)
    return img.crop((0, top, img.width, bottom))


def main():
    """Composite the transparent render onto white and add readable callouts."""
    img = Image.open(RAW_PNG).convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img)
    draw = ImageDraw.Draw(img)
    font = load_font(66)

    label(draw, "CeBr3 detector", (580, 70), (915, 470), font, (0, 118, 132))
    label(draw, "Fe shield", (210, 782), (520, 690), font, (126, 91, 0))
    label(draw, "Pb shield", (1210, 70), (1340, 320), font, (74, 82, 91))

    img = img.convert("RGB")
    img = crop_vertical_whitespace(img)
    img.save(FINAL_PNG, dpi=(600, 600), optimize=True)
    img.save(FINAL_PDF, "PDF", resolution=600.0)


if __name__ == "__main__":
    main()
