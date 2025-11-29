# main.py
import os
from pathlib import Path
from collections import Counter

from ultralytics import YOLO
from PIL import Image


# ───────────────────────────────────────────────
# KONFIGURACJA ŚCIEŻEK
# ───────────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
TILES_DIR = OUTPUT_DIR / "tiles"

IMAGE_PATH = DATA_DIR / "testowy.tiff"   # tu wstaw swoją ortofotomapę

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TILES_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────────────────────
# ŁADOWANIE MODELU
# ───────────────────────────────────────────────
def load_model():
    """
    Próbuje załadować lokalny plik yolo10s.pt.
    Jeśli go nie ma, używa nazwy 'yolo10s' (Ultralytics pobierze wagę sam).
    """
    weights_path = Path("yolov10s.pt")
    if weights_path.exists():
        print(f"[INFO] Ładuję lokalny model z {weights_path}")
        return YOLO(str(weights_path))
    else:
        print("[INFO] Nie znaleziono yolo10s.pt, używam nazwy 'yolo10s' (auto-download).")
        return YOLO("yolo10s")  # lub "yolo10n" jeśli chcesz jeszcze szybszy model


# ───────────────────────────────────────────────
# PRZYGOTOWANIE OBRAZU
# ───────────────────────────────────────────────
def prepare_image(image_path: Path) -> Path:
    """
    Wczytuje TIFF, konwertuje do RGB i zapisuje jako JPG.
    Zwraca ścieżkę do JPG.
    """
    print(f"[INFO] Wczytuję obraz: {image_path}")
    img = Image.open(image_path)
    img = img.convert("RGB")  # na wszelki wypadek, jeśli TIFF ma 4 kanały albo inne formaty

    rgb_path = OUTPUT_DIR / "testowy_rgb.jpg"
    img.save(rgb_path, format="JPEG", quality=95)
    print(f"[INFO] Zapisano wersję RGB: {rgb_path}")
    return rgb_path


# ───────────────────────────────────────────────
# DETEKCJA NA CAŁYM OBRAZIE
# ───────────────────────────────────────────────
def detect_full_image(model: YOLO, image_path: Path):
    """
    Detekcja na całym obrazie w większej rozdzielczości (imgsz=1280).
    Zapisuje wynik z narysowanymi boxami w outputs/full/.
    """
    print("[INFO] Detekcja na całym obrazie...")
    results = model(
        str(image_path),
        conf=0.1,        # niższy próg, żeby coś złapać
        imgsz=1280,      # większa rozdzielczość wejścia
        save=True,
        project=str(OUTPUT_DIR),
        name="full"      # wyniki w outputs/full/
    )

    r = results[0]
    print(f"[INFO] Liczba wykrytych obiektów (pełny obraz): {len(r.boxes)}")
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f" - {cls_name:>10} | conf={conf:.2f} | bbox={xyxy}")


# ───────────────────────────────────────────────
# DETEKCJA NA KAFELKACH
# ───────────────────────────────────────────────
def detect_on_tiles(model: YOLO, image_path: Path, tile_size: int = 1024):
    """
    Tnie obraz na kafelki tile_size x tile_size i robi detekcję na każdym.
    Zapisuje kafelki z detekcjami i wypisuje sumaryczne statystyki.
    """
    print("[INFO] Detekcja na kafelkach...")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    print(f"[INFO] Rozmiar obrazu: {w}x{h}")

    class_counter = Counter()
    tile_index = 0

    for left in range(0, w, tile_size):
        for top in range(0, h, tile_size):
            right = min(left + tile_size, w)
            bottom = min(top + tile_size, h)

            tile = img.crop((left, top, right, bottom))
            tile_path = TILES_DIR / f"tile_{tile_index}_{left}_{top}.jpg"
            tile.save(tile_path, format="JPEG", quality=95)

            # detekcja na kafelku
            results = model(
                str(tile_path),
                conf=0.1,
                imgsz=tile_size,
                save=True,
                project=str(TILES_DIR),
                name=f"pred",   # Ultralytics zrobi TILES_DIR/pred/
                exist_ok=True
            )

            r = results[0]
            if len(r.boxes) > 0:
                print(f"[INFO] Kafelek {tile_index} ({left},{top}) – wykryto {len(r.boxes)} obiektów")
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names[cls_id]
                    class_counter[cls_name] += 1
            else:
                # Jak chcesz, możesz usuwać kafelki bez detekcji, żeby nie zaśmiecały dysku
                # os.remove(tile_path)
                pass

            tile_index += 1

    print("\n[INFO] Podsumowanie detekcji na kafelkach:")
    if class_counter:
        for cls_name, count in class_counter.most_common():
            print(f" - {cls_name:>10}: {count}")
    else:
        print(" - Brak detekcji na kafelkach.")


# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────
def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {IMAGE_PATH}")

    model = load_model()
    rgb_image_path = prepare_image(IMAGE_PATH)

    # 1) detekcja na całym obrazie
    detect_full_image(model, rgb_image_path)

    # 2) detekcja na kafelkach (opcjonalnie – możesz zakomentować jeśli za wolne)
    detect_on_tiles(model, rgb_image_path, tile_size=1024)


if __name__ == "__main__":
    main()
