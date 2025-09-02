## Blackjack Card Counter (OCR + Manual)

This tool captures on-screen card regions, OCRs their ranks, and maintains a Hi-Lo running and true count. It also supports manual input.

### Install

1) Python 3.10+
2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Tesseract OCR (optional for OCR mode):
- Windows: Install from `https://github.com/UB-Mannheim/tesseract/wiki`. Note the `tesseract.exe` path, e.g. `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`.
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

If `tesseract.exe` is not on PATH (Windows), pass `--tesseract C:\\Path\\to\\tesseract.exe`.

### Usage

Start the tool, select ROIs, then the observer runs and you can also type manual inputs.

```bash
python counter.py --decks 6 --hz 2 --debug
```

- Select one rectangle per potential card position (player + dealer spots). Press Enter when done. Esc removes last.
- Terminal commands:
  - `/hand`: new hand (clears per-hand seen positions)
  - `/shoe`: new shoe (resets counts)
  - `/quit`: exit
- Manual card entry: type ranks like `2 7 K 10 A` or `2,7,K,10,A`.

Disable OCR and use manual only:

```bash
python counter.py --no-ocr
```

Specify Tesseract path (Windows):

```bash
python counter.py --tesseract "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

### Notes

- Hi-Lo values: 2-6 = +1, 7-9 = 0, 10/J/Q/K/A = -1.
- True count = running count / decks remaining.
- OCR depends on the visual style; you may need to refine ROIs and lighting/zoom.
