import argparse
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Dict
try:
	import tkinter as tk
	from tkinter import ttk
	_TK_AVAILABLE = True
except Exception:
	_TK_AVAILABLE = False

import cv2
import numpy as np
from mss import mss
from PIL import Image

try:
	import pytesseract  # Optional; requires Tesseract installed on system
	_TESS_AVAILABLE = True
except Exception:
	_TESS_AVAILABLE = False


HI_LO_VALUES: Dict[str, int] = {
	"2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
	"7": 0, "8": 0, "9": 0,
	"10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}


@dataclass
class ShoeState:
	decks: float
	seen_cards: int = 0
	running_count: int = 0

	def cards_remaining(self) -> int:
		return max(0, int(self.decks * 52) - self.seen_cards)

	def decks_remaining(self) -> float:
		return max(0.0, self.cards_remaining() / 52.0)

	def true_count(self) -> float:
		remaining = self.decks_remaining()
		return self.running_count / remaining if remaining > 0 else float(self.running_count)


@dataclass
class Roi:
	left: int
	top: int
	width: int
	height: int

	def to_mss(self) -> dict:
		return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


@dataclass
class SessionConfig:
	decks: float
	poll_hz: float = 2.0
	use_ocr: bool = True
	debug: bool = False


@dataclass
class SessionState:
	shoe: ShoeState
	rois: List[Roi] = field(default_factory=list)
	observed_this_hand: Set[Tuple[int, str]] = field(default_factory=set)
	lock: threading.Lock = field(default_factory=threading.Lock)
	stop_event: threading.Event = field(default_factory=threading.Event)


def draw_rois_on_image(img: np.ndarray, rois: List[Roi]) -> np.ndarray:
	out = img.copy()
	for idx, r in enumerate(rois):
		cv2.rectangle(out, (r.left, r.top), (r.left + r.width, r.top + r.height), (0, 255, 0), 2)
		cv2.putText(out, f"{idx}", (r.left + 4, r.top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	return out


def select_rois(full_img: np.ndarray) -> List[Roi]:
	win = full_img.copy()
	rois: List[Roi] = []
	start_pt: Optional[Tuple[int, int]] = None
	zoom: float = 1.0

	window_name = "Select ROIs - drag rectangles, +/- to zoom, ENTER when done"

	def to_fullres(pt: Tuple[int, int]) -> Tuple[int, int]:
		x, y = pt
		if zoom <= 0:
			return x, y
		return int(x / zoom), int(y / zoom)

	def on_mouse(event, x, y, flags, param):
		nonlocal start_pt, win
		# Map coords back to full-res image space according to current zoom
		x_full, y_full = to_fullres((x, y))
		if event == cv2.EVENT_LBUTTONDOWN:
			start_pt = (x_full, y_full)
		elif event == cv2.EVENT_LBUTTONUP and start_pt is not None:
			x0, y0 = start_pt
			left, top = min(x0, x_full), min(y0, y_full)
			width, height = abs(x_full - x0), abs(y_full - y0)
			if width > 5 and height > 5:
				rois.append(Roi(left, top, width, height))
				win = draw_rois_on_image(full_img, rois)
			start_pt = None

	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(window_name, 1280, 720)
	win = draw_rois_on_image(win, rois)
	cv2.setMouseCallback(window_name, on_mouse)
	while True:
		# Render with zoom applied for display only
		disp = win if zoom == 1.0 else cv2.resize(win, (int(win.shape[1] * zoom), int(win.shape[0] * zoom)))
		cv2.imshow(window_name, disp)
		key = cv2.waitKey(20) & 0xFF
		if key in (13, 10):  # Enter
			break
		elif key == 27:  # Esc to clear last
			if rois:
				rois.pop()
				win = draw_rois_on_image(full_img, rois)
		elif key in (43, 61):  # '+' or '=' increases zoom
			zoom = min(2.5, zoom + 0.1)
		elif key in (45, 95):  # '-' or '_' decreases zoom
			zoom = max(0.4, zoom - 0.1)
	cv2.destroyAllWindows()
	return rois


def grab_screen(monitor: dict) -> np.ndarray:
	with mss() as sct:
		img = np.array(sct.grab(monitor))
		# Drop alpha channel, convert BGRA -> BGR
		img = img[:, :, :3]
		return img


def preprocess_for_rank(gray: np.ndarray) -> np.ndarray:
	# Increase contrast and threshold to emphasize rank glyphs
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
	return thr


def ocr_rank(crop: np.ndarray) -> Optional[str]:
	if not _TESS_AVAILABLE:
		return None
	# Focus on top-left corner where rank is usually displayed
	h, w = crop.shape[:2]
	corner = crop[0:int(h * 0.35), 0:int(w * 0.35)]
	gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
	thr = preprocess_for_rank(gray)
	pil = Image.fromarray(thr)
	try:
		text = pytesseract.image_to_string(
			pil,
			config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789JQKA'
		)
		text = text.strip().upper().replace("O", "0")
		# Normalize to rank symbol
		if text in {"J", "Q", "K", "A"}:
			return text
		# Try to parse 10 specially
		if "10" in text or text == "IO":
			return "10"
		# Single digit 2-9
		if len(text) >= 1 and text[0] in list("23456789"):
			return text[0]
		return None
	except Exception:
		return None


def rank_to_count_delta(rank: str) -> int:
	return HI_LO_VALUES.get(rank, 0)


def format_counts(shoe: ShoeState) -> str:
	return (
		f"Seen: {shoe.seen_cards}  "
		f"Running: {shoe.running_count:+d}  "
		f"True: {shoe.true_count():+.2f}  "
		f"Decks left: {shoe.decks_remaining():.2f}"
	)


def observe_loop(cfg: SessionConfig, state: SessionState) -> None:
	interval = 1.0 / max(0.1, cfg.poll_hz)
	with mss() as sct:
		while not state.stop_event.is_set():
			for idx, roi in enumerate(state.rois):
				img = np.array(sct.grab(roi.to_mss()))[:, :, :3]
				rank = ocr_rank(img) if cfg.use_ocr else None
				if cfg.debug:
					debug_show = img.copy()
					cv2.imshow(f"roi_{idx}", debug_show)
					cv2.waitKey(1)
				if rank is None:
					continue
				with state.lock:
					key = (idx, rank)
					if key in state.observed_this_hand:
						continue
					state.observed_this_hand.add(key)
					state.shoe.seen_cards += 1
					delta = rank_to_count_delta(rank)
					state.shoe.running_count += delta
					print(f"Observed {rank} at pos {idx}: {format_counts(state.shoe)}")
			time.sleep(interval)
	if cfg.debug:
		cv2.destroyAllWindows()


def run_gui(cfg: SessionConfig, state: SessionState) -> None:
	if not _TK_AVAILABLE:
		print("Tkinter GUI not available. Falling back to terminal mode.")
		manual_input_loop(state)
		return
	root = tk.Tk()
	root.title("Blackjack Counter")
	root.geometry("360x220")

	counts_var = tk.StringVar(value=format_counts(state.shoe))
	status_label = ttk.Label(root, textvariable=counts_var, font=("Segoe UI", 12))
	status_label.pack(pady=10)

	def update_counts():
		with state.lock:
			counts_var.set(format_counts(state.shoe))
		if not state.stop_event.is_set():
			root.after(250, update_counts)

	def on_new_hand():
		with state.lock:
			state.observed_this_hand.clear()

	def on_new_shoe():
		with state.lock:
			state.shoe.seen_cards = 0
			state.shoe.running_count = 0
			state.observed_this_hand.clear()

	def on_toggle_ocr():
		cfg.use_ocr = not cfg.use_ocr
		ocr_btn.configure(text=f"OCR: {'On' if cfg.use_ocr else 'Off'}")

	def on_quit():
		state.stop_event.set()
		root.destroy()

	btn_frame = ttk.Frame(root)
	btn_frame.pack(pady=8)

	ocr_btn = ttk.Button(btn_frame, text=f"OCR: {'On' if cfg.use_ocr else 'Off'}", command=on_toggle_ocr)
	ocr_btn.grid(row=0, column=0, padx=5, pady=5)

	new_hand_btn = ttk.Button(btn_frame, text="New Hand", command=on_new_hand)
	new_hand_btn.grid(row=0, column=1, padx=5, pady=5)

	new_shoe_btn = ttk.Button(btn_frame, text="New Shoe", command=on_new_shoe)
	new_shoe_btn.grid(row=0, column=2, padx=5, pady=5)

	quit_btn = ttk.Button(root, text="Quit", command=on_quit)
	quit_btn.pack(pady=10)

	# Manual rank buttons
	manual_frame = ttk.LabelFrame(root, text="Add Card (Manual)")
	manual_frame.pack(pady=6)

	manual_ranks = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
	def on_add(rank: str):
		with state.lock:
			state.shoe.seen_cards += 1
			state.shoe.running_count += rank_to_count_delta(rank)

	for i, r in enumerate(manual_ranks):
		btn = ttk.Button(manual_frame, text=r, width=3, command=lambda rr=r: on_add(rr))
		btn.grid(row=i // 7, column=i % 7, padx=3, pady=3)

	info = ttk.Label(root, text="Controls: Select ROIs on start. GUI updates live.", font=("Segoe UI", 9))
	info.pack(pady=4)

	update_counts()
	root.mainloop()


def manual_input_loop(state: SessionState) -> None:
	print("Manual input: type ranks like 2-10,J,Q,K,A. Commands: /hand, /shoe, /quit")
	while True:
		try:
			line = input("> ").strip().upper()
		except EOFError:
			break
		if not line:
			continue
		if line in {"/Q", "/QUIT"}:
			break
		if line in {"/H", "/HAND"}:
			with state.lock:
				state.observed_this_hand.clear()
			print("New hand.")
			continue
		if line in {"/S", "/SHOE"}:
			with state.lock:
				state.shoe.seen_cards = 0
				state.shoe.running_count = 0
				state.observed_this_hand.clear()
			print("New shoe counters reset.")
			continue
		# Parse ranks possibly separated by spaces/commas
		ranks = [tok.strip() for tok in line.replace(",", " ").split()]
		for rank in ranks:
			if rank not in HI_LO_VALUES:
				print(f"Ignored '{rank}'. Valid: 2-10,J,Q,K,A")
				continue
			with state.lock:
				state.shoe.seen_cards += 1
				state.shoe.running_count += rank_to_count_delta(rank)
			print(f"Added {rank}. {format_counts(state.shoe)}")


def main():
	parser = argparse.ArgumentParser(description="Basic Blackjack Card Counter with optional OCR")
	parser.add_argument("--decks", type=float, default=8.0, help="Number of decks in shoe")
	parser.add_argument("--hz", type=float, default=2.0, help="OCR polling frequency (Hz)")
	parser.add_argument("--no-ocr", action="store_true", help="Disable OCR; manual mode only")
	parser.add_argument("--debug", action="store_true", help="Show ROI debug windows")
	parser.add_argument("--tesseract", type=str, default="", help="Path to tesseract.exe if not on PATH (Windows)")
	parser.add_argument("--gui", action="store_true", help="Show a simple GUI instead of terminal input")
	args = parser.parse_args()

	if args.tesseract:
		try:
			pytesseract.pytesseract.tesseract_cmd = args.tesseract  # type: ignore[attr-defined]
		except Exception:
			pass

	if not _TESS_AVAILABLE and not args.no_ocr:
		print("Tesseract OCR not available. Falling back to manual mode. Install Tesseract or run with --no-ocr.")

	cfg = SessionConfig(
		decks=args.decks,
		poll_hz=args.hz,
		use_ocr=(not args.no_ocr and _TESS_AVAILABLE),
		debug=args.debug,
	)
	state = SessionState(shoe=ShoeState(decks=cfg.decks))

	# Take a full-screen screenshot for ROI selection
	with mss() as sct:
		mon = sct.monitors[1]
		full = np.array(sct.grab(mon))[:, :, :3]

	print("Select one rectangle per potential card position (player + dealer). Press Enter when done. Esc removes last.")
	rois = select_rois(full)
	if not rois:
		print("No ROIs selected. Exiting.")
		return
	state.rois = rois

	print("Controls: In terminal use /hand to reset per-hand, /shoe to reset shoe, /quit to exit.")
	print("Starting observer...")

	observer = threading.Thread(target=observe_loop, args=(cfg, state), daemon=True)
	observer.start()
	try:
		if args.gui:
			run_gui(cfg, state)
		else:
			manual_input_loop(state)
	finally:
		state.stop_event.set()
		observer.join(timeout=1.0)

	print("Goodbye.")


if __name__ == "__main__":
	main()


