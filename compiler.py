import argparse
import os
import struct
import subprocess
import numpy as np

import sys

# Import the existing engine components (now in the same directory)
from ascii_video_player2 import VideoDecoder, AsciiMapper, MODE_QUANTIZE_BITS
from codec import encode_frame, DEFAULT_LEVEL, ProfileEncoder
from watermark import Watermarker, encode_payload as _wm_encode_payload
from importlib.metadata import version as _pkg_version, PackageNotFoundError

def extract_audio(video_path: str, output_path: str):
    print(f"[Audio] Attempting to extract audio to {output_path}...")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path, 
                "-vn", "-acodec", "libmp3lame", "-ab", "128k", "-ar", "44100", 
                output_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        print("[Audio] Audio extracted successfully.")
    except FileNotFoundError:
        print("[Audio] WARNING: FFmpeg not found on this system.")
        print("[Audio] The video will be compiled silently. Please install FFmpeg for audio support.")
    except subprocess.CalledProcessError:
        print("[Audio] WARNING: FFmpeg failed to extract audio. The video will be compiled silently.")

def get_video_dimensions(decoder):
    # Quick utility to get dims
    return decoder.vid_w, decoder.vid_h

def compile_video(args):
    video_path = args.video
    if not os.path.exists(video_path):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check 1: Inside the root project's videos/ folder
        fallback_1 = os.path.join(root_dir, 'videos', video_path)
        
        if os.path.exists(fallback_1):
            video_path = fallback_1
        else:
            print(f"Error: File not found -> {args.video} (Also checked videos/ folder)")
            return

    out_name = args.out or os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static_player')
    os.makedirs(out_dir, exist_ok=True)
    
    ascf_path = os.path.join(out_dir, f"{out_name}.ascf")
    audio_path = os.path.join(out_dir, f"{out_name}.mp3")
    
    pixel_mode = args.pixel
    render_mode = args.mode
    cols = args.cols
    tolerance = args.tolerance
    level = 9 if args.hard else 3
    pixel_qb = args.quantize  # bits to drop per channel in pixel mode (0 = lossless)

    # 1. Extract audio
    extract_audio(video_path, audio_path)

    # 2. Setup Decoder
    print(f"[Video] Initializing decoder for {video_path}...")
    decoder = VideoDecoder(video_path, cols, args.rows, skip_gray=pixel_mode)
    
    # Calculate rows (from stream_server logic)
    vid_w, vid_h = get_video_dimensions(decoder)
    ratio = vid_w / max(vid_h, 1)
    if args.rows == 0:
        if pixel_mode:
            rows = max(1, round(cols / ratio))
        else:
            rows = max(1, round(cols / ratio / 2))
    else:
        rows = args.rows

    # Update decoder with actual rows if it was auto-calculated
    # Actually, VideoDecoder doesn't allow changing rows after init, so we must recreate if rows was 0
    if args.rows == 0:
        decoder.release()
        decoder = VideoDecoder(video_path, cols, rows, skip_gray=pixel_mode)

    mapper = AsciiMapper()
    source_fps = decoder.fps
    
    # Decimation logic
    MAX_FPS = 30
    if source_fps > MAX_FPS:
        skip_n = round(source_fps / MAX_FPS)
        effective_fps = source_fps / skip_n
    else:
        skip_n = 1
        effective_fps = source_fps

    print(f"[Compiler] Dimensions: {cols}x{rows} | Mode: {render_mode} | Pixel: {pixel_mode} | FPS: {effective_fps:.1f}")

    # Opt-in lossy DCT profile (tag 4): a separate profile, not a tag-race competitor.
    profile_enc = None
    if args.profile:
        if not pixel_mode:
            print("Error: --profile requires --pixel (the lossy DCT profile is pixel mode only).")
            decoder.release()
            return
        # 8x8 blocks over 4:2:0 planes require cols/rows to be multiples of 16.
        pc = ((cols + 15) // 16) * 16
        pr = ((rows + 15) // 16) * 16
        if pc != cols or pr != rows:
            cols, rows = pc, pr
            decoder.release()
            decoder = VideoDecoder(video_path, cols, rows, skip_gray=pixel_mode)
        profile_enc = ProfileEncoder(cols, rows, args.qf)
        print(f"[Compiler] Lossy DCT profile (tag 4) ON | QF={args.qf} | grid padded to {cols}x{rows}")

    char_byte_lut = np.array([ord(c) for c in mapper._lut], dtype=np.uint8)
    # Same proportional gray->index mapping the live server uses, via LUT.
    gray_index_lut = mapper.index_lut()
    qb = MODE_QUANTIZE_BITS.get(render_mode, 0)
    qb_mask = np.uint8((0xFF << qb) & 0xFF) if qb else None

    # ── FORENSIC WATERMARK ──────────────────────────────────────────
    # Invisible keyed 10-digit mark, identical embedding math to the live
    # server (watermark.py). Pixel mode + ASCII colour modes: ±beta luma
    # dither on cell colours; monochrome ASCII: ±1 density-rank dither.
    watermarker = None
    if getattr(args, "watermark", None):
        if args.profile:
            print("Error: --watermark cannot be combined with --profile "
                  "(the lossy DCT quantizer destroys the luma dither). "
                  "Compile with plain --pixel or an ASCII mode.")
            decoder.release()
            return
        wm_key = (getattr(args, "watermark_key", None)
                  or os.environ.get("ASCILINE_WM_KEY"))
        if not wm_key:
            print("Error: --watermark requires --watermark-key "
                  "or env ASCILINE_WM_KEY")
            decoder.release()
            return
        try:
            watermarker = Watermarker(args.watermark, wm_key, rows, cols,
                                      block=getattr(args, "watermark_block", 1),
                                      beta=getattr(args, "watermark_beta", 8))
        except ValueError as e:
            print(f"Error: --watermark: {e}")
            decoder.release()
            return
        print(f"[Watermark] 10-digit forensic mark ON "
              f"(block={getattr(args, 'watermark_block', 1)}, "
              f"beta={max(1, min(64, getattr(args, 'watermark_beta', 8)))})")
    
    frame_buf = np.empty((rows, cols, 4), dtype=np.uint8) if render_mode > 1 else None

    with open(ascf_path, "wb") as f_out:
        # Write Header (18 bytes) — magic 'ASC2' identifies v2 format
        # Magic: 'ASC2' (4)  -- 'ASCF' = legacy 14-byte header, 'ASC2' = 18-byte with totalFrames
        # FPS: float32 (4)
        # Mode: uint8 (1)
        # Pixel: uint8 (1)
        # Cols: uint16 (2)
        # Rows: uint16 (2)
        # Total frames: uint32 (4)  -- written as 0, patched after compile
        header = struct.pack(">4sfBBHHI", b"ASC2", effective_fps, render_mode, int(pixel_mode), cols, rows, 0)
        f_out.write(header)
        
        frame_index = 0
        prev_frame = None
        bytes_written = 18
        
        try:
            while True:
                for _ in range(skip_n - 1):
                    if not decoder.grab():
                        break
                
                try:
                    gray_frame, bgr_frame = next(decoder)
                except StopIteration:
                    break

                if pixel_mode:
                    frame_px = np.ascontiguousarray(bgr_frame)
                    if pixel_qb > 0:
                        # in-place bit-drop mask == (x >> qb) << qb, one pass
                        frame_px &= np.uint8((0xFF << pixel_qb) & 0xFF)
                    if watermarker is not None:
                        watermarker.embed_pixels(frame_px, frame_index)
                    if profile_enc is not None:
                        msg, prev_frame = profile_enc.encode(frame_px)
                    else:
                        msg, prev_frame = encode_frame(
                            frame_px,
                            prev_frame, frame_index, level=level, tolerance=tolerance
                        )
                else:
                    # LUT lookup — identical mapping to the live server path.
                    indices = gray_index_lut[gray_frame]
                    if watermarker is not None and render_mode == 1:
                        # Monochrome text: ±1 glyph density-rank dither (the
                        # only carrier without colours).
                        watermarker.embed_indices(indices, frame_index,
                                                  mapper._n, gray_frame)

                    if render_mode == 1:
                        char_matrix = mapper._lut[indices]
                        lines = [''.join(r) for r in char_matrix]
                        payload = (f"{frame_index}\n" + '\n'.join(lines)).encode('utf-8')
                        msg = payload # For mode 1, we just pack the string as bytes
                    else:
                        frame_buf[:, :, 0] = char_byte_lut[indices]
                        if qb_mask is not None:
                            np.bitwise_and(bgr_frame[:, :, ::-1], qb_mask,
                                           out=frame_buf[:, :, 1:])
                        else:
                            frame_buf[:, :, 1:] = bgr_frame[:, :, ::-1]

                        # ±beta luminance dither on the cell colour (same as
                        # the live server path; glyphs untouched).
                        if watermarker is not None:
                            watermarker.embed_pixels(frame_buf[:, :, 1:],
                                                     frame_index, gray_frame)

                        msg, prev_frame = encode_frame(
                            frame_buf, prev_frame, frame_index, level=level, tolerance=tolerance
                        )
                
                # Write length prefix (uint32) + payload
                f_out.write(struct.pack(">I", len(msg)))
                f_out.write(msg)
                
                bytes_written += 4 + len(msg)
                frame_index += 1
                
                if frame_index % 50 == 0:
                    print(f"\r[Compiler] Compiled {frame_index} frames ({(bytes_written / 1024 / 1024):.2f} MB)...", end="")
        
        finally:
            decoder.release()

    # Patch total frame count into header (offset 14, uint32 big-endian)
    with open(ascf_path, "r+b") as f_patch:
        f_patch.seek(14)
        f_patch.write(struct.pack(">I", frame_index))

    print(f"\n[Compiler] Done! Total frames: {frame_index}. Output saved to {ascf_path} ({(bytes_written / 1024 / 1024):.2f} MB)")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    import logo
    print(logo.render_static())
    print("\033[1;37m" + "═"*55 + "\033[0m\n")

    try:
        __version__ = _pkg_version("asciline")
    except PackageNotFoundError:
        __version__ = "unknown"

    parser = argparse.ArgumentParser(
        usage="python compiler.py [options] <video>",
        description="\033[1;36mASCILINE Static Compiler\033[0m\n",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=35)
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    # ── Source ──
    src = parser.add_argument_group('\033[33mSource\033[0m')
    src.add_argument("video", help="Path to input video")

    # ── Render ──
    rnd = parser.add_argument_group('\033[33mRender\033[0m')
    rnd.add_argument("--cols", type=int, default=300, help="Grid columns (default 300)")
    rnd.add_argument("--rows", type=int, default=0, help="Grid rows (0 = auto)")
    rnd.add_argument("--mode", type=int, default=6, choices=[1, 2, 3, 4, 5, 6], help="Color quality: 1=B&W  2=64c  3=512c  4=32Kc  5=262Kc  6=16M Ultra")
    rnd.add_argument("--pixel", action="store_true", help="Pixel mode (no characters)")

    # ── Optimization & Export ──
    opt = parser.add_argument_group('\033[33mOptimization & Export\033[0m')
    opt.add_argument("--tolerance", type=int, default=0, help="Color drift tolerance (0=lossless)")
    opt.add_argument("--quantize", type=int, default=0, choices=[0, 1, 2, 3], metavar="0-3", help="Pixel mode color quantization (0=lossless, 3=aggressive)")
    opt.add_argument("--profile", action="store_true", help="Opt-in lossy DCT compression profile (implies --pixel)")
    opt.add_argument("--qf", type=int, default=70, help="Profile quality factor 1-100 (Default 70)")
    opt.add_argument("--hard", action="store_true", help="Use maximum zlib compression (level 9). Slower but smaller file.")
    opt.add_argument("--out", type=str, default="", help="Output base name")

    # ── Forensic watermark ──
    forensic = parser.add_argument_group('\033[33mForensic Watermark\033[0m')
    forensic.add_argument("--watermark", metavar="10-DIGITS", default=None,
        help="Invisibly burn this 10-digit ID into the compiled frames "
             "(screen-capture-proof forensic mark)")
    forensic.add_argument("--watermark-key", default=None,
        help="Secret key for the mark (or env ASCILINE_WM_KEY)")
    forensic.add_argument("--watermark-block", type=int, default=1, metavar="N",
        help="Frames per alternation clock (default 1)")
    forensic.add_argument("--watermark-beta", type=int, default=8, metavar="1-64",
        help="Pixel-mode luma dither amplitude (default 8)")

    args = parser.parse_args()
    
    # Automatically enable pixel mode and color mode 6 if profile is requested
    if args.profile:
        args.pixel = True
        
    if args.pixel and args.mode == 1:
        args.mode = 6

    compile_video(args)
