"""
CLI Demo: Run speaker diarization on a local audio file.
Usage:
    python demo.py --audio path/to/audio.wav
    python demo.py --audio path/to/audio.wav --speakers 3
    python demo.py --audio path/to/audio.wav --output result.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization CLI")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--speakers", type=int, default=None)
    parser.add_argument("--output", default=None, help="Save JSON result")
    parser.add_argument("--rttm", default=None, help="Save RTTM output")
    parser.add_argument("--srt", default=None, help="Save SRT subtitle file")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"🎙  Speaker Diarization Pipeline")
    print(f"   Audio  : {audio_path}")
    print(f"   Speakers: {'auto-detect' if args.speakers is None else args.speakers}")
    print()

    from app.pipeline import DiarizationPipeline
    from utils.audio import segments_to_rttm, segments_to_srt

    pipeline = DiarizationPipeline(device=args.device, num_speakers=args.speakers)

    print("⏳ Running diarization...")
    result = pipeline.process(audio_path, num_speakers=args.speakers)

    print(f"\n✅ Done in {result.processing_time:.2f}s")
    print(f"   Speakers : {result.num_speakers}")
    print(f"   Duration : {result.audio_duration:.2f}s")
    print(f"   Segments : {len(result.segments)}")
    print()
    print(f"{'START':>8}  {'END':>8}  {'DUR':>6}  SPEAKER")
    print("─" * 42)
    for seg in result.segments:
        print(f"{seg.start:8.3f}  {seg.end:8.3f}  {seg.duration:6.3f}  {seg.speaker}")

    if args.output:
        Path(args.output).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\n💾 JSON saved to: {args.output}")
    if args.rttm:
        Path(args.rttm).write_text(segments_to_rttm(result.segments, audio_path.stem))
        print(f"💾 RTTM saved to: {args.rttm}")
    if args.srt:
        Path(args.srt).write_text(segments_to_srt(result.segments))
        print(f"💾 SRT saved to:  {args.srt}")


if __name__ == "__main__":
    main()
