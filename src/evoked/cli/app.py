import argparse
import sys
from ephyviewer import mkQApp
from ephyviewer import get_sources_from_neo_segment, compose_mainviewer_from_sources
from evoked.io import _load_segments


def main():
    parser = argparse.ArgumentParser(
        prog="evoked-app",
        description="Run the evoked ephyviewer app."
    )
    parser.add_argument("--filename", metavar="PATH", help="Path to data file")
    parser.add_argument("--epoch", nargs="+", metavar="tuple", type=float, help="Analysis epoch")
    parser.add_argument("--event-label", metavar="str", help="BIDS or Neo event label")
    parser.add_argument("--segment", metavar="int", type=int, default=0, help="Segment number")
    args = parser.parse_args()

    if args.filename is None:
        raise ValueError("--filename is required")

    app = mkQApp()

    segments = _load_segments(args.filename, tuple(args.epoch), args.event_label)

    seg = segments[args.segment]
    sources = get_sources_from_neo_segment(seg)

    sources = {
        "signal": sources["signal"],
        "event": sources["event"],
        "spike": [],
        "epoch": [],
    }

    win = compose_mainviewer_from_sources(sources)
    win.auto_scale()

    win.show()
    app.exec()


if __name__ == "__main__":
    main()
  
   