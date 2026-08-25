import argparse
import sys
from ephyviewer import mkQApp
from ephyviewer import get_sources_from_neo_segment, compose_mainviewer_from_sources
from evoked.io import load_segments


def main():
    parser = argparse.ArgumentParser(
        prog="evoked.app",
        description="Run the evoked ephyviewer app."
    )
    parser.add_argument("--filename", help="Path to data file")
    parser.add_argument("--epoch", type=tuple, help="Analysis epoch")
    parser.add_argument("--event-label", help="BIDS or Neo event label")
    parser.add_argument("--segment", type=int, default=0, help="Segment number")
    args = parser.parse_args()

    app = mkQApp()

    segments = load_segments(args.filename, args.epoch, args.event_label)

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
  
   