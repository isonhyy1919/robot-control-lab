#!/usr/bin/env python3
"""One-shot, low-speed, four-count motion test for a right L20 Lite."""

import time

from LinkerHand.linker_hand_api import LinkerHandApi


def main() -> None:
    hand = LinkerHandApi(
        hand_joint="L20",
        hand_type="right",
        can="PCAN_USBBUS1",
    )

    time.sleep(1.0)
    before = hand.get_state()
    print("Before:", before, flush=True)

    if not before or len(before) != 20:
        raise RuntimeError("Expected exactly 20 position values; no motion sent.")

    if not all(isinstance(value, (int, float)) and 0 <= value <= 255 for value in before):
        raise RuntimeError("Invalid position data; no motion sent.")

    target = list(before)
    target[0] = max(0, int(before[0]) - 4)

    print(f"Planned joint 1 change: {before[0]} -> {target[0]}", flush=True)
    print("Target:", target, flush=True)

    hand.set_speed(speed=[30, 30, 30, 30, 30])
    time.sleep(0.1)
    hand.finger_move(pose=target)

    time.sleep(2.0)
    after = hand.get_state()
    print("After:", after, flush=True)
    if after and len(after) == 20:
        print(f"Observed joint 1 change: {before[0]} -> {after[0]}", flush=True)


if __name__ == "__main__":
    main()
