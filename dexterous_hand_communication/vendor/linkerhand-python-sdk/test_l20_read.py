#!/usr/bin/env python3
"""Read-only Windows connectivity check for a right Linker Hand L20 Lite."""

import time

from LinkerHand.linker_hand_api import LinkerHandApi


def main() -> None:
    print("Connecting to the right L20 Lite on PCAN_USBBUS1...")
    hand = LinkerHandApi(
        hand_joint="L20",
        hand_type="right",
        can="PCAN_USBBUS1",
    )

    time.sleep(1.0)
    state = hand.get_state()

    print("Position:", state)
    print("Position count:", len(state) if state else 0)

    if state and len(state) == 20:
        print("PASS: received 20 position values.")
    else:
        print("FAIL: expected 20 position values; do not send motion commands.")


if __name__ == "__main__":
    main()
