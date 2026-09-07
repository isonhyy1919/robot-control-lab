#!/usr/bin/env python3
"""Safely verify an L20 Lite that identifies as an L10 worm-gear device.

The script is read-only unless --move is supplied.  In move mode it changes
only one logical joint by a small negative delta, then reads the state back.
"""

import argparse
import time

from LinkerHand.linker_hand_api import LinkerHandApi


JOINT_NAMES = (
    "thumb",
    "thumb_rotation",
    "index",
    "middle",
    "ring",
    "pinky",
    "index_swing",
    "ring_swing",
    "pinky_swing",
    "thumb_swing",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read or perform one small L10-protocol movement."
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="enable one physical movement; without this flag the test only reads",
    )
    parser.add_argument("--joint", choices=JOINT_NAMES, default="index")
    parser.add_argument(
        "--delta",
        type=int,
        default=-4,
        help="small signed change from the current value (default: -4)",
    )
    parser.add_argument("--speed", type=int, default=30)
    return parser.parse_args()


def clamp_byte(value):
    return max(0, min(255, int(value)))


def read_state(hand, attempts=5):
    state = []
    for _ in range(attempts):
        state = list(hand.get_state())
        if len(state) == 10 and all(0 <= int(v) <= 255 for v in state):
            return [int(v) for v in state]
        time.sleep(0.1)
    raise RuntimeError(f"Expected 10 valid L10 joint values, got: {state}")


def main():
    args = parse_args()
    speed = clamp_byte(args.speed)

    hand = LinkerHandApi(
        hand_joint="L10",
        hand_type="right",
        can="PCAN_USBBUS1",
    )
    time.sleep(0.3)

    current = read_state(hand)
    print("Protocol: L10 worm-gear (device identity starts with LHT10)")
    print("Current :", current)

    if not args.move:
        print("Read-only check complete. Add --move to authorize one small movement.")
        return

    joint_index = JOINT_NAMES.index(args.joint)
    target = current.copy()
    target[joint_index] = clamp_byte(current[joint_index] + args.delta)
    if target[joint_index] == current[joint_index]:
        raise RuntimeError("Requested delta is clipped to the current value; nothing to do")

    hand.set_speed([speed] * 5)
    print(
        f"Moving {args.joint}: {current[joint_index]} -> "
        f"{target[joint_index]} at speed {speed}"
    )
    hand.finger_move(pose=target)
    time.sleep(0.8)

    after = read_state(hand)
    print("After   :", after)
    if after[joint_index] != target[joint_index]:
        raise RuntimeError(
            f"Verification failed: expected {target[joint_index]}, "
            f"read {after[joint_index]}"
        )
    print("PASS: commanded joint reached the requested value.")


if __name__ == "__main__":
    main()
