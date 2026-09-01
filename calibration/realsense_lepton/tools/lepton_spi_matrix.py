#!/usr/bin/env python3
"""Probe FLIR Lepton VoSPI headers across SPI devices, modes, and speeds.

Run on the Raspberry Pi. This checks whether any SPI configuration produces
valid Lepton packet headers before debugging the UDP sender.
"""

import argparse
import time


PACKET_SIZE = 164


def parse_header(packet):
    first = packet[0]
    second = packet[1]
    discard = (first & 0x0F) == 0x0F
    packet_number = second
    valid_packet = (not discard) and ((first & 0x0F) == 0) and (0 <= packet_number < 60)
    segment = ((first >> 4) & 0x0F) if packet_number == 20 else None
    return first, second, discard, packet_number, valid_packet, segment


def longest_sequential_run(packet_numbers):
    best = 0
    current = 0
    prev = None
    for value in packet_numbers:
        if prev is None or value == prev + 1:
            current += 1
        else:
            current = 1
        best = max(best, current)
        prev = value
    return best


def probe(spidev_mod, bus, device, mode, speed_hz, count, resync_delay_s):
    spi = spidev_mod.SpiDev()
    spi.open(bus, device)
    spi.mode = mode
    spi.bits_per_word = 8
    spi.max_speed_hz = speed_hz

    first_headers = []
    valid_numbers = []
    discards = 0
    invalid = 0
    segments = []

    try:
        for index in range(count):
            packet = bytes(spi.xfer2([0] * PACKET_SIZE))
            h0, h1, discard, pkt, valid, segment = parse_header(packet)
            if index < 8:
                first_headers.append(f"{h0:02x}{h1:02x}")
            if discard:
                discards += 1
                time.sleep(resync_delay_s)
            elif valid:
                valid_numbers.append(pkt)
                if segment is not None:
                    segments.append(segment)
            else:
                invalid += 1
    finally:
        spi.close()

    return {
        "dev": f"spidev{bus}.{device}",
        "mode": mode,
        "speed_mhz": speed_hz / 1_000_000,
        "valid": len(valid_numbers),
        "discards": discards,
        "invalid": invalid,
        "best_run": longest_sequential_run(valid_numbers),
        "first": ",".join(first_headers),
        "packets": sorted(set(valid_numbers))[:12],
        "segments": sorted(set(segments)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--bus", type=int, default=0)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--modes", default="0,1,2,3")
    parser.add_argument("--speeds-mhz", default="20,10,8,4,2,1")
    parser.add_argument("--resync-delay-ms", type=float, default=185.0)
    args = parser.parse_args()

    try:
        import spidev
    except ImportError:
        raise SystemExit("Missing python3 spidev. Install on the Pi: sudo apt install -y python3-spidev")

    devices = [int(x) for x in args.devices.split(",") if x]
    modes = [int(x) for x in args.modes.split(",") if x]
    speeds = [int(float(x) * 1_000_000) for x in args.speeds_mhz.split(",") if x]
    delay = args.resync_delay_ms / 1000.0

    print("Good sign: valid>0 and best_run grows toward 60. Bad sign: only first=0fff/87ff/0000.")
    for device in devices:
        for mode in modes:
            for speed in speeds:
                try:
                    r = probe(spidev, args.bus, device, mode, speed, args.count, delay)
                except OSError as exc:
                    print(f"spidev{args.bus}.{device} mode={mode} speed={speed/1_000_000:g}MHz ERROR {exc}")
                    continue
                print(
                    f"{r['dev']} mode={r['mode']} speed={r['speed_mhz']:g}MHz "
                    f"valid={r['valid']}/{args.count} discard={r['discards']} invalid={r['invalid']} "
                    f"best_run={r['best_run']} packets={r['packets']} segments={r['segments']} "
                    f"first={r['first']}"
                )


if __name__ == "__main__":
    main()
