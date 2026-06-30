"""
network.py — Ultra-lightweight UDP goal event dispatcher.

Goal events are dispatched as small binary packets over UDP.
Protocol: [magic_2B][seq_2B][timestamp_4B][event_type_1B][reserved_7B]
Total: 16 bytes per packet.

Designed for RoboCop HSL GameController integration — the packet format
can be swapped by replacing the _build_packet() method.
"""

import socket
import struct
import time
import threading


# Event types
EVENT_GOAL = 0x01
EVENT_NO_BALL = 0x00
EVENT_HEARTBEAT = 0xFF

# Magic bytes for packet header
MAGIC = b"\xAA\xBB"

# Packet structure (16 bytes total):
#   magic:      2 bytes  (0xAA, 0xBB)
#   sequence:   2 bytes  (unsigned short, big-endian)
#   timestamp:  4 bytes  (unsigned int, seconds since epoch)
#   event_type: 1 byte
#   confidence: 1 byte   (0-255, scaled from 0.0-1.0)
#   reserved:   6 bytes  (zero-filled)
PACKET_FORMAT = "!BBHIBB6x"  # 2 + 2 + 4 + 1 + 1 + 6 = 16 bytes


class UdpDispatcher:
    """Lightweight UDP event dispatcher with sequence numbering."""

    def __init__(self, config):
        self.target_ip = config.udp_target_ip
        self.target_port = config.udp_target_port
        self.bind_port = config.udp_bind_port
        self.broadcast = config.udp_broadcast

        self._sock = None
        self._sequence = 0
        self._lock = threading.Lock()

    def open(self):
        """Create and bind UDP socket."""
        family = socket.AF_INET
        self._sock = socket.socket(family, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        if self.broadcast:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        if self.bind_port > 0:
            self._sock.bind(("0.0.0.0", self.bind_port))

        self._sock.settimeout(None)   # blocking send, no timeout needed
        return True

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def send_goal(self, confidence=1.0):
        """Send a GOAL event.

        Args:
            confidence: detection confidence (0.0-1.0), scaled to 0-255.
        """
        self._send(EVENT_GOAL, confidence)

    def send_no_ball(self):
        """Send NO_BALL status update."""
        self._send(EVENT_NO_BALL, 0)

    def send_heartbeat(self):
        """Send keepalive heartbeat."""
        self._send(EVENT_HEARTBEAT, 0)

    def _send(self, event_type, confidence=0.0):
        if self._sock is None:
            return

        with self._lock:
            seq = self._sequence
            self._sequence = (self._sequence + 1) % 65536

        ts = int(time.time())
        conf_byte = min(255, max(0, int(confidence * 255)))

        packet = struct.pack(
            PACKET_FORMAT,
            0xAA, 0xBB,      # magic
            seq,              # sequence
            ts,               # timestamp
            event_type,       # event type
            conf_byte,        # confidence (scaled)
        )

        try:
            self._sock.sendto(packet, (self.target_ip, self.target_port))
        except OSError:
            pass  # Silently drop if network unavailable

    def update_target(self, ip, port):
        """Update target endpoint at runtime."""
        self.target_ip = ip
        self.target_port = port

    @property
    def sequence(self):
        return self._sequence

    @property
    def is_open(self):
        return self._sock is not None
