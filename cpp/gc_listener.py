"""
gc_listener.py — Lightweight Goal Review popup for Windows/WSL.
Listens for goal footage from Jetson, shows review popup,
sends commands to GameController action port.
"""

import socket, struct, threading, os, sys, io, json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

PORT = 3736
GC_CMD_HOST = "192.168.220.35"
GC_CMD_PORT = 3738
WINDOW_TITLE = "Goal Review (Test)"

class GoalReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("600x540")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)

        self.frames = []
        self.current_frame = 0
        self.team_id = 0
        self.seq = 0
        self.team_side = "home"
        self.team_label = "HOME"

        self._build_ui()
        self._update_frame_display()
        self._start_listener()

    def _build_ui(self):
        bg = "#f0f0f0"
        fg = "#222222"
        btn_bg = "#e0e0e0"
        self.root.configure(bg=bg)

        self.title_label = tk.Label(self.root, text="WAITING FOR GOAL...",
                         font=("Arial", 14, "bold"),
                         fg=fg, bg=bg)
        self.title_label.pack(pady=(10, 0))

        self.info_label = tk.Label(self.root, text="",
                                   font=("Arial", 9),
                                   fg="#666666", bg=bg)
        self.info_label.pack()

        self.frame_label = tk.Label(self.root, bg="#ffffff", width=80, height=45,
                                    relief="solid", bd=1)
        self.frame_label.pack(pady=8, padx=10, fill=tk.BOTH, expand=True)

        nav_frame = tk.Frame(self.root, bg=bg)
        nav_frame.pack(pady=4)
        self.prev_btn = tk.Button(nav_frame, text="◀", command=self._prev_frame,
                                  font=("Arial", 12), bg=btn_bg, fg=fg,
                                  relief="flat", state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=4)
        self.frame_counter = tk.Label(nav_frame, text="0 / 0",
                                      font=("Arial", 10),
                                      fg=fg, bg=bg)
        self.frame_counter.pack(side=tk.LEFT, padx=8)
        self.next_btn = tk.Button(nav_frame, text="▶", command=self._next_frame,
                                  font=("Arial", 12), bg=btn_bg, fg=fg,
                                  relief="flat", state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=4)

        team_frame = tk.Frame(self.root, bg=bg)
        team_frame.pack(pady=4)
        tk.Label(team_frame, text="Scoring:",
                 font=("Arial", 9), fg=fg, bg=bg).pack(side=tk.LEFT, padx=4)
        self.team_btn = tk.Button(team_frame, text=self.team_label,
                                  command=self._toggle_team,
                                  font=("Arial", 9, "bold"),
                                  bg=btn_bg, fg=fg, width=6, relief="flat")
        self.team_btn.pack(side=tk.LEFT, padx=4)

        action_frame = tk.Frame(self.root, bg=bg)
        action_frame.pack(pady=8)
        self.confirm_btn = tk.Button(action_frame, text="CONFIRM GOAL",
                                     command=self._confirm_goal,
                                     font=("Arial", 11, "bold"),
                                     bg="#ffffff", fg=fg, width=14, height=1,
                                     relief="solid", bd=1,
                                     state=tk.DISABLED)
        self.confirm_btn.pack(side=tk.LEFT, padx=8)
        self.revoke_btn = tk.Button(action_frame, text="REVOKE",
                                    command=self._revoke_goal,
                                    font=("Arial", 11, "bold"),
                                    bg="#ffffff", fg=fg, width=12, height=1,
                                    relief="solid", bd=1,
                                    state=tk.DISABLED)
        self.revoke_btn.pack(side=tk.LEFT, padx=8)

        self.log_text = tk.Text(self.root, height=3, bg="#ffffff",
                                fg="#666666", font=("Courier", 8),
                                relief="solid", bd=1)
        self.log_text.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _toggle_team(self):
        self.team_side = "away" if self.team_side == "home" else "home"
        self.team_label = "HOME" if self.team_side == "home" else "AWAY"
        self.team_btn.config(text=self.team_label)
        self._log(f"[UI] Team toggled to {self.team_label}")

    def _prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self._update_frame_display()

    def _next_frame(self):
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            self._update_frame_display()

    def _update_frame_display(self):
        if not self.frames:
            self.frame_label.config(image="", text="NO FRAMES", fg="#555")
            self.frame_counter.config(text="0 / 0")
            return
        total = len(self.frames)
        idx = min(self.current_frame, total - 1)
        self.frame_counter.config(text=f"{idx + 1} / {total}")
        self.prev_btn.config(state=tk.NORMAL if idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if idx < total - 1 else tk.DISABLED)
        jpg = self.frames[idx]
        img = Image.open(io.BytesIO(jpg))
        img.thumbnail((560, 380), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.frame_label.config(image=photo, text="")
        self.frame_label.image = photo

    def _send_gc(self, cmd):
        try:
            c = socket.socket()
            c.settimeout(3)
            c.connect((GC_CMD_HOST, GC_CMD_PORT))
            c.send(cmd.encode())
            c.close()
            self._log(f"[CMD] {cmd}")
        except Exception as e:
            self._log(f"[CMD] Failed: {e}")

    def _on_goal_received(self, team_id, seq, frames):
        self.team_id = team_id
        self.seq = seq
        self.frames = frames
        self.current_frame = 0
        self.title_label.config(text=f"GOAL REVIEW — Event #{seq}")
        self.info_label.config(text=f"team={team_id} seq={seq} frames={len(frames)}")
        self.root.title(f"{WINDOW_TITLE} — Event #{seq}")
        side = "home" if team_id == 1 else "away"
        self.team_side = side
        self.team_label = "HOME" if team_id == 1 else "AWAY"
        self.team_btn.config(text=self.team_label)
        self.confirm_btn.config(state=tk.NORMAL)
        self.revoke_btn.config(state=tk.NORMAL)
        self._update_frame_display()
        self._log(f"[GOAL] team={team_id} seq={seq}")
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _confirm_goal(self):
        side = self.team_side
        self._log(f"[ACTION] CONFIRMED {side}")
        self._send_gc(f"goal:{side}")
        self._reset()

    def _revoke_goal(self):
        self._log("[ACTION] REVOKED")
        self._reset()

    def _reset(self):
        self.frames = []
        self.current_frame = 0
        self.confirm_btn.config(state=tk.DISABLED)
        self.revoke_btn.config(state=tk.DISABLED)
        self.title_label.config(text="WAITING FOR GOAL...")
        self.info_label.config(text="")
        self.root.title(WINDOW_TITLE)
        self._update_frame_display()

    def _start_listener(self):
        self._log(f"[NET] Listening on port {PORT}...")

        def listen():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", PORT))
            s.listen(5)
            s.settimeout(1.0)
            while True:
                try:
                    conn, addr = s.accept()
                    header = conn.recv(8, socket.MSG_WAITALL)
                    if len(header) < 8:
                        conn.close(); continue
                    team_id = header[0]
                    seq = struct.unpack("<H", header[1:3])[0]
                    num_frames = header[7]
                    frames = []
                    for i in range(num_frames):
                        sb = conn.recv(4, socket.MSG_WAITALL)
                        if len(sb) < 4: break
                        sz = struct.unpack("<I", sb)[0]
                        frames.append(conn.recv(sz, socket.MSG_WAITALL))
                    conn.close()
                    if frames:
                        self.root.after(0, self._on_goal_received, team_id, seq, frames)
                except socket.timeout:
                    continue
                except Exception as e:
                    self._log(f"[NET] {e}")

        t = threading.Thread(target=listen, daemon=True)
        t.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = GoalReviewApp(root)
    root.mainloop()
