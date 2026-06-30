use std::net::IpAddr;

use anyhow::Result;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::Event;

const GOAL_FOOTAGE_PORT: u16 = 3737;

/// Maximum per-frame size: 100 KB (640×480 JPEG at quality 70)
const MAX_FRAME_SIZE: usize = 102_400;
/// Maximum total payload for 5 frames
const MAX_TOTAL: usize = 600_000;

/// Listens for TCP connections carrying goal event + 5 JPEG frames from the Jetson.
/// Each connection sends one event with this format:
///   [team_id:1][seq:2][timestamp:4][num_frames:1]
///   then for each frame: [jpeg_size:4][jpeg_data...]
pub struct GoalFootageReceiver {
    listener: TcpListener,
    event_sender: mpsc::UnboundedSender<Event>,
}

impl GoalFootageReceiver {
    pub async fn new(address: IpAddr, event_sender: mpsc::UnboundedSender<Event>) -> Result<Self> {
        let listener = TcpListener::bind((address, GOAL_FOOTAGE_PORT)).await?;
        Ok(Self {
            listener,
            event_sender,
        })
    }

    pub async fn run(&self) -> Result<()> {
        loop {
            let (mut stream, peer) = self.listener.accept().await?;
            let sender = self.event_sender.clone();
            tokio::spawn(async move {
                use tokio::io::AsyncReadExt;

                // Read header: team_id(1) + seq(2) + timestamp(4) + num_frames(1) = 8 bytes
                let mut header = [0u8; 8];
                if stream.read_exact(&mut header).await.is_err() {
                    return;
                }
                let team_id = header[0];
                let seq = u16::from_le_bytes([header[1], header[2]]);
                let timestamp = u32::from_le_bytes([header[3], header[4], header[5], header[6]]);
                let num_frames = header[7] as usize;
                if num_frames == 0 || num_frames > 10 {
                    return;
                }

                let mut frames = Vec::with_capacity(num_frames);
                let mut ok = true;
                for _ in 0..num_frames {
                    let mut size_buf = [0u8; 4];
                    if stream.read_exact(&mut size_buf).await.is_err() {
                        ok = false;
                        break;
                    }
                    let size = u32::from_le_bytes(size_buf) as usize;
                    if size == 0 || size > MAX_FRAME_SIZE {
                        ok = false;
                        break;
                    }
                    let mut buf = vec![0u8; size];
                    if stream.read_exact(&mut buf).await.is_err() {
                        ok = false;
                        break;
                    }
                    frames.push(buf);
                }
                if !ok {
                    return;
                }

                let _ = sender.send(Event::GoalFootage {
                    host: peer.ip(),
                    team_id,
                    seq,
                    timestamp,
                    frames,
                });
            });
        }
    }
}
