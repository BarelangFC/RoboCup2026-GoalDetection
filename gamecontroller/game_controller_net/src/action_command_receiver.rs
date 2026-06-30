use std::net::IpAddr;

use anyhow::Result;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use game_controller_core::{
    action::VAction,
    actions::{Goal, StopPlay},
    types::Side,
};

const COMMAND_PORT: u16 = 3738;

/// Listens for simple action commands on TCP port 3738.
/// Format: one of "goal:home", "goal:away", "stop", "resume"
pub struct ActionCommandReceiver {
    listener: TcpListener,
    action_sender: mpsc::UnboundedSender<VAction>,
}

impl ActionCommandReceiver {
    pub async fn new(address: IpAddr, action_sender: mpsc::UnboundedSender<VAction>) -> Result<Self> {
        let listener = TcpListener::bind((address, COMMAND_PORT)).await?;
        Ok(Self {
            listener,
            action_sender,
        })
    }

    pub async fn run(&self) -> Result<()> {
        loop {
            let (mut stream, peer) = self.listener.accept().await?;
            let sender = self.action_sender.clone();
            tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let mut buf = [0u8; 128];
                match stream.read(&mut buf).await {
                    Ok(n) if n > 0 => {
                        let text = String::from_utf8_lossy(&buf[..n]).trim().to_string();
                        let action = match text.as_str() {
                            "goal:home" => Some(VAction::Goal(Goal { side: Side::Home })),
                            "goal:away" => Some(VAction::Goal(Goal { side: Side::Away })),
                            "stop" => Some(VAction::StopPlay(StopPlay { resume: false })),
                            "resume" => Some(VAction::StopPlay(StopPlay { resume: true })),
                            _ => None,
                        };
                        if let Some(a) = action {
                            let _ = sender.send(a);
                            println!("[CMD] {} from {}", text.trim(), peer.ip());
                        } else {
                            eprintln!("[CMD] Unknown command: {}", text.trim());
                        }
                    }
                    _ => {}
                }
            });
        }
    }
}
