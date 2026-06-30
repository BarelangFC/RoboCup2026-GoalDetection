//! This crate contains network services for the GameController.

use std::net::IpAddr;

use bytes::Bytes;

mod action_command_receiver;
mod control_message_sender;
mod goal_footage_receiver;
mod monitor_request_receiver;
mod status_message_forwarder;
mod status_message_receiver;
mod team_message_receiver;
mod workaround;

pub use action_command_receiver::ActionCommandReceiver;
pub use control_message_sender::ControlMessageSender;
pub use goal_footage_receiver::GoalFootageReceiver;
pub use monitor_request_receiver::MonitorRequestReceiver;
pub use status_message_forwarder::StatusMessageForwarder;
pub use status_message_receiver::StatusMessageReceiver;
pub use team_message_receiver::TeamMessageReceiver;

/// This enumerates network events.
#[derive(Debug)]
pub enum Event {
    /// An incoming monitor request.
    MonitorRequest {
        host: IpAddr,
        data: Bytes,
        too_long: bool,
    },
    /// An incoming status message (from a player).
    StatusMessage {
        host: IpAddr,
        data: Bytes,
        too_long: bool,
    },
    /// An incoming team message (from a player).
    TeamMessage {
        host: IpAddr,
        team: u8,
        data: Bytes,
        too_long: bool,
    },
    /// An incoming goal footage event from the Jetson goal detector.
    GoalFootage {
        host: IpAddr,
        team_id: u8,
        seq: u16,
        timestamp: u32,
        frames: Vec<Vec<u8>>,
    },
}
