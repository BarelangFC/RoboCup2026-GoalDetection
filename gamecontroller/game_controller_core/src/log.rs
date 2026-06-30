use std::{net::IpAddr, time::Duration};

use serde::{Deserialize, Serialize};
use serde_with::{base64::Base64, serde_as};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

use crate::action::VAction;
use crate::types::{ActionSource, Game, Params};

#[serde_as]
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoggedMetadata {
    pub creator: String,
    pub version: u32,
    #[serde_as(as = "Rfc3339")]
    pub timestamp: OffsetDateTime,
    pub params: Box<Params>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoggedAction {
    pub source: ActionSource,
    pub action: VAction,
}

pub type LoggedGameState = Box<Game>;

#[serde_as]
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoggedMonitorRequest {
    pub host: IpAddr,
    #[serde_as(as = "Base64")]
    pub data: Vec<u8>,
}

#[serde_as]
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoggedStatusMessage {
    pub host: IpAddr,
    #[serde_as(as = "Base64")]
    pub data: Vec<u8>,
}

#[serde_as]
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoggedTeamMessage {
    pub team: u8,
    pub host: IpAddr,
    #[serde_as(as = "Base64")]
    pub data: Vec<u8>,
}

#[serde_as]
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoggedGoalFootage {
    pub host: IpAddr,
    pub team_id: u8,
    pub seq: u32,
    pub timestamp: u32,
    pub frame_count: u8,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum LogEntry {
    Metadata(LoggedMetadata),
    Action(LoggedAction),
    GameState(LoggedGameState),
    MonitorRequest(LoggedMonitorRequest),
    StatusMessage(LoggedStatusMessage),
    TeamMessage(LoggedTeamMessage),
    GoalFootage(LoggedGoalFootage),
    End,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TimestampedLogEntry {
    pub timestamp: Duration,
    pub entry: LogEntry,
}

pub trait Logger {
    fn append(&mut self, entry: TimestampedLogEntry);
}

pub struct NullLogger;

impl Logger for NullLogger {
    fn append(&mut self, _entry: TimestampedLogEntry) {}
}
