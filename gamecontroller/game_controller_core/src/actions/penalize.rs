use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionContext, VAction};
use crate::actions::Unpenalize;
use crate::timer::{BehaviorAtZero, RunCondition, Timer};
use crate::types::{Penalty, PenaltyCall, Phase, PlayerNumber, Side, State};

/// This struct defines an action to apply a penalty to players.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Penalize {
    /// The side whose player is penalized.
    pub side: Side,
    /// The number of the player who is penalized.
    pub player: PlayerNumber,
    /// The penalty which has been called for the player.
    pub call: PenaltyCall,
}

impl Action for Penalize {
    fn execute(&self, c: &mut ActionContext) {
        // Map the penalty call to a penalty.
        let mut penalty = match self.call {
            PenaltyCall::IllegalPosition => Penalty::IllegalPositioning,
            PenaltyCall::MotionInSet => Penalty::MotionInSet,
            PenaltyCall::MotionInStop => Penalty::MotionInStop,
            PenaltyCall::LocalGameStuck => Penalty::LocalGameStuck,
            PenaltyCall::IncapableRobot => Penalty::IncapableRobot,
            PenaltyCall::RequestForPickUp => Penalty::PickedUp,
            PenaltyCall::BallHolding => Penalty::BallHolding,
            PenaltyCall::LeavingTheField => Penalty::LeavingTheField,
            PenaltyCall::PlayingWithArmsHands => Penalty::PlayingWithArmsHands,
            PenaltyCall::Pushing => Penalty::Pushing,
            PenaltyCall::Caution => Penalty::Cautioned,
            PenaltyCall::SendOff => Penalty::SentOff,
        };

        if self.call == PenaltyCall::Caution {
            c.game.teams[self.side][self.player].cautions += 1;
            if c.game.teams[self.side][self.player].cautions >= 2 {
                c.game.teams[self.side][self.player].cautions = 0;
                penalty = Penalty::SentOff;
            }
        }

        c.game.teams[self.side][self.player].penalty_duration = match (c.game.state, penalty) {
            (State::Initial | State::Timeout, Penalty::PickedUp) => Duration::ZERO,
            (_, Penalty::MotionInSet) => c.params.competition.penalties[penalty].duration,
            // The duration is composed of the base duration plus the increment for
            // each previous incremental penalty of this team.
            _ => {
                c.params.competition.penalties[penalty].duration
                    + c.params.competition.penalty_duration_increment
                        * c.game.teams[self.side].penalty_counter
            }
        };
        c.game.teams[self.side][self.player].penalty_timer = match penalty {
            Penalty::MotionInSet => Timer::Started {
                remaining: c.game.teams[self.side][self.player]
                    .penalty_duration
                    .try_into()
                    .unwrap(),
                run_condition: RunCondition::ReadyOrPlaying,
                // Motion in Set is removed automatically.
                behavior_at_zero: BehaviorAtZero::Expire(vec![VAction::Unpenalize(Unpenalize {
                    side: self.side,
                    player: self.player,
                    force: true,
                })]),
            },
            _ => Timer::Stopped,
        };
        c.game.teams[self.side][self.player].penalty = penalty;
        if c.params.competition.penalties[penalty].incremental {
            c.game.teams[self.side].penalty_counter += 1;
        }
    }

    fn is_legal(&self, c: &ActionContext) -> bool {
        // Most penalties can only be given to unpenalized players, except for yellow / red cards,
        // which can be given to any player that is not a substitute or already sent off.
        (c.game.teams[self.side][self.player].penalty == Penalty::NoPenalty
            || (!matches!(
                c.game.teams[self.side][self.player].penalty,
                Penalty::Substitute | Penalty::SentOff
            ) && matches!(self.call, PenaltyCall::Caution | PenaltyCall::SendOff)))
            && (match self.call {
                PenaltyCall::RequestForPickUp => true,
                PenaltyCall::IllegalPosition => {
                    c.game.phase != Phase::PenaltyShootout
                        && (c.game.state == State::Ready // Not possible in this state, but can
                                                         // happen if it happens shortly before a
                                                         // goal and the GameController presses the
                                                         // goal first.
                            || c.game.state == State::Set
                            || c.game.state == State::Playing)
                }
                PenaltyCall::MotionInSet => c.game.state == State::Set,
                PenaltyCall::MotionInStop => c.game.stopped,
                PenaltyCall::IncapableRobot => {
                    c.game.state == State::Ready
                        || c.game.state == State::Set
                        || c.game.state == State::Playing
                }
                PenaltyCall::LocalGameStuck => {
                    c.game.phase != Phase::PenaltyShootout && c.game.state == State::Playing
                }
                PenaltyCall::BallHolding => {
                    c.game.state == State::Ready // Not possible in this state, but can happen in
                                                 // Playing shortly before a goal and the
                                                 // GameController operator clicks the goal first.
                        || c.game.state == State::Playing
                }
                PenaltyCall::Pushing => {
                    // Not possible in Set, but can happen in Ready shortly before the timer
                    // expires.
                    (c.game.phase != Phase::PenaltyShootout
                        && (c.game.state == State::Ready || c.game.state == State::Set))
                        || c.game.state == State::Playing
                }
                PenaltyCall::PlayingWithArmsHands => {
                    c.game.state == State::Ready // Not possible in this state, but can happen in
                                                 // Playing shortly before a goal and the
                                                 // GameController operator clicks the goal first.
                        || c.game.state == State::Playing
                }
                PenaltyCall::LeavingTheField => {
                    // Not possible in Set, but can happen in Ready shortly before the timer
                    // expires.
                    (c.game.phase != Phase::PenaltyShootout
                        && (c.game.state == State::Ready || c.game.state == State::Set))
                        || c.game.state == State::Playing
                }
                PenaltyCall::Caution | PenaltyCall::SendOff => true,
            })
    }
}
