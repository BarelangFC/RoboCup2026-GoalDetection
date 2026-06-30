use std::{mem::replace, time::Duration};

use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionContext};
use crate::timer::Timer;
use crate::types::{Penalty, Phase, PlayerNumber, Side, State};

/// This struct defines an action to substitute players.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Substitute {
    /// The side which does the substitution.
    pub side: Side,
    /// The player who comes in (currently a substitute).
    pub player_in: PlayerNumber,
    /// The player who comes off (will become a substitute).
    pub player_out: PlayerNumber,
}

impl Action for Substitute {
    fn execute(&self, c: &mut ActionContext) {
        c.game.teams[self.side][self.player_in].penalty = replace(
            &mut c.game.teams[self.side][self.player_out].penalty,
            Penalty::Substitute,
        );
        // If the player substituted player wasn't penalized, it gets a picked up penalty.
        if c.game.teams[self.side][self.player_in].penalty == Penalty::NoPenalty {
            c.game.teams[self.side][self.player_in].penalty = Penalty::PickedUp;
        }
        c.game.teams[self.side][self.player_in].penalty_duration = replace(
            &mut c.game.teams[self.side][self.player_out].penalty_duration,
            Duration::ZERO,
        );
        c.game.teams[self.side][self.player_in].penalty_timer = Timer::Stopped;
        c.game.teams[self.side][self.player_out].penalty_timer = Timer::Stopped;
        if c.game.teams[self.side].goalkeeper == Some(self.player_out) {
            c.game.teams[self.side].goalkeeper = Some(self.player_in);
        }
    }

    fn is_legal(&self, c: &ActionContext) -> bool {
        c.game.phase != Phase::PenaltyShootout
            && (matches!(
                c.game.state,
                State::Initial | State::Set | State::Finished | State::Timeout
            ) || (c.game.stopped
                && c.game.teams[self.side][self.player_out].penalty != Penalty::NoPenalty))
            && self.player_in != self.player_out
            && c.game.teams[self.side][self.player_in].penalty == Penalty::Substitute
            && !matches!(
                c.game.teams[self.side][self.player_out].penalty,
                Penalty::SentOff | Penalty::Substitute
            )
    }
}
