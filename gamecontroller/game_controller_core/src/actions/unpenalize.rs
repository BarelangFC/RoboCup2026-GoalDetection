use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionContext};
use crate::timer::{BehaviorAtZero, RunCondition, Timer};
use crate::types::{Penalty, PlayerNumber, Side};

/// This struct defines an action to unpenalize players.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Unpenalize {
    /// The side whose player is unpenalized.
    pub side: Side,
    /// The number of the player who is unpenalized.
    pub player: PlayerNumber,
    /// Whether the removal of a penalty should be allowed before its timer is over.
    pub force: bool,
}

impl Action for Unpenalize {
    fn execute(&self, c: &mut ActionContext) {
        if self.force
            || matches!(
                c.game.teams[self.side][self.player].penalty_timer,
                Timer::Started { .. }
            )
            || c.game.teams[self.side][self.player]
                .penalty_duration
                .is_zero()
        {
            if self.force
                || c.game.teams[self.side][self.player]
                    .penalty_timer
                    .get_remaining()
                    .is_zero()
                || c.game.teams[self.side][self.player]
                    .penalty_duration
                    .is_zero()
            {
                c.game.teams[self.side][self.player].penalty_duration = Duration::ZERO;
                c.game.teams[self.side][self.player].penalty = Penalty::NoPenalty;
            }
            c.game.teams[self.side][self.player].penalty_timer = Timer::Stopped;
        } else {
            c.game.teams[self.side][self.player].penalty_timer = Timer::Started {
                remaining: ({
                    c.game.teams[self.side][self.player]
                        .penalty_duration
                        .try_into()
                        .unwrap()
                }),
                run_condition: RunCondition::ReadyOrPlaying,
                behavior_at_zero: BehaviorAtZero::Clip,
            };
        }
    }

    fn is_legal(&self, c: &ActionContext) -> bool {
        !matches!(
            c.game.teams[self.side][self.player].penalty,
            Penalty::NoPenalty | Penalty::SentOff | Penalty::Substitute
        )
    }
}
