use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionContext};
use crate::types::{Phase, PlayerNumber, Side, State};

/// This struct defines an action to select the goalkeeper.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectGoalkeeper {
    /// The side which selects the goalkeeper.
    pub side: Side,
    /// The player who becomes goalkeeper.
    pub player: PlayerNumber,
}

impl Action for SelectGoalkeeper {
    fn execute(&self, c: &mut ActionContext) {
        c.game.teams[self.side].goalkeeper = Some(self.player);
    }

    fn is_legal(&self, c: &ActionContext) -> bool {
        c.game.phase != Phase::PenaltyShootout
            && matches!(
                c.game.state,
                State::Initial | State::Set | State::Finished | State::Timeout
            )
    }
}
