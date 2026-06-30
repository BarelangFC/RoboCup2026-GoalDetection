import { useEffect, useState, useCallback } from "react";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { applyAction } from "../api.js";

export default function GoalReviewOverlay() {
  const [event, setEvent] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);

  useEffect(() => {
    let unlisten = null;
    const setup = async () => {
      const win = getCurrentWebviewWindow();
      unlisten = await win.listen("goal-footage", (e) => {
        setEvent(e.payload);
        setCurrentFrame(0);
      });
    };
    setup();
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  const close = useCallback(() => {
    setEvent(null);
    setCurrentFrame(0);
  }, []);

  const confirmGoal = useCallback(() => {
    if (!event) return;
    const side = event.team_id === 1 ? "home" : "away";
    applyAction({ type: "goal", args: { side } });
    close();
  }, [event, close]);

  const revokeGoal = useCallback(() => {
    close();
  }, [close]);

  if (!event) return null;

  const frames = event.frames || [];
  const frameData = frames[currentFrame];

  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-lg border border-gray-300 p-4 max-w-lg w-11/12 text-center">
        <h2 className="text-lg font-bold text-black mb-1">Goal Review</h2>
        <p className="text-gray-600 text-xs mb-3">
          {event.team_id === 1 ? "HOME" : "AWAY"} — Event #{event.seq}
        </p>

        <div className="bg-gray-100 border border-gray-300 rounded overflow-hidden mb-2 aspect-[4/3] flex items-center justify-center">
          {frameData ? (
            <img
              src={`data:image/jpeg;base64,${btoa(String.fromCharCode(...frameData))}`}
              alt={`Frame ${currentFrame + 1}`}
              className="w-full h-full object-contain"
            />
          ) : (
            <span className="text-gray-400 text-sm">No frame data</span>
          )}
        </div>

        <div className="flex items-center justify-center gap-3 mb-3">
          <button
            className="bg-gray-200 text-black rounded px-3 py-1 text-sm disabled:opacity-30 border border-gray-300"
            disabled={currentFrame <= 0}
            onClick={() => setCurrentFrame((f) => Math.max(0, f - 1))}
          >◀</button>
          <span className="text-black text-sm min-w-[50px]">
            {currentFrame + 1} / {frames.length}
          </span>
          <button
            className="bg-gray-200 text-black rounded px-3 py-1 text-sm disabled:opacity-30 border border-gray-300"
            disabled={currentFrame >= frames.length - 1}
            onClick={() => setCurrentFrame((f) => Math.min(frames.length - 1, f + 1))}
          >▶</button>
        </div>

        <div className="flex gap-3 justify-center">
          <button
            className="bg-white text-black border-2 border-green-600 rounded px-6 py-2 text-sm font-bold hover:bg-green-50"
            onClick={confirmGoal}
          >CONFIRM</button>
          <button
            className="bg-white text-black border-2 border-red-500 rounded px-6 py-2 text-sm font-bold hover:bg-red-50"
            onClick={revokeGoal}
          >REVOKE</button>
        </div>
      </div>
    </div>
  );
}
