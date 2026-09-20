# Visualisation: trajectory overlay videos

`lightnav.viz` renders what the model predicted on top of the frame it saw: the waypoint
chunk as a ground-plane ribbon, the pointing tokens as pixel markers, and a small
telemetry HUD. The same renderer serves both consumers of the engine:

* **Real robots** — `lightnav-serve --record_dir DIR` records every connection's episodes
  (the client's JPEG frames + one JSON record per prediction); `lightnav-render DIR`
  turns them into mp4s afterwards. The server never renders online.
* **Habitat evaluation** — `lightnav-eval-habitat --save_video` writes one mp4 per episode
  while it runs, and `--record_dir` optionally keeps the raw episodes in the same layout.

Rendering needs the `video` extra (`pip install -e ".[video]"`: OpenCV, imageio,
imageio-ffmpeg). Recording needs only the core install (numpy + Pillow).

## 1. What the overlay shows

Layout of a rendered frame: header bar with the instruction, a `GO` / `STOP` pill, the
step counter and the step rate; a blue corridor on the floor for the predicted path; mint
and magenta discs for the pointing channels; a `VX` / `VY` / `VYAW` readout bottom-left.

### Trajectory ribbon

The predicted `(H, 3)` chunk `[forward_m, lateral_m, yaw_rad]` — row `k` is the robot-local
ground-plane pose `k` steps ahead — is projected onto the floor plane through a pinhole camera model
(horizontal FOV + camera height, see §7) and drawn as a corridor. Its width at the bottom
edge of the frame is `--traj-width` of the frame width (default 0.25) and shrinks with
distance; the colour runs azure (near) to ice (far) and fades at the end of the chunk.
The robot's own pose is prepended, so the corridor starts under the camera.

**Forward offset caveat.** With the default `--forward-offset auto` the corridor is
displaced outward by the *bottom-edge depth* — the ground distance at which the camera
sees the bottom edge of the image — so the near waypoints, which a forward-facing camera
cannot see, stay visible. The shape of the path is exact; its placement is shifted by
that depth (about 0.6 m for a 0.5 m camera with a 112° lens on a 16:9 frame). Pass
`--forward-offset 0` for exact placement (the first waypoints then fall below the frame).

The ribbon needs at least two waypoint rows; after a failed decode there is no chunk and
no ribbon, but the HUD still shows the raw model text's consequences (`STOP` pill if the
fallback is a stop, the instruction, the step).

### Pointing discs

Checkpoints that emit grounding tokens get haloed discs at the decoded pixels:

| colour | channel | meaning |
|---|---|---|
| mint | `apos` | where the agent should go (goal / next position) |
| magenta (drawn last) | `opos` | where the target object is |

Only *pixel* channels are drawn — a channel whose state is `point` in the
`pointing` payload ([PROTOCOL.md](PROTOCOL.md)). The directive states (`rot_left`,
`rot_right`, `stop`) and `not_visible` have no pixel and are not drawn. The HUD's
`GO` / `STOP` pill shows the **decoded action** (an all-zero chunk / `<traj_0>`), which is
a different thing from an `apos` stop directive. Pixels are rescaled from the payload's
`frame_size` to the rendered frame, so a `--height` upscale keeps them in place.

### HUD

| field | source |
|---|---|
| instruction | the record's `instruction`, normalised: whitespace collapsed, first letter capitalised, a period added when there is no terminal punctuation |
| `GO` / `STOP` | the record's `stop` (decoded action) |
| `STEP nnnn` | the record's `step` (server: frames in the history buffer; eval: policy step) |
| rate `x.xx Hz` | `1000 / step_dt_ms` of the record (completion-to-completion of the previous interval); `--.-- Hz` when unknown (first step, or eval videos, which have no wall-clock timing) |
| `VX` / `VY` / `VYAW` | the **first waypoint** divided by `dt`: `forward_m / dt` (m/s), `lateral_m / dt` (m/s), `yaw_rad / dt` (rad/s); the bars have a fixed full scale (3.0 m/s, 0.5 m/s, 3.5 rad/s) so they compare across frames |

`dt` is a display convention, not a property of the model: waypoint rows are per-step
displacements and carry no time base of their own (the trajectory vocabularies cap a step
at ~0.25 m / 30°). The default `dt = 0.1 s` (`--dt` in `lightnav-render`,
`--waypoint_dt_s` in the server and the eval client, stored in the manifest) makes the
readout comparable across recordings; it does not have to match your control period.

## 2. Recording layout and record schema

Both the server and the eval client write, per run:

```
<record_dir>/
  run_<YYYYmmdd_HHMMSS>/
    <connection label>/                # server: clientId if it is a safe name, else conn001, ...
                                       # eval:   eval
      episode_000/
        manifest.json                  # run parameters (below)
        image_000002.jpg               # one frame per recorded step, named by `step`
        image_000003.jpg
        actions.jsonl                  # appended + flushed per step WHILE the episode is open
        actions.json                   # the same records as one JSON array, written when it ends
        traj_pointing.mp4              # added by lightnav-render
      episode_001/
        ...
```

`actions.jsonl` exists only while the episode is open (it survives a crash and
`lightnav-render` accepts it); `end_episode` writes `actions.json` atomically and removes
the jsonl. Frames are the client's JPEG bytes verbatim on the server (never re-encoded),
and JPEG-encoded (quality 95) `obs["rgb"]` in the eval client.

`manifest.json`:

```json
{"schema": 1, "created_at": "2026-…", "conn": "robot-01", "episode": 0,
 "task": "tracking", "model_path": "/path/to/hf_ckpt",
 "video_fps": 10, "video_timeline": "realtime", "waypoint_dt_s": 0.1,
 "overlay_hfov_deg": 90.0, "overlay_cam_height": 0.5, "overlay_forward_offset": null,
 "frame_size": [480, 270], "instruction": "follow the man in the black shirt", "extra": {}}
```

One record (a line of `actions.jsonl`, an element of `actions.json`):

| key | meaning |
|---|---|
| `step` | server: frames in the history buffer when the prediction was made (`actions.step` on the wire); eval: the policy step |
| `seq` | server: the client's `seq`; eval: equals `step` |
| `received_at` | ISO timestamp (ms) |
| `step_dt_ms` | completion-to-completion duration of the previous interval, `0.0` on the first step |
| `step_fps` | `1000 / step_dt_ms`, or `null` |
| `instruction` | the instruction of that request |
| `waypoints` | the `(H, 3)` chunk as a list of `[forward_m, lateral_m, yaw_rad]`, or `null` after a failed decode |
| `stop` | decoded stop (all-zero chunk) |
| `visible` | tracking visibility flag, or `null` |
| `raw_text` | the model's token text |
| `latency_ms` | server: end-to-end predict time of that request; eval: `policy.act` wall time |
| `pointing` | the `pointing` payload the client received (server) / would have received (eval), or `null` |
| `frame_size` | `[width, height]` of the recorded frame |

The eval client adds `episode_id`, `habitat_episode_id` and `scene_id` to every record.

## 3. Timebase: `realtime` vs `per_step`

The recording carries one frame per prediction, and `lightnav-render` decides how long
each one stays on screen:

* **`realtime`** (server default) — a step is repeated `round(step_dt_ms * fps / 1000)`
  times, clamped to `[1, 20]` frames, so the video plays at the pace the robot ran and a
  stall reads as a stall (a 240 s network hiccup still costs at most 2 s of video at 10 fps).
* **`per_step`** (eval default) — exactly one frame per step; the video length is the step
  count over `fps`. Simulator steps have no meaningful wall-clock spacing.

The manifest's `video_timeline` and `video_fps` are the defaults; `--timeline` and `--fps`
override them per render.

## 4. `lightnav-render`

```bash
lightnav-render output/episodes                 # every episode under the tree
lightnav-render output/episodes/run_*/robot-01/episode_003 --fps 15 --height 1080 --overwrite
lightnav-render output/episodes --timeline per_step --forward-offset 0 --no-hud
```

| flag | default | meaning |
|---|---|---|
| `paths` | required | episode directories, or trees containing them (`actions.json` or `actions.jsonl`) |
| `--out-name` | `traj_pointing.mp4` | output file name inside each episode directory |
| `--fps` | manifest `video_fps` (10) | output frame rate |
| `--timeline {realtime,per_step}` | manifest `video_timeline` | see §3 |
| `--dt` | manifest `waypoint_dt_s` (0.1) | seconds per waypoint row for the HUD velocity readout |
| `--traj-width` | `0.25` | ribbon width at the bottom edge, fraction of the frame width |
| `--min-steps` | `0` | skip episodes with fewer records |
| `--height` | `0` (keep) | upscale every frame to this height before drawing (e.g. `1080` for a 270 px recording) |
| `--forward-offset` | `auto` | `auto` = bottom-edge depth (near waypoints visible); a number in metres is added to the manifest's `overlay_forward_offset`; `0` = exact placement |
| `--no-pointing` / `--no-hud` | off | drop the discs / the telemetry overlay |
| `--overwrite` | off | replace an existing output (otherwise the episode is skipped and counted as done) |

The output is H.264 / yuv420p (odd frame sizes get one replicated edge row/column),
written under a temporary name and renamed when complete. One summary line is printed per
episode; records whose image is missing are skipped and listed. Exit code 0 iff every
episode rendered. Python: `lightnav.viz.render_episode_dir(episode_dir, ...)` with the
same keywords, `lightnav.viz.render_frame(rgb, waypoints=..., pointing=..., ...)` for a
single frame.

## 5. Server: `lightnav-serve --record_dir`

```bash
RECORD_DIR=output/episodes CAM_HFOV_DEG=112 CAM_HEIGHT=0.45 lightnav-serve --task tracking ...
# or
lightnav-serve ... --record_dir output/episodes --cam_hfov_deg 112 --cam_height 0.45
```

| flag | env | default | meaning |
|---|---|---|---|
| `--record_dir` | `RECORD_DIR` | `""` (off) | root of the recording tree; one `run_<timestamp>/` per server start |
| `--record_fps` | `RECORD_FPS` | `10` | `video_fps` written to the manifest |
| `--record_timeline` | `RECORD_TIMELINE` | `realtime` | default timebase written to the manifest |
| `--record_images` / `--no_record_images` | `RECORD_IMAGES` (`1`/`0`) | on | store the frames; without them the records are still written but no video can be rendered |
| `--cam_hfov_deg` | `CAM_HFOV_DEG` | `90.0` | horizontal FOV of the **client's** camera (ribbon projection only) |
| `--cam_height` | `CAM_HEIGHT` | `0.5` | height of the client's camera above the floor, metres |
| `--traj_forward_offset` | `TRAJ_FORWARD_OFFSET` | unset (auto) | fixed forward displacement of the ribbon, metres |
| `--waypoint_dt_s` | `WAYPOINT_DT_S` | `0.1` | HUD velocity readout convention |

Behaviour: one connection recorder per WebSocket connection, labelled by the `clientId`
sent at `login` when it is a plain name (`[A-Za-z0-9._-]`), else `conn001`, `conn002`,
…. `reset` starts a new episode; the first prediction of a connection starts one if none
is open. Every **predicted** `next` (not the buffer-only acknowledgement) adds a record
with the request's JPEG bytes and the `pointing` payload that went on the wire. The
connection's episode ends when the socket closes; the server's shutdown closes everything.

Recording is diagnostics: it runs after the reply has been sent, never changes a response
or its order, and any per-step recorder failure (full disk, …) is logged and dropped; an
unusable `--record_dir` is rejected at start-up, before the model loads. It costs one JPEG write and one JSON line per prediction; rendering is deliberately
left to `lightnav-render` so the serving loop never pays for video encoding.

`scripts/start_servers.sh` and `docker/entrypoint.sh` read the same environment
variables. `start_servers.sh` gives each server `RECORD_DIR/port<PORT>/` so several
servers started in the same second never share a run directory; in Docker, mount
`RECORD_DIR` as a volume so the recordings outlive the container.

## 6. Habitat evaluation: `lightnav-eval-habitat --save_video`

```bash
lightnav-eval-habitat --model_path /path/to/hf_ckpt --server tcp://localhost:5555 \
    --episodes 20 --output_dir output/r2r_viz --save_video
```

| flag | default | meaning |
|---|---|---|
| `--save_video` | off | write `<output_dir>/videos/<episode_id>.mp4` per episode, one frame per policy step (`per_step` timebase) plus the terminal observation |
| `--video_fps` | `10` | playback frame rate of those videos |
| `--hfov_deg` | `120.0` | horizontal FOV of the agent camera (the shipped `habitat_server/configs/*.yaml`) |
| `--cam_height` | `0.88` | camera height in metres (same yamls) |
| `--waypoint_dt_s` | `0.1` | HUD velocity readout convention |
| `--record_dir` | `""` (off) | also record the raw episodes (JPEG frames + records, §2) under `<record_dir>/run_*/eval/episode_NNN/` for `lightnav-render` |

Every frame is rendered from the observation the policy acted on, with that step's
prediction, *before* the environment advances; the observation returned by the last step
is appended with the last prediction so the terminal view is visible. The writer opens on
the first frame and the file is renamed into place when the episode ends, so an
interrupted run leaves no half-written `episode_*.mp4`. Each episode's result record
(`results.jsonl`) gains `"video": "videos/episode_000.mp4"` when a video was written.

`--save_video` requires the `video` extra; a missing `cv2` / `imageio` / `imageio_ffmpeg`
is reported before the model loads. A video that fails to encode mid-episode is dropped
with a warning; the evaluation itself continues.

## 7. Camera parameters

The ribbon projection assumes a forward-facing pinhole camera, level with the floor, at
`cam_height` metres with a horizontal FOV of `hfov_deg` (the vertical FOV follows from the
frame aspect). Only the overlay uses these numbers — the model never sees them.

| where | horizontal FOV | height | source |
|---|---|---|---|
| server (`--cam_hfov_deg`, `--cam_height`) | your robot's camera | your robot's camera | defaults 90° / 0.5 m are a placeholder — set them for your platform |
| eval (`--hfov_deg`, `--cam_height`) | 120° | 0.88 m | `habitat_server/configs/*.yaml` |

A wrong FOV stretches or squeezes the corridor laterally; a wrong height slides it up or
down. Neither affects the pointing discs (those are pixels) or the HUD.

## 8. Fonts

The HUD uses a TrueType font when one is installed — DejaVu Sans (Condensed / Mono
Bold) or Liberation Sans Narrow / Mono from `/usr/share/fonts/truetype/` — and falls back
to OpenCV's built-in Hershey vector font otherwise, so it renders on a bare container.
`apt-get install fonts-dejavu-core` (Debian/Ubuntu) restores the intended look.
