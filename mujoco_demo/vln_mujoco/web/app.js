(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const ui = {
    cameraImage: $("camera-image"), path: $("path-overlay"),
    firstPersonView: $("first-person-view"), thirdPersonView: $("third-person-view"),
    simTime: $("sim-time"), cameraFps: $("camera-fps"), latency: $("latency"), count: $("waypoint-count"),
    serverUrl: $("server-url"), saveServer: $("save-server"), instruction: $("instruction"), vlnToggle: $("vln-toggle"),
    vlnState: $("vln-state"), vlnDetail: $("vln-detail"), controlToggle: $("control-toggle"), stop: $("stop-button"), reset: $("reset-button"),
    toast: $("toast")
  };
  const state = {socket:null, connected:false, reconnect:null, data:null, keys:new Set(), toastTimer:null, cameraBusy:false, cameraUrl:null, cameraView:"first", serverUrlDirty:false};

  function send(payload) {
    if (state.socket?.readyState !== WebSocket.OPEN) { toast("Web connection not ready", true); return false; }
    state.socket.send(JSON.stringify(payload));
    return true;
  }
  function connect() {
    clearTimeout(state.reconnect);
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${location.host}/ws`);
    state.socket = socket;
    socket.onopen = () => { state.connected = true; render(); };
    socket.onclose = () => {
      state.connected = false; state.keys.clear(); render();
      state.reconnect = setTimeout(connect, 1000);
    };
    socket.onerror = () => socket.close();
    socket.onmessage = (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch (_) { return; }
      if (["snapshot", "runtime"].includes(message.type)) {
        state.data = message.data;
        if (message.type === "snapshot" && !state.serverUrlDirty && typeof message.data?.vln?.server_url === "string") {
          ui.serverUrl.value = message.data.vln.server_url;
        }
      }
      if (message.type === "command_result") {
        if (message.ok && typeof message.server_url === "string") {
          ui.serverUrl.value = message.server_url;
          state.serverUrlDirty = false;
        }
        toast(message.message || (message.ok ? "Done" : "Failed"), !message.ok);
      }
      render();
    };
  }
  async function refreshCamera() {
    if (state.cameraBusy) return;
    state.cameraBusy = true;
    const cameraView = state.cameraView;
    let nextUrl = null;
    try {
      const endpoint = cameraView === "third" ? "/api/third-person.jpg" : "/api/camera.jpg";
      const response = await fetch(`${endpoint}?t=${Date.now()}`, {cache:"no-store"});
      if (!response.ok) throw new Error("camera unavailable");
      nextUrl = URL.createObjectURL(await response.blob());
      if (cameraView !== state.cameraView) {
        URL.revokeObjectURL(nextUrl);
        nextUrl = null;
        return;
      }
      await new Promise((resolve, reject) => {
        ui.cameraImage.onload = resolve; ui.cameraImage.onerror = reject; ui.cameraImage.src = nextUrl;
      });
      const oldUrl = state.cameraUrl; state.cameraUrl = nextUrl; nextUrl = null;
      if (oldUrl) URL.revokeObjectURL(oldUrl);
    } catch (_) {
      if (nextUrl) URL.revokeObjectURL(nextUrl);
    } finally { state.cameraBusy = false; }
  }
  function render() {
    const data = state.data || {};
    const sim = data.simulation || {}, camera = sim.camera || {}, vln = data.vln || {}, mpc = data.mpc || {}, control = data.control || {};
    ui.simTime.textContent = `${Number(sim.sim_time || 0).toFixed(1)} s`;
    ui.cameraFps.textContent = camera.ready ? `${Number(camera.fps || 0).toFixed(1)} Hz` : "—";
    ui.firstPersonView.classList.toggle("active", state.cameraView === "first");
    ui.thirdPersonView.classList.toggle("active", state.cameraView === "third");
    ui.latency.textContent = Number.isFinite(Number(vln.latency_ms)) ? `${Number(vln.latency_ms).toFixed(0)} ms` : "—";
    const waypoints = Array.isArray(vln.waypoints) ? vln.waypoints : [];
    ui.count.textContent = String(waypoints.length);
    ui.vlnState.textContent = vln.state || "IDLE";
    ui.vlnToggle.textContent = control.auto ? "Stop VLN" : "Start VLN";
    ui.vlnToggle.classList.toggle("active", Boolean(control.auto));
    ui.vlnDetail.textContent = mpc.error || vln.error || (vln.stop === true
      ? "VLN reached the goal and stopped automatically."
      : (control.auto
        ? (vln.connected ? (vln.visible === false ? "Target is currently not visible." : `MPC tracking the VLN path${Number.isFinite(Number(mpc.solve_ms)) ? ` · ${Number(mpc.solve_ms).toFixed(1)} ms` : ""}`) : "Connecting to the VLN server…")
        : "Set the server URL to get started."));
    ui.controlToggle.textContent = control.manual ? "Release" : (control.locked ? "In use" : "Take control");
    ui.controlToggle.classList.toggle("active", Boolean(control.manual));
    ui.controlToggle.disabled = !state.connected || (control.locked && !control.manual);
    document.querySelectorAll(".drive-key").forEach((button) => {
      button.disabled = !control.manual;
      button.classList.toggle("active", state.keys.has(button.dataset.key));
    });
    drawPath(waypoints, Boolean(control.auto), vln, state.cameraView === "first");
  }
  function drawPath(points, enabled, vln, firstPerson) {
    const context = ui.path.getContext("2d");
    context.clearRect(0, 0, ui.path.width, ui.path.height);
    if (!firstPerson) return;
    if (enabled && points.length) {
      const project = (point) => [240 - Number(point[1]) * 75, 258 - Number(point[0]) * 75];
      context.lineWidth = 3; context.strokeStyle = "#60a5fa"; context.fillStyle = "#60a5fa";
      context.beginPath(); context.moveTo(240, 258);
      points.forEach((point) => { const [x,y] = project(point); context.lineTo(x,y); });
      context.stroke();
      points.forEach((point, index) => { const [x,y] = project(point); context.beginPath(); context.arc(x,y,index === points.length-1 ? 5 : 3,0,Math.PI*2); context.fill(); });
    }
    const drawMarker = (pixel, channel, semanticState, color) => {
      if (!Array.isArray(pixel) || pixel.length !== 2) return;
      const x = Number(pixel[0]), y = Number(pixel[1]);
      if (![x, y].every(Number.isFinite) || x < 0 || x >= 480 || y < 0 || y >= 270) return;
      const label = semanticState ? `${channel} · ${semanticState}` : channel;
      context.save();
      context.strokeStyle = color; context.fillStyle = color; context.lineWidth = 2;
      context.beginPath(); context.arc(x, y, 6, 0, Math.PI * 2); context.stroke();
      context.beginPath(); context.moveTo(x - 9, y); context.lineTo(x + 9, y); context.moveTo(x, y - 9); context.lineTo(x, y + 9); context.stroke();
      context.font = "700 9px ui-monospace, monospace";
      const textWidth = context.measureText(label).width;
      const labelX = Math.max(3, Math.min(480 - textWidth - 9, x + 9));
      const labelY = y < 18 ? y + 18 : y - 8;
      context.fillStyle = "rgba(4, 6, 10, .78)"; context.fillRect(labelX - 3, labelY - 10, textWidth + 6, 13);
      context.fillStyle = color; context.fillText(label, labelX, labelY);
      context.restore();
    };
    drawMarker(vln.apos_px, "APOS", vln.apos_state, "#ffb454");
    drawMarker(vln.opos_px, "OPOS", vln.opos_state, "#6ee7b7");
  }
  function toast(message, error=false) {
    clearTimeout(state.toastTimer); ui.toast.textContent = message; ui.toast.classList.toggle("error", error); ui.toast.classList.add("visible");
    state.toastTimer = setTimeout(() => ui.toast.classList.remove("visible"), 2200);
  }
  const MANUAL_LINEAR = 1.0;   // m/s
  const MANUAL_ANGULAR = 1.0;  // rad/s
  function manualVelocity() {
    return {
      linear: ((state.keys.has("w") ? 1 : 0) - (state.keys.has("s") ? 1 : 0)) * MANUAL_LINEAR,
      angular: ((state.keys.has("a") ? 1 : 0) - (state.keys.has("d") ? 1 : 0)) * MANUAL_ANGULAR
    };
  }
  function publishManual() {
    if (!state.data?.control?.manual) return;
    send({type:"twist", ...manualVelocity()});
  }
  function setKey(key, active) {
    if (!"wasd".includes(key)) return;
    if (active) state.keys.add(key); else state.keys.delete(key);
    publishManual(); render();
  }

  ui.serverUrl.addEventListener("input", () => { state.serverUrlDirty = true; });
  ui.saveServer.addEventListener("click", () => send({type:"set_server_url", server_url:ui.serverUrl.value.trim()}));
  ui.firstPersonView.addEventListener("click", () => { state.cameraView = "first"; render(); refreshCamera(); });
  ui.thirdPersonView.addEventListener("click", () => { state.cameraView = "third"; render(); refreshCamera(); });
  ui.vlnToggle.addEventListener("click", () => {
    const enabled = !state.data?.control?.auto;
    const instruction = ui.instruction.value.trim();
    if (enabled && !instruction) return toast("Enter a navigation instruction", true);
    send({type:"set_vln", enabled, instruction});
  });
  ui.controlToggle.addEventListener("click", () => send({type:state.data?.control?.manual ? "release_control" : "acquire_control"}));
  ui.stop.addEventListener("click", () => { state.keys.clear(); send({type:"stop"}); });
  ui.reset.addEventListener("click", () => { state.keys.clear(); send({type:"reset"}); });
  document.querySelectorAll(".drive-key").forEach((button) => {
    const key = button.dataset.key;
    button.addEventListener("pointerdown", (event) => { event.preventDefault(); button.setPointerCapture(event.pointerId); setKey(key, true); });
    button.addEventListener("pointerup", () => setKey(key, false));
    button.addEventListener("pointercancel", () => setKey(key, false));
  });
  window.addEventListener("keydown", (event) => {
    if (event.repeat || ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if ("wasd".includes(event.key.toLowerCase())) { event.preventDefault(); setKey(event.key.toLowerCase(), true); }
    if (event.key === " ") { event.preventDefault(); state.keys.clear(); send({type:"stop"}); }
  });
  window.addEventListener("keyup", (event) => setKey(event.key.toLowerCase(), false));
  window.addEventListener("blur", () => { state.keys.clear(); publishManual(); render(); });
  setInterval(publishManual, 100);
  setInterval(refreshCamera, 90);
  connect(); render();
})();
