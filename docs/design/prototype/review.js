/* Call review screen. Everything here is driven by VL.reviews[id]. Playback is simulated (no audio file), but the
   playhead, clip range, transcript sync, and keyboard behaviour are the real spec. */
(function () {
  const VL = window.VL;
  const params = new URLSearchParams(location.search);
  const id = params.get("call") in VL.reviews ? params.get("call") : "c_1030_0917_a3f2";
  const R = VL.reviews[id];
  const persona = VL.personaById(R.persona), scenario = VL.scenarioById(R.scenario);
  const isTurn = R.mode === "turn";
  const state = { t: 0, playing: false, sel: null, clip: null, speed: 1, filter: "all", humanVerdict: null, raf: null, last: 0 };
  const LANE_CSS = { w: 0 };

  const keys = [
    ["Play or pause", "Space"], ["Play selected finding's clip", "Enter"], ["Next / previous finding", "j k"],
    ["Confirm finding", "y"], ["Dismiss finding", "n"], ["Add note to finding", "c"],
    ["Seek 2 s / 10 s", "← → Shift"], ["Jump to transcript turn", "t"], ["Next / previous call needing review", "] ["],
    ["Set call verdict pass / fail / review / invalid", "1 2 3 4"], ["Filter transcript to findings", "f"], ["Speed 1x / 1.5x / 2x", "s"],
  ];

  VL.pageInit = function () {
    VL.renderShell({ crumbs: ["Calls", R.run, R.id], keys });
    render();
    bindKeys();
    window.addEventListener("resize", drawWaves);
    document.addEventListener("vl-theme", drawWaves);
    // Start on the first open finding so the reviewer lands on the work
    const first = R.findings.find(f => !f.state);
    if (first) selectFinding(first.id, { seek: true });
  };

  function render() {
    const root = document.getElementById("review");
    root.innerHTML = `
      ${head()}
      <div class="body">
        <section class="timeline" aria-label="Timeline">${timeline()}</section>
        <section class="transcript" id="transcript" aria-label="Transcript">${transcript()}</section>
        <aside class="side" id="side" aria-label="Findings and evaluators">${side()}</aside>
      </div>`;
    bindUI();
    requestAnimationFrame(drawWaves);
    updatePlayhead();
  }

  function head() {
    return `<div class="head">
      <div class="title"><span class="mono">${R.id}</span> ${VL.verdict(R.verdict)} ${VL.mode(R.mode)}</div>
      <div class="meta">
        <span><b>${scenario.id}</b> ${scenario.name}</span>
        <span>persona <b>${persona.name}</b></span>
        <span>profile <b class="mono">${R.profile}</b></span>
        <span>${R.run}, rep <b>${R.rep}</b></span>
        <span><b>${VL.fmtShort(R.duration)}</b> call time${isTurn ? `, <b>${R.wallClock}</b> wall clock` : ""}</span>
        <span>${R.turns}/${R.turnLimit} turns</span>
      </div>
      <div class="nav">
        <span class="faint" id="findCounter"></span>
        <a class="btn sm" href="review.html?call=c_1030_0917_a3f2" title="Previous call needing review"><kbd>[</kbd> prev</a>
        <a class="btn sm primary" href="review.html?call=${R.next}" title="Next call needing review">next needing review <kbd>]</kbd></a>
      </div>
      ${isTurn ? `<div class="note warn" style="flex-basis:100%">${VL.icons.info}<span><b>Turn-based call.</b> Each side waited for the other to finish, so the gaps you see are synthetic and timing was not measured. Transcript, tools, database changes, pronunciation, and rubric results are real.</span></div>` : ""}
    </div>`;
  }

  function timeline() {
    const D = R.duration;
    const ticks = []; const step = D > 240 ? 30 : 15;
    for (let s = 0; s <= D; s += step) ticks.push(`<span style="left:${(s / D) * 100}%">${VL.fmtShort(s)}</span>`);
    const pins = R.findings.map(f => `<button class="pin ${f.sev} ${f.state ? "done" : ""}" data-f="${f.id}" style="left:${(f.t / D) * 100}%" title="${f.title}" aria-label="Finding at ${VL.fmtShort(f.t)}: ${f.title}">${f.id.replace("f", "")}</button>`).join("");
    const evs = R.events.map(e => `<button class="ev ${e.kind}" data-t="${e.t}" style="left:${(e.t / D) * 100}%" title="${e.detail.replace(/"/g, "&quot;")}">${e.kind === "tool" ? VL.icons.tool : e.kind === "db" ? VL.icons.db : ""} ${e.label}</button>`).join("");
    return `
      <div class="tl-toolbar">
        <button class="btn sm" id="playBtn" aria-label="Play">${VL.icons.play} <kbd>Space</kbd></button>
        <span class="time" id="clock">${VL.fmt(0)} <span>/ ${VL.fmt(D)}</span></span>
        <button class="btn ghost sm" id="speedBtn" title="Playback speed">1x</button>
        <span class="faint" id="clipLabel"></span>
        <span class="grow"></span>
        <span class="legend">
          <span><i style="background:var(--caller)"></i>caller</span>
          <span><i style="background:var(--bot)"></i>bot</span>
          <span><i class="hatch"></i>overlap</span>
          <span><i style="background:var(--accent-soft);border:1px solid var(--accent)"></i>clip</span>
        </span>
        ${isTurn ? `<span class="tag" title="Timing evaluator skipped">gaps synthetic</span>` : `<span class="tag mono" title="median bot reply latency">p50 ${R.timing.p50}s</span>`}
      </div>
      <div class="tl-wrap">
        <div class="tl-lanes" id="lanes">
          <div class="ln">findings</div><div class="findings-lane">${pins}</div>
          <div class="ln caller"><i></i>caller</div><div class="lane caller"><canvas id="wCaller" height="56"></canvas></div>
          <div class="ln bot"><i></i>bot</div><div class="lane bot"><canvas id="wBot" height="56"></canvas></div>
          <div class="ln">events</div><div class="events">${evs}</div>
          <div class="ln"></div><div class="ruler">${ticks.join("")}</div>
        </div>
        <div class="tl-clip" id="clip" hidden></div>
        <div class="tl-playhead" id="playhead"></div>
      </div>`;
  }

  function diffWords(intended, heard) {
    // Tiny word diff, enough to show substitution highlights. Production uses a proper alignment from the pronunciation evaluator.
    const a = intended.split(" "), b = heard.split(" ");
    let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++;
    let j = 0; while (j < a.length - i && j < b.length - i && a[a.length - 1 - j] === b[b.length - 1 - j]) j++;
    const pre = b.slice(0, i).join(" "), post = b.slice(b.length - j).join(" ");
    const del = a.slice(i, a.length - j).join(" "), ins = b.slice(i, b.length - j).join(" ");
    return `${pre} <span class="diff-del" title="intended">${del}</span> <span class="diff-ins" title="heard by ASR">${ins}</span> ${post}`;
  }

  function transcript() {
    const rows = R.transcript.map((u, idx) => {
      if (u.who === "event") {
        return `<div class="turn event" data-t="${u.t}" data-idx="${idx}"><div class="who">${u.kind === "db" ? "db" : "tool"}<small>${VL.fmtShort(u.t)}</small></div><div class="txt ${u.kind === "db" ? "dbc" : ""}">${u.kind === "db" ? VL.icons.db : VL.icons.tool} ${esc(u.text)}</div></div>`;
      }
      const f = u.finding ? R.findings.find(x => x.id === u.finding) : null;
      const body = u.diff ? diffWords(u.intended, u.heard) : esc(u.heard);
      return `<div class="turn ${u.who}" data-t="${u.t}" data-idx="${idx}" ${f ? `data-f="${f.id}"` : ""} tabindex="0">
        <div class="who">${u.who}<small>${VL.fmtShort(u.t)}</small></div>
        <div class="txt">
          <div class="heard">${body}${u.barge ? `<span class="barge"> interrupted</span>` : ""}${f ? `<span class="mark ${f.sev}">${f.id.replace("f", "")} ${f.source.replace("_", " ")}</span>` : ""}</div>
          ${u.intended && !u.diff && u.intended !== u.heard ? `<div class="intended"><span class="lab">intended</span>${esc(u.intended)}</div>` : ""}
          ${u.diff ? `<div class="intended"><span class="lab">intended</span>${esc(u.intended)}</div>` : ""}
          ${u.overlap ? `<div class="overlap-note">overlaps bot speech</div>` : ""}
        </div></div>`;
    }).join("");
    return `<div class="filterbar">
        <div class="seg" role="group" aria-label="Transcript filter">
          <button aria-pressed="true" data-filter="all">All turns</button>
          <button aria-pressed="false" data-filter="findings">With findings <kbd>f</kbd></button>
          <button aria-pressed="false" data-filter="events">Tools and db</button>
        </div>
        <span class="grow"></span>
        <span class="faint">bot text shows <span class="diff-del">intended</span> <span class="diff-ins">heard</span> when they differ</span>
      </div>${rows}`;
  }

  function side() {
    const open = R.findings.filter(f => !f.state).length;
    const findings = R.findings.map(findingCard).join("");
    const rubric = R.rubric.map(c => `<tr>
        <td>${c.result === "pass" ? VL.verdict("pass") : c.result === "fail" ? VL.verdict("fail") : VL.verdict("needs_review")}<div class="must">${c.kind === "must" ? "must" : "must not"} ${c.id}</div></td>
        <td>${esc(c.text)}${c.quote ? `<div class="q">"${esc(c.quote)}" <a href="#" data-seek="${c.at}">${VL.fmtShort(c.at)}</a></div>` : c.note ? `<div class="q">${esc(c.note)}</div>` : ""}</td></tr>`).join("");
    const evals = R.evaluators.map(e => `<tr><td style="width:120px">${esc(e.name)}${e.advisory ? `<div class="adv-lab">advisory</div>` : ""}</td><td>${e.status === "skipped" ? `<span class="faint">${esc(e.summary)}</span>` : esc(e.summary)}<div class="faint mono" style="font-size:10px">${e.model || ""}</div></td></tr>`).join("");
    const timing = isTurn
      ? `<div class="kvs"><dt>reply latency</dt><dd class="na">not measured (turn-based)</dd><dt>overlaps</dt><dd class="na">not measured</dd><dt>dead air</dt><dd class="na">not measured</dd></div>`
      : `<div class="kvs"><dt>reply p50 / p95</dt><dd>${R.timing.p50} s / ${R.timing.p95} s</dd><dt>overlaps</dt><dd>${R.timing.overlaps}</dd><dt>dead air events</dt><dd>${R.timing.deadAir}</dd><dt>barge-in stop time</dt><dd>${R.timing.bargeReact} s</dd></div>`;
    return `
      <div class="sec" id="secFindings">
        <div class="sh">Findings <span class="tag mono">${open} open</span><span class="grow"></span><span class="faint" style="text-transform:none;letter-spacing:0;font-weight:500"><kbd>j</kbd><kbd>k</kbd> move, <kbd>y</kbd> confirm, <kbd>n</kbd> dismiss</span></div>
        ${findings}
      </div>
      <div class="sec"><div class="sh">Rubric, ${scenario.id}</div><table class="rubric">${rubric}</table></div>
      <div class="sec"><div class="sh">Database changes</div><div class="dbdiff">${R.dbdiff.map(d => `<div class="${d.k}">${esc(d.s)}</div>`).join("")}</div></div>
      <div class="sec"><div class="sh">Timing</div>${timing}</div>
      <div class="sec"><div class="sh">Evaluators</div><table class="rubric">${evals}</table></div>
      <div class="verdict-bar" id="verdictBar">
        <span class="muted">Call verdict</span> ${VL.verdict(R.verdict)}
        <span class="grow"></span>
        <span class="seg" role="group" aria-label="Set human verdict">
          <button data-v="pass"><kbd>1</kbd> pass</button><button data-v="fail"><kbd>2</kbd> fail</button><button data-v="needs_review" aria-pressed="true"><kbd>3</kbd> review</button><button data-v="invalid"><kbd>4</kbd> invalid</button>
        </span>
      </div>`;
  }

  function findingCard(f) {
    const cls = `finding ${f.state ? f.state : ""} ${f.sev === "adv" ? "advisory" : ""} ${state.sel === f.id ? "sel" : ""}`;
    const src = { timing: "timing", pronunciation: "pronunciation", transcript_judge: "transcript judge", state: "state assertions", audio_listening: "audio listening", audio_quality: "audio quality" }[f.source];
    return `<div class="${cls}" data-f="${f.id}" tabindex="0" role="button" aria-pressed="${state.sel === f.id}">
      <div class="fh"><span class="sev ${f.sev}">${f.sev === "adv" ? "advisory, cannot fail the call" : f.sev}</span><span class="t">${VL.fmtShort(f.t)}</span><span class="src">${src}</span></div>
      <div class="ft">${esc(f.title)}</div>
      <div class="fe">${f.intended ? `<div><span class="faint">intended</span> <code>${esc(f.intended)}</code></div><div><span class="faint">heard</span> <code>${esc(f.heard)}</code></div>` : `<span class="q">${esc(f.evidence)}</span>`}</div>
      ${f.intended ? `<div class="fe" style="border:0;padding:0;margin-top:3px"><span class="q">${esc(f.evidence)}</span></div>` : ""}
      <div class="fx">
        <button class="btn sm" data-play="${f.id}">${VL.icons.play} clip ${VL.fmtShort(Math.max(0, f.t - 1))}-${VL.fmtShort(f.end + 1)} <kbd>↵</kbd></button>
        <span class="grow"></span>
        <button class="btn sm pass" data-confirm="${f.id}" aria-pressed="${f.state === "confirmed"}">${VL.icons.check} <kbd>y</kbd></button>
        <button class="btn sm" data-dismiss="${f.id}" aria-pressed="${f.state === "dismissed"}">${VL.icons.x} <kbd>n</kbd></button>
        <button class="btn sm ghost" data-note="${f.id}" title="Add note">note <kbd>c</kbd></button>
      </div>
      <div class="fnote" data-notewrap="${f.id}" ${f.note == null && !f.noteOpen ? "hidden" : ""}>
        ${f.noteOpen ? `<textarea class="in" data-noteinput="${f.id}" placeholder="Why? One line is enough. Saved on blur or Ctrl+Enter.">${f.note || ""}</textarea>` : f.note ? `<div class="saved">${esc(f.note)} <a href="#" data-note="${f.id}">edit</a></div>` : ""}
      </div>
    </div>`;
  }

  /* ---------- waveform ---------- */
  function seeded(n) { const x = Math.sin(n * 12.9898) * 43758.5453; return x - Math.floor(x); }
  function drawWave(canvas, segs, color, D) {
    const w = canvas.clientWidth, h = canvas.height; canvas.width = Math.max(1, Math.floor(w * devicePixelRatio));
    const ctx = canvas.getContext("2d"); ctx.scale(devicePixelRatio, 1);
    ctx.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    const over = css.getPropertyValue("--overlap").trim();
    // overlaps: hatched band first so bars draw over it
    R.overlaps.forEach(([a, b]) => {
      const x0 = a / D * w, x1 = b / D * w;
      ctx.save(); ctx.beginPath(); ctx.rect(x0, 0, x1 - x0, h); ctx.clip();
      ctx.strokeStyle = over; ctx.lineWidth = 2;
      for (let x = x0 - h; x < x1 + h; x += 6) { ctx.beginPath(); ctx.moveTo(x, h); ctx.lineTo(x + h, 0); ctx.stroke(); }
      ctx.restore();
    });
    ctx.fillStyle = color;
    const mid = h / 2;
    for (let x = 0; x < w; x += 2) {
      const t = x / w * D;
      let amp = 0.025;
      for (const [a, b] of segs) {
        if (t >= a && t <= b) {
          const rel = (t - a) / (b - a);
          const env = Math.min(1, rel * 8, (1 - rel) * 6);
          amp = (0.25 + 0.65 * seeded(x * 0.37 + a)) * env * (0.6 + 0.4 * Math.abs(Math.sin(t * 7)));
          break;
        }
      }
      const bh = Math.max(1, amp * (h - 6));
      ctx.fillRect(x, mid - bh / 2, 1.4, bh);
    }
  }
  function drawWaves() {
    const c1 = document.getElementById("wCaller"), c2 = document.getElementById("wBot"); if (!c1) return;
    const css = getComputedStyle(document.documentElement);
    drawWave(c1, R.speech.caller, css.getPropertyValue("--caller").trim(), R.duration);
    drawWave(c2, R.speech.bot, css.getPropertyValue("--bot").trim(), R.duration);
    LANE_CSS.w = c1.clientWidth;
    updatePlayhead();
  }

  /* ---------- playback (simulated) ---------- */
  function laneRect() { const l = document.getElementById("lanes"); const c = document.getElementById("wCaller"); const lr = l.getBoundingClientRect(), cr = c.getBoundingClientRect(); return { left: cr.left - lr.left, width: cr.width }; }
  function updatePlayhead() {
    const ph = document.getElementById("playhead"); if (!ph) return;
    const r = laneRect();
    ph.style.left = (r.left + state.t / R.duration * r.width) + "px";
    document.getElementById("clock").innerHTML = `${VL.fmt(state.t)} <span>/ ${VL.fmt(R.duration)}</span>`;
    const clip = document.getElementById("clip");
    if (state.clip) { clip.hidden = false; clip.style.left = (r.left + state.clip[0] / R.duration * r.width) + "px"; clip.style.width = ((state.clip[1] - state.clip[0]) / R.duration * r.width) + "px"; }
    else clip.hidden = true;
    // active transcript turn
    let active = null;
    document.querySelectorAll("#transcript .turn").forEach(el => { const t = +el.dataset.t; if (t <= state.t) active = el; });
    document.querySelectorAll("#transcript .turn.active").forEach(el => el !== active && el.classList.remove("active"));
    if (active && !active.classList.contains("active")) { active.classList.add("active"); if (state.playing) active.scrollIntoView({ block: "center", behavior: "smooth" }); }
  }
  function tick(now) {
    if (!state.playing) return;
    const dt = (now - state.last) / 1000 * state.speed; state.last = now;
    state.t = Math.min(R.duration, state.t + dt);
    if (state.clip && state.t >= state.clip[1]) { state.t = state.clip[1]; pause(); }
    if (state.t >= R.duration) pause();
    updatePlayhead();
    state.raf = requestAnimationFrame(tick);
  }
  function play() { state.playing = true; state.last = performance.now(); document.getElementById("playBtn").innerHTML = `${VL.icons.pause} <kbd>Space</kbd>`; document.getElementById("playBtn").setAttribute("aria-label", "Pause"); state.raf = requestAnimationFrame(tick); }
  function pause() { state.playing = false; cancelAnimationFrame(state.raf); const b = document.getElementById("playBtn"); b.innerHTML = `${VL.icons.play} <kbd>Space</kbd>`; b.setAttribute("aria-label", "Play"); }
  function toggle() { state.playing ? pause() : play(); }
  function seek(t) { state.t = Math.max(0, Math.min(R.duration, t)); updatePlayhead(); }
  function playClip(f) {
    state.clip = [Math.max(0, f.t - 1), Math.min(R.duration, f.end + 1)];
    document.getElementById("clipLabel").textContent = `clip ${VL.fmtShort(state.clip[0])} to ${VL.fmtShort(state.clip[1])}, finding ${f.id.replace("f", "")}`;
    seek(state.clip[0]); play();
  }

  /* ---------- findings ---------- */
  function selectFinding(fid, opts = {}) {
    state.sel = fid;
    document.querySelectorAll(".finding").forEach(el => { const on = el.dataset.f === fid; el.classList.toggle("sel", on); el.setAttribute("aria-pressed", on); });
    document.querySelectorAll(".pin").forEach(el => el.classList.toggle("sel", el.dataset.f === fid));
    const f = R.findings.find(x => x.id === fid);
    const card = document.querySelector(`.finding[data-f="${fid}"]`); if (card) card.scrollIntoView({ block: "nearest" });
    if (opts.seek && f) { state.clip = [Math.max(0, f.t - 1), Math.min(R.duration, f.end + 1)]; seek(state.clip[0]); document.getElementById("clipLabel").textContent = `clip ready, finding ${f.id.replace("f", "")}`; }
    const turn = document.querySelector(`#transcript .turn[data-f="${fid}"]`); if (turn) turn.scrollIntoView({ block: "center" });
    const i = R.findings.findIndex(x => x.id === fid);
    document.getElementById("findCounter").textContent = `finding ${i + 1} of ${R.findings.length}`;
  }
  function moveFinding(dir) {
    const i = R.findings.findIndex(x => x.id === state.sel);
    const n = (i + dir + R.findings.length) % R.findings.length;
    selectFinding(R.findings[n].id, { seek: true });
  }
  function setState(fid, s) {
    const f = R.findings.find(x => x.id === fid);
    f.state = f.state === s ? null : s;
    rerenderFinding(fid);
    const open = R.findings.filter(x => !x.state).length;
    document.querySelector("#secFindings .tag").textContent = `${open} open`;
    VL.toast(f.state ? `Finding ${fid.replace("f", "")} ${f.state}` : `Finding ${fid.replace("f", "")} reopened`);
    const pin = document.querySelector(`.pin[data-f="${fid}"]`); pin.classList.toggle("done", !!f.state);
    // Advance to the next open finding so the reviewer keeps moving
    if (f.state) { const nx = R.findings.find(x => !x.state); if (nx) selectFinding(nx.id, { seek: true }); else { VL.toast("All findings reviewed. Set the call verdict (1-4) or press ] for the next call."); document.getElementById("verdictBar").scrollIntoView({ block: "end" }); } }
  }
  function rerenderFinding(fid) {
    const f = R.findings.find(x => x.id === fid);
    const el = document.querySelector(`.finding[data-f="${fid}"]`);
    const tmp = document.createElement("div"); tmp.innerHTML = findingCard(f); el.replaceWith(tmp.firstElementChild);
    const ta = document.querySelector(`[data-noteinput="${fid}"]`);
    if (ta) { ta.focus(); ta.addEventListener("blur", () => saveNote(fid)); ta.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") ta.blur(); if (e.key === "Escape") { f.noteOpen = false; rerenderFinding(fid); } }); }
  }
  function saveNote(fid) {
    const f = R.findings.find(x => x.id === fid);
    const ta = document.querySelector(`[data-noteinput="${fid}"]`); if (!ta) return;
    f.note = ta.value.trim() || null; f.noteOpen = false; rerenderFinding(fid); if (f.note) VL.toast("Note saved");
  }
  function openNote(fid) { const f = R.findings.find(x => x.id === fid); f.noteOpen = true; selectFinding(fid); rerenderFinding(fid); }

  /* ---------- bindings ---------- */
  function bindUI() {
    const root = document.getElementById("review");
    root.addEventListener("click", e => {
      const el = e.target.closest("[data-f],[data-t],[data-play],[data-confirm],[data-dismiss],[data-note],[data-seek],[data-filter],[data-v],#playBtn,#speedBtn");
      if (!el) return;
      if (el.id === "playBtn") return toggle();
      if (el.id === "speedBtn") return cycleSpeed();
      if (el.dataset.play) { const f = R.findings.find(x => x.id === el.dataset.play); selectFinding(f.id); return playClip(f); }
      if (el.dataset.confirm) return setState(el.dataset.confirm, "confirmed");
      if (el.dataset.dismiss) return setState(el.dataset.dismiss, "dismissed");
      if (el.dataset.note) { e.preventDefault(); return openNote(el.dataset.note); }
      if (el.dataset.seek) { e.preventDefault(); return seek(+el.dataset.seek); }
      if (el.dataset.filter) return setFilter(el.dataset.filter);
      if (el.dataset.v) return setVerdict(el.dataset.v);
      if (el.classList.contains("pin")) { const f = R.findings.find(x => x.id === el.dataset.f); selectFinding(f.id); return playClip(f); }
      if (el.classList.contains("finding")) return selectFinding(el.dataset.f, { seek: true });
      if (el.classList.contains("turn")) { seek(+el.dataset.t); if (el.dataset.f) selectFinding(el.dataset.f); return; }
      if (el.classList.contains("ev")) return seek(+el.dataset.t - 0.5);
    });
    // click on waveform to seek
    document.getElementById("lanes").addEventListener("click", e => {
      const lane = e.target.closest(".lane"); if (!lane) return;
      const r = lane.getBoundingClientRect(); state.clip = null; document.getElementById("clipLabel").textContent = "";
      seek((e.clientX - r.left) / r.width * R.duration);
    });
  }
  function cycleSpeed() { state.speed = state.speed === 1 ? 1.5 : state.speed === 1.5 ? 2 : 1; document.getElementById("speedBtn").textContent = state.speed + "x"; }
  function setFilter(f) {
    state.filter = f;
    document.querySelectorAll("[data-filter]").forEach(b => b.setAttribute("aria-pressed", b.dataset.filter === f));
    document.querySelectorAll("#transcript .turn").forEach(el => {
      const show = f === "all" || (f === "findings" && el.dataset.f) || (f === "events" && el.classList.contains("event"));
      el.style.display = show ? "" : "none";
    });
  }
  function setVerdict(v) {
    state.humanVerdict = v;
    document.querySelectorAll("#verdictBar [data-v]").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === v));
    VL.toast(`Human verdict set to ${VL.verdictLabel[v]}. Press ] for the next call.`);
  }
  function bindKeys() {
    document.addEventListener("keydown", e => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      const k = e.key;
      if (k === " ") { e.preventDefault(); return toggle(); }
      if (k === "Enter" && state.sel) { const f = R.findings.find(x => x.id === state.sel); if (!e.target.closest(".btn")) { e.preventDefault(); playClip(f); } return; }
      if (k === "j") return moveFinding(1);
      if (k === "k") return moveFinding(-1);
      if (k === "y" && state.sel) return setState(state.sel, "confirmed");
      if (k === "n" && state.sel) return setState(state.sel, "dismissed");
      if (k === "c" && state.sel) { e.preventDefault(); return openNote(state.sel); }
      if (k === "ArrowLeft") { e.preventDefault(); return seek(state.t - (e.shiftKey ? 10 : 2)); }
      if (k === "ArrowRight") { e.preventDefault(); return seek(state.t + (e.shiftKey ? 10 : 2)); }
      if (k === "f") return setFilter(state.filter === "findings" ? "all" : "findings");
      if (k === "s") return cycleSpeed();
      if (k === "t") { const a = document.querySelector("#transcript .turn.active"); if (a) a.focus(); return; }
      if (k === "]") return location.href = `review.html?call=${R.next}`;
      if (k === "[") return location.href = `review.html?call=c_1030_0917_a3f2`;
      if (["1", "2", "3", "4"].includes(k)) return setVerdict(["pass", "fail", "needs_review", "invalid"][+k - 1]);
    });
  }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
})();
