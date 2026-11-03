/* Shared shell: nav rail, topbar, theme, keyboard help, small helpers. */
(function () {
  const VL = window.VL;

  VL.icons = {
    overview: '<svg class="ico" viewBox="0 0 16 16"><path d="M2 9.5h4V14H2zM6 2h4v12H6zM10 6h4v8h-4z"/></svg>',
    calls: '<svg class="ico" viewBox="0 0 16 16"><path d="M2 3h12M2 8h12M2 13h8"/></svg>',
    review: '<svg class="ico" viewBox="0 0 16 16"><path d="M1 8h2l2-5 3 10 2-7 1 2h4"/></svg>',
    persona: '<svg class="ico" viewBox="0 0 16 16"><circle cx="8" cy="5" r="3"/><path d="M2 14c1-3 3-4 6-4s5 1 6 4"/></svg>',
    scenario: '<svg class="ico" viewBox="0 0 16 16"><path d="M3 2h7l3 3v9H3z"/><path d="M6 8h4M6 11h4"/></svg>',
    run: '<svg class="ico" viewBox="0 0 16 16"><path d="M4 2l10 6-10 6z"/></svg>',
    talk: '<svg class="ico" viewBox="0 0 16 16"><rect x="5.5" y="1.5" width="5" height="8" rx="2.5"/><path d="M3 7.5a5 5 0 0 0 10 0M8 12.5v2"/></svg>',
    upload: '<svg class="ico" viewBox="0 0 16 16"><path d="M8 11V3M4.5 6.5 8 3l3.5 3.5M2 13h12"/></svg>',
    compare: '<svg class="ico" viewBox="0 0 16 16"><path d="M2 3h5v10H2zM9 3h5v10H9z"/></svg>',
    settings: '<svg class="ico" viewBox="0 0 16 16"><circle cx="8" cy="8" r="2.2"/><path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4"/></svg>',
    play: '<svg class="ico" viewBox="0 0 16 16"><path d="M4 2.5l9 5.5-9 5.5z"/></svg>',
    pause: '<svg class="ico" viewBox="0 0 16 16"><path d="M4.5 3v10M11.5 3v10"/></svg>',
    check: '<svg class="ico" viewBox="0 0 16 16"><path d="M3 8.5l3 3 7-7"/></svg>',
    x: '<svg class="ico" viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
    warn: '<svg class="ico" viewBox="0 0 16 16"><path d="M8 2 1.5 13.5h13z"/><path d="M8 6.5v3.5M8 12v.5"/></svg>',
    info: '<svg class="ico" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.5"/><path d="M8 7v4M8 5v.5"/></svg>',
    sun: '<svg class="ico" viewBox="0 0 16 16"><circle cx="8" cy="8" r="3"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6 13 13M3 13l1.4-1.4M11.6 4.4 13 3"/></svg>',
    moon: '<svg class="ico" viewBox="0 0 16 16"><path d="M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7z"/></svg>',
    live: '<svg class="ico" viewBox="0 0 16 16"><path d="M1 8h3l2-4 3 8 2-5 1 1h3"/></svg>',
    turn: '<svg class="ico" viewBox="0 0 16 16"><path d="M3 5h8l-2-2M13 11H5l2 2"/></svg>',
    mic: '<svg class="ico" viewBox="0 0 16 16"><rect x="5.5" y="1.5" width="5" height="8" rx="2.5"/><path d="M3 7.5a5 5 0 0 0 10 0M8 12.5v2"/></svg>',
    headphones: '<svg class="ico" viewBox="0 0 16 16"><path d="M2 10V8a6 6 0 0 1 12 0v2"/><rect x="2" y="9" width="3" height="5" rx="1"/><rect x="11" y="9" width="3" height="5" rx="1"/></svg>',
    tool: '<svg class="ico" viewBox="0 0 16 16"><path d="M9.5 2.5a3.5 3.5 0 0 0 4 4L8 12l-2 2-2-2 2-2 5.5-5.5"/></svg>',
    db: '<svg class="ico" viewBox="0 0 16 16"><ellipse cx="8" cy="4" rx="5.5" ry="2"/><path d="M2.5 4v8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2V4M2.5 8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2"/></svg>',
  };

  const NAV = [
    { group: "Results", items: [
      { href: "index.html", key: "overview", label: "Overview" },
      { href: "calls.html", key: "calls", label: "Calls", count: 7, hot: true },
      { href: "compare.html", key: "compare", label: "Compare" },
    ]},
    { group: "Test design", items: [
      { href: "scenario.html", key: "scenario", label: "Scenarios" },
      { href: "persona.html", key: "persona", label: "Personas" },
    ]},
    { group: "Run", items: [
      { href: "run.html", key: "run", label: "Run launcher" },
      { href: "talk.html", key: "talk", label: "Talk to the bot" },
      { href: "upload.html", key: "upload", label: "Upload recording" },
    ]},
    { group: "System", items: [
      { href: "settings.html", key: "settings", label: "Models and settings" },
    ]},
  ];

  function themeInit() {
    const saved = localStorage.getItem("vl-theme");
    const q = new URLSearchParams(location.search).get("theme");
    document.documentElement.dataset.theme = q || saved || "dark";
  }
  function themeToggle() {
    const el = document.documentElement;
    el.dataset.theme = el.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("vl-theme", el.dataset.theme);
    document.dispatchEvent(new CustomEvent("vl-theme"));
    const b = document.getElementById("themeBtn"); if (b) b.innerHTML = themeIcon();
  }
  function themeIcon() { return document.documentElement.dataset.theme === "dark" ? VL.icons.sun : VL.icons.moon; }

  VL.renderShell = function (opts) {
    const page = document.body.dataset.page;
    const rail = document.getElementById("rail");
    if (rail) {
      let h = `<div class="brand"><span class="mark"></span><div><b>voice-eval-lab</b><small>Fernhill Internet bot</small></div></div>`;
      NAV.forEach(g => {
        h += `<div class="group">${g.group}</div>`;
        g.items.forEach(it => {
          const cur = it.key === page || (page === "review" && it.key === "calls");
          h += `<a class="item" href="${it.href}" ${cur ? 'aria-current="page"' : ""}>${VL.icons[it.key] || ""}<span>${it.label}</span>${it.count ? `<span class="count ${it.hot ? "hot" : ""}" title="${it.count} calls need review">${it.count}</span>` : ""}</a>`;
        });
      });
      h += `<div class="spacer"></div><div class="foot">
        <div class="row"><span>profile</span><span class="mono">local-default</span></div>
        <div class="row"><span class="health loading" title="judge LLM loading">6 of 7 models</span><button class="btn ghost sm" id="themeBtn" aria-label="Toggle theme">${themeIcon()}</button></div>
        <div class="row"><span>192.168.1.50:8443</span><span><kbd>?</kbd> keys</span></div>
      </div>`;
      rail.innerHTML = h;
      document.getElementById("themeBtn").addEventListener("click", themeToggle);
    }
    const tb = document.getElementById("topbar");
    if (tb) {
      const crumbs = (opts.crumbs || []).map((c, i, a) => i === a.length - 1 ? `<span class="cur">${c}</span>` : `<span>${c}</span><span class="sep">/</span>`).join("");
      tb.innerHTML = `<div class="crumbs">${crumbs}</div><div class="grow"></div>
        <div class="search" role="button" tabindex="0" aria-label="Jump to call, scenario, or run">Jump to call, scenario, run <kbd>Ctrl</kbd><kbd>K</kbd></div>
        ${opts.actions || ""}`;
    }
    // Keyboard help overlay
    const ov = document.createElement("div");
    ov.className = "overlay"; ov.id = "keysOverlay";
    ov.innerHTML = `<div class="dialog" role="dialog" aria-label="Keyboard shortcuts"><div class="ph"><h2>Keyboard</h2><span class="grow"></span><button class="btn ghost sm" data-close>Close <kbd>Esc</kbd></button></div>
      <div class="pb"><dl class="keys">${(opts.keys || VL.globalKeys).map(k => `<dt><span>${k[0]}</span><span class="kk">${k[1].split(" ").map(x => `<kbd>${x}</kbd>`).join("")}</span></dt><dd></dd>`).join("")}</dl>
      <p class="faint" style="margin-top:12px">Global: <kbd>?</kbd> this help, <kbd>Ctrl</kbd><kbd>K</kbd> jump, <kbd>g</kbd> then <kbd>o</kbd> overview, <kbd>g</kbd> <kbd>c</kbd> calls, <kbd>g</kbd> <kbd>r</kbd> next review, <kbd>g</kbd> <kbd>t</kbd> talk.</p></div></div>`;
    document.body.appendChild(ov);
    ov.addEventListener("click", e => { if (e.target === ov || e.target.closest("[data-close]")) ov.classList.remove("show"); });
    let gPending = false;
    document.addEventListener("keydown", e => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable) return;
      if (e.key === "?") { ov.classList.toggle("show"); e.preventDefault(); return; }
      if (e.key === "Escape") { ov.classList.remove("show"); }
      if (gPending) { gPending = false; const m = { o: "index.html", c: "calls.html", r: "review.html", t: "talk.html", s: "scenario.html", l: "run.html" }[e.key]; if (m) location.href = m; return; }
      if (e.key === "g") { gPending = true; setTimeout(() => gPending = false, 900); }
    });
    const t = document.createElement("div"); t.className = "toast"; t.id = "toast"; document.body.appendChild(t);
  };

  VL.globalKeys = [["Toggle this help", "?"], ["Jump to anything", "Ctrl K"], ["Overview", "g o"], ["Calls", "g c"], ["Next call needing review", "g r"], ["Talk to the bot", "g t"]];

  let toastTimer;
  VL.toast = function (msg) {
    const t = document.getElementById("toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), 1800);
  };

  VL.fmt = function (s) { if (s == null) return "--:--"; const m = Math.floor(s / 60), r = s - m * 60; return `${String(m).padStart(2, "0")}:${r.toFixed(1).padStart(4, "0")}`; };
  VL.fmtShort = function (s) { if (s == null) return "--:--"; const m = Math.floor(s / 60), r = Math.floor(s - m * 60); return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`; };
  VL.verdictLabel = { pass: "pass", fail: "fail", needs_review: "needs review", invalid: "invalid", running: "running", queued: "queued" };
  VL.verdict = function (v, extra) { return `<span class="v ${v}" ${extra || ""}>${VL.verdictLabel[v] || v}</span>`; };
  VL.mode = function (m) {
    const map = { live: ["live", VL.icons.live, "Live, real time. Timing metrics are real."], turn: ["turn-based", VL.icons.turn, "Turn-based, slower than real time. Gaps between turns are synthetic, so timing metrics are not measured."], human: ["mic", VL.icons.mic, "Human caller through the browser microphone. Live."], upload: ["upload", VL.icons.upload, "Uploaded recording. Reduced evidence."] };
    const [lab, ico, tip] = map[m] || [m, "", ""];
    return `<span class="mode ${m}" title="${tip}">${ico}${lab}</span>`;
  };
  VL.rateBar = function (r) {
    const n = r.n || (r.pass + r.fail + r.review + r.invalid);
    const seg = (c, v) => v ? `<i class="${c}" style="flex:${v}" title="${c}: ${v}"></i>` : "";
    return `<div class="rate" role="img" aria-label="${r.pass} pass, ${r.fail} fail, ${r.review} needs review, ${r.invalid} invalid of ${n}">${seg("p", r.pass)}${seg("f", r.fail)}${seg("r", r.review)}${seg("i", r.invalid)}</div>`;
  };
  // Wilson 95% interval, shown so a pass rate from 8 calls is not read like one from 80
  VL.ci = function (k, n) {
    if (!n) return "";
    const z = 1.96, p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n), m = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    return `${Math.round(100 * (c - m) / d)}-${Math.round(100 * (c + m) / d)}%`;
  };

  themeInit();
  document.addEventListener("DOMContentLoaded", () => { if (window.VL.pageInit) window.VL.pageInit(); });
})();
