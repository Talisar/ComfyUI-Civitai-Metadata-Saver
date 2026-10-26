import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ROUTE = "/kobold_prompt";

const CSS = `
.kp-panel { display: flex; flex-direction: column; gap: 4px; font: 12px sans-serif; color: var(--input-text, #ddd); }
.kp-row { display: flex; gap: 4px; align-items: center; }
.kp-row select { flex: 1; min-width: 0; }
.kp-panel select, .kp-panel button, .kp-modal button, .kp-modal input {
  background: var(--comfy-input-bg, #222); color: var(--input-text, #ddd);
  border: 1px solid var(--border-color, #555); border-radius: 4px; padding: 2px 6px; font: inherit; }
.kp-panel button:hover, .kp-modal button:hover { filter: brightness(1.3); cursor: pointer; }
.kp-status { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kp-status.warn { white-space: normal; color: #f5b041; line-height: 1.3; }
.kp-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; background: #888; }
.kp-dot.ok { background: #3c3; } .kp-dot.bad { background: #d33; }
.kp-shade { position: fixed; inset: 0; background: rgba(0,0,0,.5); z-index: 10000; display: flex; align-items: center; justify-content: center; }
.kp-modal { background: var(--comfy-menu-bg, #353535); color: var(--fg-color, #fff); border: 1px solid var(--border-color, #555);
  border-radius: 8px; padding: 14px; width: min(760px, 92vw); display: flex; flex-direction: column; gap: 10px; font: 14px sans-serif; }
.kp-modal h3 { margin: 0; font-size: 15px; }
.kp-modal p { margin: 0; opacity: .8; font-size: 12px; }
.kp-modal textarea { width: 100%; box-sizing: border-box; min-height: 45vh; resize: vertical; font: 14px/1.45 sans-serif;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #ddd); border: 1px solid var(--border-color, #555); border-radius: 4px; padding: 8px; }
.kp-modal input { width: 100%; box-sizing: border-box; padding: 6px; }
.kp-buttons { display: flex; gap: 8px; justify-content: flex-end; }
.kp-buttons .primary { background: #2a6fd6; border-color: #2a6fd6; color: #fff; }
.kp-preview { width: 100%; box-sizing: border-box; height: 100%; resize: none; font: 12px/1.4 sans-serif; opacity: .85;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #ddd); border: 1px solid var(--border-color, #555); border-radius: 4px; padding: 4px; }
`;

function el(tag, props = {}, children = []) {
  const e = Object.assign(document.createElement(tag), props);
  for (const c of children) e.append(c);
  return e;
}

/** Small modal. buttons: [{label, value, primary}]; resolves {value, text}. Escape resolves with value null unless noEscape. */
function modal({ title, note, text, input, buttons, noEscape }) {
  return new Promise((resolve) => {
    const field = text !== undefined ? el("textarea", { value: text }) : input !== undefined ? el("input", { value: input }) : null;
    const shade = el("div", { className: "kp-shade" });
    const close = (value) => {
      shade.remove();
      document.removeEventListener("keydown", onKey, true);
      resolve({ value, text: field?.value });
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); if (!noEscape) close(null); }
      else if (e.key === "Enter" && (e.ctrlKey || e.metaKey || field?.tagName === "INPUT")) {
        e.preventDefault(); e.stopPropagation(); close(buttons.find((b) => b.primary)?.value ?? null);
      }
    };
    const box = el("div", { className: "kp-modal" }, [
      el("h3", { textContent: title }),
      ...(note ? [el("p", { textContent: note })] : []),
      ...(field ? [field] : []),
      el("div", { className: "kp-buttons" }, buttons.map((b) =>
        el("button", { textContent: b.label, className: b.primary ? "primary" : "", onclick: () => close(b.value) }))),
    ]);
    shade.append(box);
    document.body.append(shade);
    document.addEventListener("keydown", onKey, true);
    (field ?? box.querySelector(".primary"))?.focus();
    shade.close = close;
    return shade;
  });
}

async function getJSON(path, options) {
  const resp = await api.fetchApi(ROUTE + path, options);
  return { status: resp.status, data: await resp.json().catch(() => ({})) };
}

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

// ------------------------------------------------------------------ KoboldCpp Prompt node

function addPanel(node) {
  const dot = el("span", { className: "kp-dot" });
  const label = el("span", { textContent: "Checking KoboldCpp…" });
  const refresh = el("button", { textContent: "↻", title: "Ask KoboldCpp which model is loaded" });
  const select = el("select", { title: "Saved context files" });
  const load = el("button", { textContent: "Load", title: "Replace the context box with this file" });
  const save = el("button", { textContent: "Save…", title: "Save the context box to a .txt file" });
  const random = el("button", { textContent: "🎲 Random request",
    title: "Set the request to __random__: a new prompt based only on your guidelines (the context box), with camera angle, expression, lighting, setting, clothing and hair picked for you. Each run picks again and avoids the last runs' picks." });
  const status = el("div", { className: "kp-status" }, [dot, label]);
  const panel = el("div", { className: "kp-panel" }, [
    el("div", { className: "kp-row" }, [status, refresh]),
    el("div", { className: "kp-row" }, [select, load, save]),
    el("div", { className: "kp-row" }, [random]),
  ]);
  random.style.flex = "1";
  for (const e of [select, load, save, refresh, random]) e.addEventListener("pointerdown", (ev) => ev.stopPropagation());

  const setStatus = (s) => {
    dot.className = "kp-dot " + (s.ok ? "ok" : "bad");
    status.classList.toggle("warn", !s.ok);
    label.textContent = s.ok
      ? `${s.model}${s.context ? ` · context ${s.context}` : ""}`
      : s.running
        ? "KoboldCpp is running but has no model loaded. Load a text model, then click ↻."
        : `KoboldCpp isn't running at ${s.url ?? "?"}. Start it and load a text model, then click ↻.`;
    status.title = s.ok ? `Connected to ${s.url}` : (s.error ?? "");
  };
  node.kpRefresh = async () => {
    const url = widget(node, "server_url")?.value ?? "";
    try {
      setStatus((await getJSON(`/status?url=${encodeURIComponent(url)}`)).data);
    } catch (e) {
      setStatus({ ok: false, url, error: String(e) });
    }
  };
  node.kpSetModel = (model) => {
    if (model && dot.classList.contains("ok")) label.textContent = label.textContent.replace(/^[^·]*/, model + " ");
  };

  const fillPresets = async (selected) => {
    const { data } = await getJSON("/contexts");
    const names = data.names ?? [];
    select.replaceChildren(...(names.length
      ? names.map((n) => el("option", { value: n, textContent: n }))
      : [el("option", { value: "", textContent: "(no saved contexts)" })]));
    if (selected && names.includes(selected)) select.value = selected;
    select.title = data.folder ? `Saved in ${data.folder}` : "";
  };

  refresh.onclick = () => node.kpRefresh();
  random.onclick = async () => {
    const req = widget(node, "request");
    if (!req || req.value.trim() === "__random__") return;
    if (req.value.trim()) {
      const r = await modal({ title: "Replace the request with a random one?", note: "The request box becomes __random__. Each run then writes a new prompt from your guidelines with picked details.",
        buttons: [{ label: "Cancel", value: null }, { label: "Replace", value: 1, primary: true }] });
      if (!r.value) return;
    }
    req.value = "__random__";
    req.callback?.(req.value);
    node.setDirtyCanvas?.(true, true);
  };
  load.onclick = async () => {
    if (!select.value) return;
    const ctx = widget(node, "context");
    const { status, data } = await getJSON(`/contexts/${encodeURIComponent(select.value)}`);
    if (status !== 200) return modal({ title: "Couldn't load context", note: data.error, buttons: [{ label: "OK", value: 1, primary: true }] });
    if (ctx.value.trim() && ctx.value !== data.text) {
      const r = await modal({ title: `Load "${select.value}"?`, note: "This replaces what's in the context box.", buttons: [{ label: "Cancel", value: null }, { label: "Load", value: 1, primary: true }] });
      if (!r.value) return;
    }
    ctx.value = data.text;
    ctx.callback?.(ctx.value);
    node.setDirtyCanvas?.(true, true);
  };
  save.onclick = async () => {
    const text = widget(node, "context").value;
    const r = await modal({ title: "Save context as", note: "Saved as a .txt file you can also edit in any text editor.", input: select.value || "",
      buttons: [{ label: "Cancel", value: null }, { label: "Save", value: 1, primary: true }] });
    const name = r.text?.trim();
    if (!r.value || !name) return;
    const post = (overwrite) => getJSON("/contexts", { method: "POST", body: JSON.stringify({ name, text, overwrite }) });
    let res = await post(false);
    if (res.status === 409) {
      const c = await modal({ title: `"${name}" already exists`, note: "Overwrite it with the current context box?", buttons: [{ label: "Cancel", value: null }, { label: "Overwrite", value: 1, primary: true }] });
      if (!c.value) return;
      res = await post(true);
    }
    if (res.status !== 200) return modal({ title: "Couldn't save context", note: res.data.error, buttons: [{ label: "OK", value: 1, primary: true }] });
    await fillPresets(res.data.name);
  };

  node.addDOMWidget("kobold_panel", "kobold_panel", panel, {
    serialize: false,
    getValue: () => "",
    setValue: () => {},
    getMinHeight: () => 94,
    getMaxHeight: () => 94,
  });

  const urlWidget = widget(node, "server_url");
  if (urlWidget) {
    const cb = urlWidget.callback;
    let timer;
    urlWidget.callback = function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => node.kpRefresh(), 400);
      return cb?.apply(this, args);
    };
  }
  fillPresets();
  // let a loaded workflow restore server_url before the first check
  setTimeout(() => node.kpRefresh(), 50);
}

// ------------------------------------------------------------------ Review & Edit node

function addPreview(node) {
  const area = el("textarea", { className: "kp-preview", readOnly: true, placeholder: "The prompt you approve shows here." });
  node.addDOMWidget("approved_text", "kobold_preview", area, {
    serialize: false,
    getValue: () => area.value,
    setValue: (v) => { area.value = v ?? ""; },
    getMinHeight: () => 80,
  });
  node.kpShow = (text) => { area.value = text; };
}

const openReviews = new Map(); // node_id -> modal shade

function showReview({ node_id, text }) {
  if (openReviews.has(node_id)) return;
  const node = app.graph?.getNodeById?.(Number(node_id)) ?? app.graph?.getNodeById?.(node_id);
  const title = node?.title && node.title !== "Review & Edit Prompt" ? `Review & Edit Prompt: ${node.title}` : "Review & Edit Prompt";
  const p = modal({
    title,
    note: "Edit the prompt if needed, then Continue (Ctrl+Enter). Cancel stops this run.",
    text,
    buttons: [{ label: "Cancel run", value: "cancel" }, { label: "Continue", value: "continue", primary: true }],
    noEscape: true, // a stray Esc shouldn't throw away the run
  });
  openReviews.set(node_id, document.querySelector(".kp-shade:last-of-type"));
  p.then(async ({ value, text: edited }) => {
    openReviews.delete(node_id);
    if (value === null) value = "cancel";
    await getJSON(`/review/${encodeURIComponent(node_id)}`, { method: "POST", body: JSON.stringify({ action: value, text: edited }) });
  });
}

// ------------------------------------------------------------------ extension

app.registerExtension({
  name: "Talisar.KoboldCppPrompt",

  async setup() {
    document.head.append(el("style", { textContent: CSS }));
    api.addEventListener("kobold_prompt.review", (e) => showReview(e.detail));
    api.addEventListener("kobold_prompt.review_done", (e) => {
      const shade = openReviews.get(e.detail.node_id);
      openReviews.delete(e.detail.node_id);
      shade?.remove();
    });
    api.addEventListener("execution_start", () => {
      for (const n of app.graph?._nodes ?? []) n.kpRefresh?.();
    });
    // a review that started before this page was (re)loaded
    try {
      const { data } = await getJSON("/review");
      for (const p of data.pending ?? []) showReview(p);
    } catch {}
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "KoboldCppPrompt") {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = created?.apply(this, arguments);
        addPanel(this);
        return r;
      };
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        executed?.apply(this, arguments);
        this.kpSetModel?.(message?.model?.[0]);
      };
    } else if (nodeData.name === "ReviewEditPrompt") {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = created?.apply(this, arguments);
        addPreview(this);
        return r;
      };
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        executed?.apply(this, arguments);
        this.kpShow?.(message?.text?.[0] ?? "");
      };
    }
  },
});
