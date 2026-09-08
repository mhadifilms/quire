"use strict";
const $ = (selector, base = document) => base.querySelector(selector);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const state = {
  project: null,
  quality: null,
  page: 1,
  selected: null,
  target: "",
  filter: "review",
  view: "manuscript",
  drafts: new Map(),
  bookDrafts: new Map(),
  selectedAt: Date.now(),
};
const token = $('meta[name="quire-token"]').content;
const names = {
  en: "English",
  fa: "Persian",
  ar: "Arabic",
  ur: "Urdu",
  he: "Hebrew",
  fr: "French",
  de: "German",
  es: "Spanish",
  zh: "Chinese",
  ja: "Japanese",
  und: "Confirm language",
};
const kindNames = {
  paragraph: "Paragraph",
  heading: "Heading",
  footnote: "Footnote",
  quote: "Quotation",
  poetry: "Poetry",
  table: "Table",
  figure: "Illustration",
  caption: "Caption",
};
const languageName = (code) => names[code] || code;
const dir = (code) =>
  /^(ar|fa|ur|he|ps|yi|sd)(-|$)/.test(code) ? "rtl" : "ltr";
let toastTimer;
function toast(message, error = false) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  el.classList.toggle("error", error);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), error ? 9000 : 3500);
}
async function api(path, body) {
  const response = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body
      ? { "Content-Type": "application/json", "X-Quire-Token": token }
      : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}
const endpoint = (action) =>
  `/api/projects/${state.project.id}${action ? "/" + action : ""}`;
function accept(data) {
  state.project = data.project;
  state.quality = data.quality;
}
async function refresh() {
  accept(await api(endpoint() + `?target=${encodeURIComponent(state.target)}`));
}
async function perform(action, body = {}) {
  try {
    const data = await api(endpoint(action), {
      revision: state.project.revision,
      ...body,
    });
    if (data.project) accept(data);
    return data;
  } catch (error) {
    toast(error.message, true);
    return null;
  }
}
function dialog(content) {
  $("#dialog-body").innerHTML = content;
  $("#dialog").showModal();
}
async function library() {
  if (state.project) state.bookDrafts.set(state.project.id, state.drafts);
  state.project = null;
  state.selected = null;
  state.drafts = new Map();
  $("#book-heading").innerHTML = "";
  $("#book-tools").hidden = true;
  $("#toolbar").innerHTML =
    '<span class="muted small">Your local book workspace</span>';
  history.replaceState(null, "", location.pathname);
  try {
    const books = await api("/api/projects");
    $("#main").innerHTML =
      `<section class="library"><div class="library-top"><div><h1>A place for every page.</h1><p class="muted">Bring a book from its original scan to a carefully reviewed edition.</p></div><span class="small muted">${books.length} ${books.length === 1 ? "book" : "books"}</span></div><div class="dropzone" id="dropzone"><div><strong>Start with a PDF</strong><span class="muted">Drop a book here. Quire keeps the original beside your manuscript.</span></div><button class="primary" id="import">Import a book</button></div><div class="shelf">${books.map((book) => `<button class="book" data-book="${esc(book.id)}"><div class="book-cover"><strong>${esc(book.title)}</strong><span class="small muted">${esc(book.author || "Author not set")}</span></div><div class="book-meta"><span>${book.page_count} ${book.page_count === 1 ? "page" : "pages"} · ${esc(languageName(book.language.language))}</span><span>${book.quality.reviewed_passages}/${book.quality.passages} reviewed</span></div></button>`).join("")}</div></section>`;
    $("#import").onclick = () => $("#pdf-file").click();
    document
      .querySelectorAll("[data-book]")
      .forEach((el) => (el.onclick = () => openBook(el.dataset.book)));
    const drop = $("#dropzone");
    drop.ondragover = (e) => {
      e.preventDefault();
      drop.classList.add("drag");
    };
    drop.ondragleave = () => drop.classList.remove("drag");
    drop.ondrop = (e) => {
      e.preventDefault();
      drop.classList.remove("drag");
      if (e.dataTransfer.files[0]) importBook(e.dataTransfer.files[0]);
    };
  } catch (error) {
    toast(error.message, true);
  }
}
async function importBook(file) {
  if (file.size > 150 * 1024 * 1024) {
    toast("Use the import command for PDFs larger than 150 MB", true);
    return;
  }
  dialog(
    `<h2>Import your book</h2><p>${esc(file.name)}</p><label for="import-language">Source language</label><select id="import-language"><option value="auto">Detect automatically</option>${Object.entries(
      names,
    )
      .filter(([c]) => c !== "und")
      .map(([code, name]) => `<option value="${code}">${name}</option>`)
      .join(
        "",
      )}</select><p class="small muted">Recognition runs on this computer. Choose the language for scans without a text layer if you already know it.</p><div class="actions"><button class="primary" id="start-import">Import book</button></div>`,
  );
  $("#start-import").onclick = async () => {
    const language = $("#import-language").value;
    $("#dialog").close();
    try {
      const response = await fetch("/api/import", {
        method: "POST",
        headers: {
          "X-Quire-Token": token,
          "X-Filename": encodeURIComponent(file.name),
          "X-Language": language,
        },
        body: file,
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.error);
      await watchJob(result.job_id, async (result) =>
        openBook(result.project_id),
      );
    } catch (error) {
      toast(error.message, true);
    }
  };
}
async function watchJob(id, done) {
  const activity = $("#activity");
  activity.hidden = false;
  activity.textContent = "Starting…";
  for (;;) {
    let job;
    try {
      job = await api(`/api/jobs/${id}`);
    } catch (error) {
      activity.hidden = true;
      toast(error.message, true);
      return;
    }
    activity.textContent = job.progress || "Working on your book…";
    if (job.state === "complete") {
      activity.hidden = true;
      await done(job.result);
      return;
    }
    if (job.state === "failed") {
      activity.hidden = true;
      toast(job.error, true);
      if (state.project) {
        await refresh();
        renderWorkspace();
      }
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1100));
  }
}
async function openBook(id) {
  try {
    if (state.project) state.bookDrafts.set(state.project.id, state.drafts);
    accept(await api(`/api/projects/${id}`));
    state.page = 1;
    state.selected = null;
    state.target = "";
    state.drafts = state.bookDrafts.get(id) || new Map();
    state.bookDrafts.set(id, state.drafts);
    history.replaceState(null, "", `#book=${encodeURIComponent(id)}`);
    renderWorkspace();
  } catch (error) {
    toast(error.message, true);
  }
}
function nodeText(node) {
  return state.target ? node.translations[state.target]?.text || "" : node.text;
}
function nodeStatus(node) {
  if (node.status === "excluded") return "excluded";
  return state.target
    ? node.translations[state.target]?.status || "pending"
    : node.status;
}
function issues() {
  return (state.quality?.findings || []).filter(
    (f) =>
      state.filter !== "uncertain" ||
      !["unreviewed", "page_unchecked", "translation_review"].includes(f.code),
  );
}
function renderWorkspace() {
  const p = state.project;
  if (!p) return;
  $("#book-tools").hidden = false;
  $("#book-tools").onclick = toolsDialog;
  const nodes = p.nodes.filter((n) => n.page === state.page);
  if (!nodes.some((n) => n.id === state.selected))
    state.selected = nodes[0]?.id || null;
  const page = p.pages.find((p) => p.number === state.page);
  const targets = [
    ...new Set(p.nodes.flatMap((n) => Object.keys(n.translations))),
  ];
  if (state.target && !targets.includes(state.target))
    targets.push(state.target);
  $("#book-heading").innerHTML =
    `<strong>${esc(p.title)}</strong><span>${esc(p.author || "Author not set")} · ${esc(languageName(p.language.language))}</span>`;
  $("#toolbar").innerHTML =
    `<select aria-label="Manuscript language" id="target"><option value="">Original</option>${targets.map((code) => `<option value="${esc(code)}" ${state.target === code ? "selected" : ""}>${esc(languageName(code))}</option>`).join("")}<option value="_new">Add translation…</option></select><button id="details" class="quiet optional">Book details</button><button id="history" class="quiet optional">History</button><button id="translate">Translate</button><button id="publish" class="primary">Export edition</button>`;
  const pct = Math.round(
    (100 * state.quality.reviewed_passages) /
      Math.max(1, state.quality.passages),
  );
  $("#main").innerHTML =
    `<div class="workspace"><aside class="sidebar"><h2>Review progress</h2><div class="small muted">${state.quality.reviewed_passages} of ${state.quality.passages} passages checked</div><div class="progress-track"><span style="width:${pct}%"></span></div><span class="status-badge ${state.quality.ready ? "ready" : ""}">${state.quality.ready ? "Ready for export" : state.quality.review_items + state.quality.errors + " items to review"}</span><div class="page-controls"><button id="prev" aria-label="Previous page">‹</button><input id="page-number" aria-label="Source page" type="number" min="1" max="${p.source.page_count}" value="${state.page}"><button id="next" aria-label="Next page">›</button></div><div class="small muted">of ${p.source.page_count} source ${p.source.page_count === 1 ? "page" : "pages"}</div><select class="issue-filter" id="issue-filter" aria-label="Review queue"><option value="review">All review items</option><option value="uncertain" ${state.filter === "uncertain" ? "selected" : ""}>Uncertain passages first</option></select><div class="issue-list">${
      issues()
        .slice(0, 150)
        .map(
          (f) =>
            `<button class="issue ${f.node_id === state.selected ? "active" : ""}" data-node="${esc(f.node_id || "")}" data-page="${f.page || state.page}"><strong>${f.page ? "Page " + f.page : "Book details"}</strong><${f.severity === "error" ? "em" : "span"}>${esc(f.message)}</${f.severity === "error" ? "em" : "span"}></button>`,
        )
        .join("") ||
      '<p class="small muted">No remaining items in this view.</p>'
    }</div></aside><section class="desk"><div class="deskbar"><div><h2>Page ${state.page} <span>of ${p.source.page_count}</span></h2><span>${page?.reviewed ? "Completeness checked" : "Compare all content on this source page"}</span></div><div class="mobile-view"><button id="mobile-prev" aria-label="Previous page">‹</button><button id="toggle-view">${state.view === "scan" ? "Manuscript" : "Show scan"}</button><button id="mobile-next" aria-label="Next page">›</button></div><button id="review-page" class="page-check" ${page?.reviewed ? "disabled" : ""}>${page?.reviewed ? "Page checked" : "Check completeness"}</button></div><div class="proof-pair"><section class="proof-column ${state.view === "scan" ? "" : "hide-mobile"}"><div class="pane-label"><span>Original scan</span><span>Select a passage to locate its source</span></div><div class="scan-scroll"><div class="scan-page">${page ? `<img id="scan" alt="Original source page ${state.page}" src="${endpoint()}/pages/${state.page}.png">${nodes.map((n) => `<button class="region ${n.id === state.selected ? "selected" : ""}" data-select="${esc(n.id)}" aria-label="Select ${esc(kindNames[n.kind])}: ${esc((n.text || n.alt || "Illustration").slice(0, 65))}" style="left:${(100 * n.bbox[0]) / page.width}%;top:${(100 * n.bbox[1]) / page.height}%;width:${(100 * (n.bbox[2] - n.bbox[0])) / page.width}%;height:${(100 * (n.bbox[3] - n.bbox[1])) / page.height}%"></button>`).join("")}` : '<p class="empty-page">This page has not been extracted yet.</p>'}</div></div></section><section class="proof-column ${state.view === "manuscript" ? "" : "hide-mobile"}"><div class="pane-label"><span>${state.target ? esc(languageName(state.target)) + " translation" : "Working manuscript"}</span><span>Changes save to this book</span></div><div class="manuscript-scroll">${nodes.map(renderPassage).join("") || `<div class="empty-page"><h3>${page?.state === "failed" ? "This page needs another extraction" : "No text on this page"}</h3><p>${esc(page?.error || "Compare the scan, then check completeness if this is a blank or decorative page.")}</p></div>`}</div></section></div></section></div>`;
  $("#target").onchange = async (e) => {
    if (e.target.value === "_new") {
      translationDialog();
      return;
    }
    state.target = e.target.value;
    await refresh();
    renderWorkspace();
  };
  $("#details").onclick = detailsDialog;
  $("#history").onclick = historyDialog;
  $("#translate").onclick = translationDialog;
  $("#publish").onclick = publishDialog;
  $("#prev").onclick = () => changePage(-1);
  $("#next").onclick = () => changePage(1);
  $("#mobile-prev").onclick = () => changePage(-1);
  $("#mobile-next").onclick = () => changePage(1);
  $("#page-number").onchange = (e) => goPage(Number(e.target.value));
  $("#toggle-view").onclick = () => {
    state.view = state.view === "scan" ? "manuscript" : "scan";
    renderWorkspace();
  };
  $("#issue-filter").onchange = (e) => {
    state.filter = e.target.value;
    renderWorkspace();
  };
  $("#review-page").onclick = async () => {
    if (await perform("review-page", { page: state.page })) {
      await refresh();
      renderWorkspace();
      toast("Page completeness checked");
    }
  };
  document.querySelectorAll("[data-select]").forEach(
    (el) =>
      (el.onclick = (e) => {
        if (e.target.closest(".edit-panel")) return;
        selectNode(el.dataset.select);
      }),
  );
  document.querySelectorAll(".issue").forEach(
    (el) =>
      (el.onclick = () => {
        state.page = Number(el.dataset.page);
        state.selected = el.dataset.node || null;
        state.selectedAt = Date.now();
        renderWorkspace();
        scrollSelected();
      }),
  );
  wireEditor();
  wireRecovery();
}
function renderPassage(node) {
  const selected = node.id === state.selected;
  const text = nodeText(node);
  const draft = state.drafts.get(node.id + ":" + state.target);
  const status = nodeStatus(node);
  const warnings = node.warnings || [];
  let content =
    esc(text) ||
    '<span class="muted">' +
      (node.kind === "figure"
        ? "Illustration"
        : state.target
          ? "Awaiting translation"
          : "Empty passage") +
      "</span>";
  const cells = state.target
    ? node.translations[state.target]?.cells
    : node.cells;
  if (node.kind === "table" && cells)
    content = `<table class="table-preview">${cells.map((row) => `<tr>${row.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</table>`;
  return `<article class="passage ${esc(node.kind)} ${selected ? "selected" : ""} ${node.status === "excluded" ? "excluded" : ""}" dir="${dir(state.target || node.language)}" data-select="${esc(node.id)}" tabindex="0" aria-label="${esc(kindNames[node.kind])}"><div class="passage-meta" dir="ltr"><span>${esc(kindNames[node.kind])}${warnings.length ? " · " + esc(warnings[0]) : ""}</span><span class="review-state ${status === "reviewed" ? "done" : ""}">${status === "reviewed" ? "Reviewed" : node.status === "excluded" ? "Excluded" : "Needs review"}</span></div><div class="body">${content}</div>${
    selected
      ? `<div class="edit-panel" dir="ltr"><label class="small muted" for="passage-text">${state.target ? "Translation" : "Passage text"}</label><textarea id="passage-text" dir="${dir(state.target || node.language)}" spellcheck="false">${esc(draft?.text ?? text)}</textarea>${node.kind === "table" ? `<div class="table-editor"><table>${(draft?.cells || cells || [[""]]).map((row, r) => `<tr>${row.map((c, col) => `<td><input data-cell="${r},${col}" aria-label="Row ${r + 1}, column ${col + 1}" value="${esc(c)}"></td>`).join("")}</tr>`).join("")}</table></div><div class="actions"><button id="add-row">Add row</button><button id="add-column">Add column</button></div>` : ""}<div class="actions"><button class="primary" id="approve">Approve passage</button><button id="save">Save changes</button><button class="quiet" id="next-passage">Next</button><button class="quiet" id="discard-draft">Discard unsaved edits</button></div><details class="edit-options"><summary>Structure, notes and source</summary>${
          !state.target
            ? `<label>Passage type<select id="kind">${Object.entries(kindNames)
                .map(
                  ([k, v]) =>
                    `<option value="${k}" ${(draft?.kind || node.kind) === k ? "selected" : ""}>${v}</option>`,
                )
                .join(
                  "",
                )}</select></label><label>Heading level<input id="level" type="number" min="1" max="6" value="${draft?.level || node.level || 2}"></label><label>Passage language<input id="node-language" value="${esc(draft?.language || node.language)}"></label><label>Linked footnotes<select id="note-ids" multiple>${state.project.nodes
                .filter((n) => n.kind === "footnote" && n.id !== node.id)
                .map(
                  (n) =>
                    `<option value="${esc(n.id)}" ${(draft?.note_ids || node.note_ids).includes(n.id) ? "selected" : ""}>Page ${n.page}: ${esc(n.text.slice(0, 65))}</option>`,
                )
                .join(
                  "",
                )}</select></label><label>Illustration description<textarea id="alt">${esc(draft?.alt ?? node.alt ?? "")}</textarea></label><div class="actions"><button id="move-up">Move earlier</button><button id="move-down">Move later</button><button id="join-next">Join with next</button></div><label>Reason for exclusion<input id="exclude-reason" value="${esc(draft?.exclusion_reason ?? node.exclusion_reason ?? "")}" placeholder="For example: repeated running header"></label><div class="actions"><button id="exclude" class="danger">Exclude passage</button><button id="restore">Restore passage</button></div>`
            : ""
        }<label>Editorial note for an unreadable source<textarea id="uncertainty">${esc(draft?.uncertainty_note ?? (state.target ? node.translations[state.target] : node)?.uncertainty_note ?? "")}</textarea></label><p class="small muted">This note is included in the edition. Preserve uncertainty when the original cannot be read.</p><strong class="small">Original extraction</strong><div class="original-text" dir="${dir(node.language)}">${esc(node.source_regions?.map((r) => r.original).join("\n") || node.original)}</div></details></div>`
      : ""
  }</article>`;
}
function selectNode(id) {
  state.selected = id;
  state.selectedAt = Date.now();
  renderWorkspace();
  scrollSelected();
}
function scrollSelected() {
  const el = $(".passage.selected");
  if (el) el.scrollIntoView({ block: "nearest" });
}
function goPage(page) {
  state.page = Math.min(
    state.project.source.page_count,
    Math.max(1, page || 1),
  );
  state.selected = null;
  renderWorkspace();
}
function changePage(offset) {
  goPage(state.page + offset);
}
function nextPassage() {
  const nodes = state.project.nodes.filter((n) => n.status !== "excluded");
  const index = nodes.findIndex((n) => n.id === state.selected);
  const next = nodes[index + 1];
  if (next) {
    state.page = next.page;
    selectNode(next.id);
  } else toast("You reached the last passage");
}
function collectDraft() {
  const node = state.project.nodes.find((n) => n.id === state.selected);
  if (!node || !$("#passage-text")) return {};
  const previous = state.drafts.get(node.id + ":" + state.target);
  const draft = {
    text: $("#passage-text").value,
    uncertainty_note: $("#uncertainty").value,
    base: previous?.base || JSON.stringify(node),
  };
  if (!state.target)
    Object.assign(draft, {
      kind: $("#kind").value,
      level: Number($("#level").value),
      language: $("#node-language").value,
      alt: $("#alt").value,
      note_ids: [...$("#note-ids").selectedOptions].map((o) => o.value),
      exclusion_reason: $("#exclude-reason").value,
    });
  const cellInputs = [...document.querySelectorAll("[data-cell]")];
  if (cellInputs.length) {
    draft.cells = [];
    cellInputs.forEach((input) => {
      const [r, c] = input.dataset.cell.split(",").map(Number);
      draft.cells[r] ??= [];
      draft.cells[r][c] = input.value;
    });
  }
  state.drafts.set(node.id + ":" + state.target, draft);
  $("#discard-draft").hidden = false;
  return draft;
}
async function savePassage(status) {
  const { base, ...draft } = collectDraft();
  const node = state.project.nodes.find((n) => n.id === state.selected);
  if (base !== JSON.stringify(node)) {
    toast(
      "This passage changed since you started editing. Copy your draft, then discard it to review the latest version.",
      true,
    );
    return;
  }
  const changes = {
    ...draft,
    review_seconds: Math.min(3600, (Date.now() - state.selectedAt) / 1000),
    uncertainty_note: $("#uncertainty").value,
  };
  if (!state.target) {
    Object.assign(changes, {
      kind: $("#kind").value,
      level: Number($("#level").value),
      language: $("#node-language").value,
      alt: $("#alt").value,
      note_ids: [...$("#note-ids").selectedOptions].map((o) => o.value),
    });
    if (status === "excluded")
      changes.exclusion_reason = $("#exclude-reason").value;
  }
  if (status) changes.status = status;
  const id = state.selected;
  const result = await perform("edit", {
    node_id: id,
    changes,
    target: state.target || null,
  });
  if (result) {
    state.drafts.delete(id + ":" + state.target);
    state.selectedAt = Date.now();
    await refresh();
    renderWorkspace();
    toast(status === "reviewed" ? "Passage approved" : "Changes saved");
  }
}
function wireEditor() {
  if (!state.selected) return;
  $("#discard-draft").hidden = !state.drafts.has(
    state.selected + ":" + state.target,
  );
  $("#passage-text").oninput = collectDraft;
  document
    .querySelectorAll(
      "[data-cell], .edit-options input, .edit-options textarea, .edit-options select",
    )
    .forEach((el) => (el.oninput = collectDraft));
  $("#discard-draft").onclick = () => {
    state.drafts.delete(state.selected + ":" + state.target);
    renderWorkspace();
  };
  $("#save").onclick = () => savePassage();
  $("#approve").onclick = () => savePassage("reviewed");
  $("#next-passage").onclick = nextPassage;
  if ($("#exclude")) {
    $("#exclude").onclick = () => savePassage("excluded");
    $("#restore").onclick = () => savePassage("pending");
    $("#move-up").onclick = () => move(-1);
    $("#move-down").onclick = () => move(1);
    $("#join-next").onclick = async () => {
      if (state.drafts.has(state.selected + ":" + state.target)) {
        toast("Save your edits before joining passages.");
        return;
      }
      if (await perform("join", { node_id: state.selected })) {
        renderWorkspace();
        toast("Passages joined. Review the combined text and its translation.");
      }
    };
  }
  if ($("#add-row")) {
    $("#add-row").onclick = () => {
      const draft = collectDraft();
      draft.cells.push(Array(draft.cells[0].length).fill(""));
      renderWorkspace();
    };
    $("#add-column").onclick = () => {
      const draft = collectDraft();
      draft.cells.forEach((row) => row.push(""));
      renderWorkspace();
    };
  }
  document.querySelectorAll(".passage").forEach(
    (el) =>
      (el.onkeydown = (e) => {
        if (e.target === el && ["Enter", " "].includes(e.key)) {
          e.preventDefault();
          selectNode(el.dataset.select);
        }
      }),
  );
}
async function move(offset) {
  if (await perform("move", { node_id: state.selected, offset }))
    renderWorkspace();
}
function detailsDialog() {
  const p = state.project;
  dialog(
    `<h2>Book details</h2><label for="title">Title</label><input id="title" value="${esc(p.title)}"><label for="author">Author</label><input id="author" value="${esc(p.author)}"><label for="language">Source language code</label><input id="language" value="${esc(p.language.language)}"><p class="small muted">For example: fa for Persian, ar for Arabic, en for English. An explicit selection confirms the detected language.</p><div class="actions"><button class="primary" id="save-details">Save details</button></div>`,
  );
  $("#save-details").onclick = async () => {
    const result = await perform("metadata", {
      changes: {
        title: $("#title").value,
        author: $("#author").value,
        language: $("#language").value,
      },
    });
    if (result) {
      $("#dialog").close();
      renderWorkspace();
    }
  };
}
function toolsDialog() {
  const q = state.quality;
  dialog(
    `<h2>Book tools</h2><p>${q.reviewed_passages} of ${q.passages} passages reviewed. ${q.review_items + q.errors} items need attention.</p><div class="actions"><button id="mobile-details">Book details</button><button id="mobile-history">Edit history</button></div><label for="jump-page">Go to source page</label><input id="jump-page" type="number" min="1" max="${state.project.source.page_count}" value="${state.page}"><button id="jump">Go to page</button><h3>Review queue</h3><div class="issue-list">${
      issues()
        .slice(0, 200)
        .map(
          (f, i) =>
            `<button data-review-item="${i}">Page ${f.page || state.page}: ${esc(f.message)}</button>`,
        )
        .join("") || "No remaining review items."
    }</div>`,
  );
  $("#mobile-details").onclick = () => {
    $("#dialog").close();
    detailsDialog();
  };
  $("#mobile-history").onclick = () => {
    $("#dialog").close();
    historyDialog();
  };
  $("#jump").onclick = () => {
    const page = Number($("#jump-page").value);
    $("#dialog").close();
    goPage(page);
  };
  document.querySelectorAll("[data-review-item]").forEach(
    (button) =>
      (button.onclick = () => {
        const finding = issues()[Number(button.dataset.reviewItem)];
        $("#dialog").close();
        if (!finding.page) {
          detailsDialog();
          return;
        }
        state.page = finding.page;
        state.selected = finding.node_id;
        state.view = "manuscript";
        renderWorkspace();
        scrollSelected();
      }),
  );
}
function historyDialog() {
  const events = state.project.history.slice().reverse();
  dialog(
    `<h2>Edit history</h2><p class="muted">Every edit stays attached to its source passage. Undo restores the previous saved version.</p><div class="history-list">${
      events
        .slice(0, 60)
        .map(
          (e) =>
            `<div class="history-entry"><strong>${esc(e.action.replaceAll("_", " "))}${e.undone ? " · undone" : ""}</strong><span class="muted"> ${esc(new Date(e.at).toLocaleString())}</span>${Object.values(
              e.after,
            )
              .filter((n) => n && typeof n === "object" && "text" in n)
              .map((n) => `<p>${esc(n.text)}</p>`)
              .join("")}</div>`,
        )
        .join("") || '<p class="muted">Edits will appear here as you work.</p>'
    }</div><div class="actions"><button id="undo">Undo last edit</button></div>`,
  );
  $("#undo").onclick = async () => {
    if (await perform("undo")) {
      $("#dialog").close();
      await refresh();
      renderWorkspace();
      toast("Previous version restored");
    }
  };
}
function translationDialog() {
  dialog(
    `<h2>Translate this book</h2><p class="muted">Keep each translated passage connected to its original, with shared terminology across chapters.</p><label for="translation-language">Target language</label><select id="translation-language">${Object.entries(
      names,
    )
      .filter(([c]) => c !== "und")
      .map(
        ([code, name]) =>
          `<option value="${code}" ${(state.target || "en") === code ? "selected" : ""}>${name}</option>`,
      )
      .join(
        "",
      )}</select><label for="glossary">Terminology</label><textarea id="glossary" rows="5" placeholder="One source term = preferred translation per line"></textarea><p class="small muted">One source term = preferred translation per line. Save terminology before translating.</p><div class="actions"><button id="save-glossary">Save terminology</button><button id="manual-translation">Write translation</button></div><hr><p class="small muted">Automatic translation sends book text and terminology to Gemini using the configured API key. Completed passages are saved as drafts for review. Each run is limited to 100,000 tokens.</p><div class="actions"><button class="primary" id="start-translation">Translate remaining passages</button></div>`,
  );
  const fill = () => {
    $("#glossary").value = Object.entries(
      state.project.glossaries[$("#translation-language").value] || {},
    )
      .map(([k, v]) => k + " = " + v)
      .join("\n");
  };
  fill();
  $("#translation-language").onchange = fill;
  $("#save-glossary").onclick = async () => {
    const terms = {};
    for (const line of $("#glossary").value.split("\n")) {
      if (!line.trim()) continue;
      const split = line.indexOf("=");
      if (split < 1) {
        toast("Write each term as source = translation", true);
        return;
      }
      terms[line.slice(0, split).trim()] = line.slice(split + 1).trim();
    }
    if (
      await perform("glossary", {
        target: $("#translation-language").value,
        terms,
      })
    )
      toast("Terminology saved");
  };
  $("#manual-translation").onclick = async () => {
    state.target = $("#translation-language").value;
    $("#dialog").close();
    await refresh();
    renderWorkspace();
  };
  $("#start-translation").onclick = async () => {
    const target = $("#translation-language").value;
    const result = await perform("translate", { target });
    if (result) {
      $("#dialog").close();
      await watchJob(result.job_id, async () => {
        state.target = target;
        await refresh();
        renderWorkspace();
        toast("Translation run saved; review the translated passages");
      });
    }
  };
}
function publishDialog() {
  dialog(
    `<h2>Make an edition</h2><p class="muted">Every format uses the same saved manuscript, notes and corrections.</p><label for="template">Reading format</label><select id="template"><option value="reading">Reading · A5</option><option value="study">Study · A4</option><option value="large-print">Large print · A4</option></select><div class="check-row"><label><input type="checkbox" id="format-pdf" checked> PDF</label><label><input type="checkbox" id="format-epub" checked> EPUB</label><label><input type="checkbox" id="format-html" checked> Web</label><label><input type="checkbox" id="format-markdown" checked> Markdown</label><label><input type="checkbox" id="format-text" checked> Plain text</label></div>${state.target ? '<label><input type="checkbox" id="bilingual"> Include the original beside the translation</label>' : ""}<label><input type="checkbox" id="draft" ${state.quality.ready ? "" : "checked"}> Export a draft for review</label><p class="small muted">${state.quality.ready ? "All manuscript checks are complete." : `${state.quality.review_items + state.quality.errors} items still need attention. Drafts remain clearly labeled.`} EPUB and accessibility validation results are included with the edition.</p><div class="actions"><button class="primary" id="export">Create edition</button></div><div class="downloads">${state.project.releases
      .slice(-3)
      .reverse()
      .map(
        (r) =>
          `<a href="${endpoint()}/${r.directory}/release.json" target="_blank" rel="noopener">${esc(r.state.replaceAll("_", " "))} · ${esc(new Date(r.at).toLocaleString())} · validation report</a>`,
      )
      .join("")}</div>`,
  );
  $("#export").onclick = async () => {
    const result = await perform("publish", {
      target: state.target || null,
      bilingual: $("#bilingual")?.checked || false,
      template: $("#template").value,
      draft: $("#draft").checked,
      formats: ["pdf", "epub", "html", "markdown", "text"].filter(
        (f) => $("#format-" + f).checked,
      ),
    });
    if (result) {
      $("#dialog").close();
      await watchJob(result.job_id, async (release) => {
        await refresh();
        renderWorkspace();
        dialog(
          `<h2>Your edition is ready to open</h2><p>${release.state === "draft" ? "Draft for review" : release.state === "validation_pending" ? "Manuscript reviewed. Some external validators are unavailable; see the report." : "Manuscript and available automated checks passed."}</p><div class="downloads">${Object.entries(
            release.files,
          )
            .map(
              ([format, file]) =>
                `<a href="${endpoint()}/${release.directory}/${file}" target="_blank" rel="noopener">Open ${esc(format.toUpperCase())}</a>`,
            )
            .join(
              "",
            )}<a href="${endpoint()}/${release.directory}/release.json" target="_blank" rel="noopener">View validation report</a></div>`,
        );
      });
    }
  };
}
$("#home").onclick = library;
$("#pdf-file").onchange = (e) => {
  if (e.target.files[0]) importBook(e.target.files[0]);
  e.target.value = "";
};
document.addEventListener("keydown", (e) => {
  if (
    (e.metaKey || e.ctrlKey) &&
    e.key === "s" &&
    state.project &&
    $("#passage-text")
  ) {
    e.preventDefault();
    savePassage();
  }
  if (e.key === "Escape" && $("#dialog").open) $("#dialog").close();
});
window.addEventListener("beforeunload", (event) => {
  if (
    state.drafts.size ||
    [...state.bookDrafts.values()].some((drafts) => drafts.size)
  ) {
    event.preventDefault();
    event.returnValue = "";
  }
});
const initial = new URLSearchParams(location.hash.slice(1)).get("book");
if (initial) openBook(initial);
else library();

function wireRecovery() {
  if (state.target) return;
  const label = document.querySelector(".proof-column:last-child .pane-label");
  if (label) {
    const button = document.createElement("button");
    button.className = "quiet small";
    button.textContent = "Add missing passage";
    button.id = "add-passage";
    button.onclick = beginRegion;
    label.lastElementChild.replaceWith(button);
  }
  const page = state.project.pages.find((p) => p.number === state.page);
  if (page?.state === "failed") {
    const box = document.querySelector(".empty-page");
    if (box) {
      const button = document.createElement("button");
      button.textContent = "Retry unfinished pages";
      button.onclick = async () => {
        const job = await perform("retry");
        if (job)
          await watchJob(job.job_id, async () => {
            await refresh();
            renderWorkspace();
          });
      };
      box.append(button);
    }
  }
}
function beginRegion() {
  state.view = "scan";
  renderWorkspace();
  const scan = document.querySelector(".scan-page");
  if (!scan) return;
  const picker = document.createElement("div");
  picker.className = "region-picker";
  picker.setAttribute("aria-label", "Draw a region on the scan");
  const cancel = document.createElement("button");
  cancel.textContent = "Cancel selection";
  cancel.className = "cancel-region";
  cancel.onclick = () => picker.remove();
  picker.append(cancel);
  scan.append(picker);
  toast("Drag a box around the missing text or illustration.");
  let start, box;
  picker.onpointerdown = (e) => {
    if (e.target === cancel) return;
    e.preventDefault();
    picker.setPointerCapture(e.pointerId);
    const r = picker.getBoundingClientRect();
    start = [e.clientX - r.left, e.clientY - r.top];
    box = document.createElement("div");
    box.className = "drawn-region";
    picker.append(box);
  };
  picker.onpointermove = (e) => {
    if (!start) return;
    const r = picker.getBoundingClientRect();
    const end = [
      Math.max(0, Math.min(r.width, e.clientX - r.left)),
      Math.max(0, Math.min(r.height, e.clientY - r.top)),
    ];
    Object.assign(box.style, {
      left: Math.min(start[0], end[0]) + "px",
      top: Math.min(start[1], end[1]) + "px",
      width: Math.abs(start[0] - end[0]) + "px",
      height: Math.abs(start[1] - end[1]) + "px",
    });
  };
  picker.onpointerup = (e) => {
    if (!start) return;
    const r = picker.getBoundingClientRect();
    const end = [
      Math.max(0, Math.min(r.width, e.clientX - r.left)),
      Math.max(0, Math.min(r.height, e.clientY - r.top)),
    ];
    const page = state.project.pages.find((p) => p.number === state.page);
    const bbox = [
      (Math.min(start[0], end[0]) / r.width) * page.width,
      (Math.min(start[1], end[1]) / r.height) * page.height,
      (Math.max(start[0], end[0]) / r.width) * page.width,
      (Math.max(start[1], end[1]) / r.height) * page.height,
    ];
    start = null;
    if (bbox[2] - bbox[0] < 3 || bbox[3] - bbox[1] < 3) return;
    picker.remove();
    dialog(
      `<h2>Recover a source passage</h2><p class="muted">This passage will retain the region you selected on page ${state.page}.</p><label for="new-kind">Passage type</label><select id="new-kind">${Object.entries(
        kindNames,
      )
        .map(([k, v]) => `<option value="${k}">${v}</option>`)
        .join(
          "",
        )}</select><label for="new-text">Text or illustration description</label><textarea id="new-text" rows="5" dir="auto"></textarea><div class="actions"><button class="primary" id="create-passage">Add passage</button></div>`,
    );
    document.querySelector("#create-passage").onclick = async () => {
      const previous = new Set(state.project.nodes.map((n) => n.id));
      const result = await perform("add", {
        page: state.page,
        bbox,
        text: document.querySelector("#new-text").value,
        kind: document.querySelector("#new-kind").value,
      });
      if (result) {
        state.selected = state.project.nodes.find(
          (n) => !previous.has(n.id),
        )?.id;
        document.querySelector("#dialog").close();
        state.view = "manuscript";
        renderWorkspace();
        scrollSelected();
        toast(
          "Source passage recovered. Review it and place it in reading order.",
        );
      }
    };
  };
}
