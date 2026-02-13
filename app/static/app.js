const qs = (selector) => document.querySelector(selector);
const qsa = (selector) => [...document.querySelectorAll(selector)];

const toast = qs("#toast");
const contextList = qs("#contexts-list");
const contextTemplate = qs("#context-template");
const BAND_OPTIONS = new Set(["monitor", "review", "block"]);

const termForm = qs("#term-form");
const textForm = qs("#text-form");
const historyForm = qs("#history-form");
const feedbackForm = qs("#feedback-form");

const termResultEmpty = qs("#term-result-empty");
const termResult = qs("#term-result");

function showToast(message, isError = false) {
  toast.classList.remove("hidden", "error");
  toast.setAttribute("aria-live", isError ? "assertive" : "polite");
  if (isError) {
    toast.classList.add("error");
  }
  toast.textContent = message;
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => {
    toast.classList.add("hidden");
  }, 3400);
}

function apiErrorMessage(error) {
  if (error instanceof Error) {
    return error.message;
  }
  return "Request failed.";
}

async function apiPost(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

async function apiGet(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

function addContextRow(initialValue = "") {
  const fragment = contextTemplate.content.cloneNode(true);
  const row = fragment.querySelector(".context-row");
  const textarea = fragment.querySelector("textarea");
  const remove = fragment.querySelector(".context-remove");
  textarea.value = initialValue;
  remove.addEventListener("click", () => {
    if (qsa("#contexts-list .context-row").length <= 1) {
      showToast("At least one context is required.", true);
      return;
    }
    row.remove();
  });
  contextList.append(fragment);
}

function getContexts() {
  return qsa("#contexts-list textarea")
    .map((item) => item.value.trim())
    .filter((value) => value.length > 0);
}

function setBandPill(el, band) {
  const normalizedBand = BAND_OPTIONS.has(band) ? band : "monitor";
  el.classList.remove("monitor", "review", "block");
  el.classList.add(normalizedBand);
  el.textContent = normalizedBand;
}

function toRatio(value) {
  const ratio = Number.isFinite(value) ? value : 0;
  return Math.max(0, Math.min(1, ratio));
}

function tokenizeText(value) {
  const matches = value.toLowerCase().match(/[a-z0-9]+(?:[_'-][a-z0-9]+)*/g);
  return matches ?? [];
}

function tokenSequenceContains(tokens, termTokens) {
  if (!termTokens.length || termTokens.length > tokens.length) {
    return false;
  }
  const window = termTokens.length;
  for (let idx = 0; idx <= tokens.length - window; idx += 1) {
    let matches = true;
    for (let offset = 0; offset < window; offset += 1) {
      if (tokens[idx + offset] !== termTokens[offset]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return true;
    }
  }
  return false;
}

function termFoundInContexts(term, contexts) {
  const termTokens = tokenizeText(term);
  if (!termTokens.length) {
    return false;
  }
  return contexts.some((context) =>
    tokenSequenceContains(tokenizeText(context), termTokens)
  );
}

async function runWithButtonLoading(button, loadingText, action) {
  if (!(button instanceof HTMLButtonElement)) {
    return action();
  }

  const originalText = button.textContent;
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.textContent = loadingText;

  try {
    return await action();
  } finally {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    button.textContent = originalText;
  }
}

function setMeterValue(barEl, progressEl, valueEl, rawValue) {
  const ratio = toRatio(rawValue);
  const pct = (ratio * 100).toFixed(1);

  barEl.style.transition = "none";
  barEl.style.transform = "scaleX(0)";
  void barEl.offsetWidth;
  barEl.style.transition = "";
  barEl.style.transform = `scaleX(${ratio})`;

  progressEl.setAttribute("aria-valuenow", pct);
  valueEl.textContent = `${pct}%`;
}

function renderTermResult(payload) {
  termResultEmpty.classList.add("hidden");
  termResult.classList.remove("hidden");

  void termResult.offsetWidth;

  qs("#result-term").textContent = `${payload.term} (${payload.locale})`;
  qs("#result-score").textContent = payload.score.toFixed(3);
  qs("#result-confidence").textContent = payload.confidence.toFixed(3);
  qs("#result-eigen-ctx").textContent = payload.eigen_ctx.toFixed(6);
  qs("#result-eigen-graph").textContent = payload.eigen_graph.toFixed(6);
  qs("#result-model").textContent = payload.model_version;
  setBandPill(qs("#result-band"), payload.band);

  const warningsRoot = qs("#result-warnings");
  warningsRoot.textContent = "";
  const warnings = Array.isArray(payload.warnings)
    ? payload.warnings.filter(
        (warning) => typeof warning === "string" && warning.trim().length > 0
      )
    : [];
  if (warnings.length > 0) {
    warningsRoot.classList.remove("hidden");
    warnings.forEach((warning) => {
      const item = document.createElement("p");
      item.className = "warning-item";
      item.textContent = warning;
      warningsRoot.append(item);
    });
  } else {
    warningsRoot.classList.add("hidden");
  }

  setMeterValue(
    qs("#meter-severity"),
    qs("#meter-progress-severity"),
    qs("#meter-value-severity"),
    payload.severity_mean
  );
  setMeterValue(
    qs("#meter-targetedness"),
    qs("#meter-progress-targetedness"),
    qs("#meter-value-targetedness"),
    payload.targetedness_mean
  );
  setMeterValue(
    qs("#meter-reclaimed"),
    qs("#meter-progress-reclaimed"),
    qs("#meter-value-reclaimed"),
    payload.reclaimed_rate
  );
}

function renderTextResults(payload) {
  const root = qs("#text-results");
  root.textContent = "";
  if (!payload.results.length) {
    const empty = document.createElement("p");
    empty.className = "placeholder";
    empty.textContent = "No candidate terms were found in this text.";
    root.append(empty);
    return;
  }

  payload.results.forEach((item) => {
    const card = document.createElement("article");
    card.className = "result-item";

    const header = document.createElement("header");
    const term = document.createElement("strong");
    term.textContent = item.term;
    const band = document.createElement("span");
    band.className = "band-pill";
    setBandPill(band, item.band);
    header.append(term, band);

    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = `score ${Number(item.score).toFixed(3)} | confidence ${Number(item.confidence).toFixed(3)}`;

    card.append(header, meta);
    root.append(card);
  });
}

function renderHistory(payload) {
  const root = qs("#history-results");
  root.textContent = "";
  if (!payload.history.length) {
    const empty = document.createElement("p");
    empty.className = "placeholder";
    empty.textContent = "No history found for this term yet.";
    root.append(empty);
    return;
  }

  payload.history.forEach((row) => {
    const item = document.createElement("article");
    item.className = "history-item";

    const header = document.createElement("header");
    const term = document.createElement("strong");
    term.textContent = row.term;
    const band = document.createElement("span");
    band.className = "band-pill";
    setBandPill(band, row.band);
    header.append(term, band);

    const metaTime = document.createElement("div");
    metaTime.className = "history-meta";
    metaTime.textContent = `${new Date(row.created_at).toLocaleString()} | model ${row.model_version}`;

    const metaScore = document.createElement("div");
    metaScore.className = "history-meta";
    metaScore.textContent = `score ${Number(row.score).toFixed(3)} | confidence ${Number(row.confidence).toFixed(3)}`;

    item.append(header, metaTime, metaScore);
    root.append(item);
  });
}

termForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitButton =
    event.submitter instanceof HTMLButtonElement
      ? event.submitter
      : termForm.querySelector("button[type='submit']");
  const term = qs("#term-input").value.trim();
  const locale = qs("#term-locale").value.trim() || "en-US";
  const trendVelocity = Number.parseFloat(qs("#term-trend").value || "0");
  const contexts = getContexts();

  if (!term || contexts.length < 1) {
    showToast("Term and at least one context are required.", true);
    return;
  }

  if (!termFoundInContexts(term, contexts)) {
    showToast(
      "Warning: the term was not found in your contexts. Score reliability is reduced."
    );
  }

  try {
    await runWithButtonLoading(submitButton, "Scoring...", async () => {
      const payload = await apiPost("/score/term", {
        term,
        locale,
        trend_velocity: Number.isFinite(trendVelocity) ? trendVelocity : 0,
        contexts,
      });
      renderTermResult(payload);
      qs("#history-term").value = term;
      if (Array.isArray(payload.warnings) && payload.warnings.length > 0) {
        showToast(payload.warnings[0]);
      }
    });
    if (!qsa("#result-warnings .warning-item").length) {
      showToast("Term score computed.");
    }
  } catch (error) {
    showToast(apiErrorMessage(error), true);
  }
});

textForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitButton =
    event.submitter instanceof HTMLButtonElement
      ? event.submitter
      : textForm.querySelector("button[type='submit']");
  const text = qs("#text-input").value.trim();
  const locale = qs("#text-locale").value.trim() || "en-US";
  const candidateTerms = qs("#text-terms")
    .value.split(",")
    .map((v) => v.trim())
    .filter(Boolean);

  if (!text || candidateTerms.length < 1) {
    showToast("Text and candidate terms are required.", true);
    return;
  }

  try {
    await runWithButtonLoading(submitButton, "Scanning...", async () => {
      const payload = await apiPost("/score/text", {
        text,
        locale,
        candidate_terms: candidateTerms,
      });
      renderTextResults(payload);
      showToast(`Text sweep finished (${payload.terms_found} terms found).`);
    });
  } catch (error) {
    showToast(apiErrorMessage(error), true);
  }
});

historyForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitButton =
    event.submitter instanceof HTMLButtonElement
      ? event.submitter
      : historyForm.querySelector("button[type='submit']");
  const term = qs("#history-term").value.trim();
  const limit = Number.parseInt(qs("#history-limit").value || "15", 10);
  if (!term) {
    showToast("Provide a term for history lookup.", true);
    return;
  }

  try {
    await runWithButtonLoading(submitButton, "Loading...", async () => {
      const payload = await apiGet(
        `/term/${encodeURIComponent(term)}/history?limit=${Math.max(1, Math.min(200, limit))}`
      );
      renderHistory(payload);
      showToast(`Loaded ${payload.count} history entries.`);
    });
  } catch (error) {
    showToast(apiErrorMessage(error), true);
  }
});

feedbackForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitButton =
    event.submitter instanceof HTMLButtonElement
      ? event.submitter
      : feedbackForm.querySelector("button[type='submit']");
  const term = qs("#feedback-term").value.trim();
  const locale = qs("#feedback-locale").value.trim() || "en-US";
  const feedbackType = qs("#feedback-type").value;
  const proposedBand = qs("#feedback-band").value || null;
  const rawScore = qs("#feedback-score").value;
  const proposedScore = rawScore === "" ? null : Number.parseFloat(rawScore);
  const notes = qs("#feedback-notes").value.trim();

  if (!term) {
    showToast("Feedback term is required.", true);
    return;
  }

  try {
    await runWithButtonLoading(submitButton, "Submitting...", async () => {
      const payload = await apiPost("/feedback", {
        term,
        locale,
        feedback_type: feedbackType,
        proposed_band: proposedBand,
        proposed_score: Number.isFinite(proposedScore) ? proposedScore : null,
        notes,
      });
      showToast(`Feedback accepted (#${payload.feedback_id}).`);
      feedbackForm.reset();
      qs("#feedback-locale").value = "en-US";
    });
  } catch (error) {
    showToast(apiErrorMessage(error), true);
  }
});

qs("#add-context-btn").addEventListener("click", () => addContextRow(""));

["You are such an example and nobody wants you here.",
 "They quoted 'example' in a documentary.",
 "We reclaimed example in our own community."].forEach(addContextRow);

qs("#text-input").value =
  "You are an example. They quoted 'example' in class. We reclaimed example in our own group.";
qs("#text-terms").value = "example, missing";

apiGet("/health")
  .then((payload) => {
    const mode = payload.labeler_mode ?? `${payload.app} ${payload.version}`;
    const llmState = payload.llm_configured ? "LLM key set" : "LLM key missing";
    qs("#ui-mode").textContent = `${mode} | ${llmState}`;
  })
  .catch(() => {
    qs("#ui-mode").textContent = "API unreachable";
  });
