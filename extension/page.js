// Functions injected into the page with chrome.scripting.executeScript.
// They must be self-contained: no imports, no closures over popup state.

export function collectHeadlines() {
  const MARK = "data-bns-id";
  const wordCount = (s) => (s.match(/[\p{L}\p{N}]+/gu) || []).length;
  const skip = (el) => el.closest("nav, footer, header nav, [role=navigation], [aria-hidden=true]");
  // Undo a previous scan first; a card it hid would otherwise be skipped as not rendered.
  for (const el of document.querySelectorAll(`[${MARK}], [data-bns-card]`)) {
    el.removeAttribute(MARK);
    el.removeAttribute("data-bns-card");
  }
  const seen = new Set();
  const found = [];
  let id = 0;
  for (const el of document.querySelectorAll("h1, h2, h3, h4, a")) {
    if (skip(el) || el.closest(`[${MARK}]`) || el.querySelector(`[${MARK}]`)) continue;
    const text = el.innerText.replace(/\s+/g, " ").trim();
    const n = wordCount(text);
    const isLink = el.tagName === "A";
    if (n < 4 || n > 30 || (isLink && n < 5) || seen.has(text)) continue;
    if (!el.getClientRects().length) continue; // not rendered
    seen.add(text);
    el.setAttribute(MARK, String(id));
    found.push({ id: id++, text });
  }
  return found;
}

export function markHeadlines(flagged, mode = "highlight") {
  const STYLE_ID = "bns-style";
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      [data-bns-bait] { outline: 2px solid #ea580c !important; outline-offset: 2px; border-radius: 2px;
                        background: rgba(234, 88, 12, .08) !important; }
      [data-bns-bait]::after { content: " \\1F33D " attr(data-bns-bait); font: 600 11px/1 system-ui, sans-serif;
                               color: #fff; background: #ea580c; border-radius: 999px; padding: 2px 6px;
                               margin-left: 6px; vertical-align: middle; white-space: nowrap; }
      [data-bns-card=blur] { filter: blur(6px) grayscale(.7) !important; opacity: .45 !important;
                             transition: filter .15s, opacity .15s; }
      [data-bns-card=blur]:hover { filter: none !important; opacity: 1 !important; }
      [data-bns-card=hide] { display: none !important; }`;
    document.head.append(style);
  }

  // The story card around a headline: climb while the ancestor still holds only
  // this one scanned headline, so hiding it never takes a neighbouring story along.
  const storyCard = (el) => {
    let card = el;
    for (let depth = 0; depth < 8; depth++) {
      const parent = card.parentElement;
      if (!parent || parent === document.body || parent.matches("main, [role=main]")) break;
      if (parent.querySelectorAll("[data-bns-id]").length > 1) break;
      if (parent.getBoundingClientRect().height > innerHeight * 0.9) break;
      card = parent;
      if (card.matches("article, li, [role=article]")) break;
    }
    return card;
  };

  for (const el of document.querySelectorAll("[data-bns-bait]")) el.removeAttribute("data-bns-bait");
  for (const el of document.querySelectorAll("[data-bns-card]")) el.removeAttribute("data-bns-card");
  for (const { id, probability } of flagged) {
    const el = document.querySelector(`[data-bns-id="${id}"]`);
    if (!el) continue;
    if (mode !== "hide") el.setAttribute("data-bns-bait", `${Math.round(probability * 100)}%`);
    if (mode !== "highlight") storyCard(el).setAttribute("data-bns-card", mode);
  }
}

export function clearMarks() {
  for (const el of document.querySelectorAll("[data-bns-bait]")) el.removeAttribute("data-bns-bait");
  for (const el of document.querySelectorAll("[data-bns-card]")) el.removeAttribute("data-bns-card");
  document.getElementById("bns-style")?.remove();
}
