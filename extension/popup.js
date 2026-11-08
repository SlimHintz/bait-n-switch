import { createScorer } from "./scorer.js";
import { collectHeadlines, markHeadlines, clearMarks } from "./page.js";

const $ = (s) => document.querySelector(s);
const run = (tabId, func, args = []) =>
  chrome.scripting.executeScript({ target: { tabId }, func, args }).then(([r]) => r.result);

async function main() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const scorer = createScorer(await (await fetch("weights.json")).json());

  let headlines;
  try {
    headlines = await run(tab.id, collectHeadlines);
  } catch {
    $("#big").textContent = "–";
    $("#sub").textContent = "Can't read this page (browser pages and the Web Store are off limits).";
    return;
  }
  const scored = headlines.map((h) => ({ ...h, probability: scorer.probability(h.text) }))
                          .sort((a, b) => b.probability - a.probability);

  // Sensitivity is the inverse of the threshold: higher sensitivity flags more.
  const sens = $("#sens");
  sens.value = (1 - scorer.threshold).toFixed(2);

  async function render() {
    const threshold = 1 - Number(sens.value);
    $("#sensval").textContent = `(threshold ${threshold.toFixed(2)})`;
    const flagged = scored.filter((h) => h.probability >= threshold);
    $("#big").textContent = scored.length ? `${flagged.length} / ${scored.length}` : "0";
    $("#sub").textContent = scored.length ? "headlines look like clickbait" : "No headlines found on this page.";
    $("#list").replaceChildren(...flagged.slice(0, 50).map((h) => {
      const li = document.createElement("li");
      const b = document.createElement("b");
      b.textContent = `${Math.round(h.probability * 100)}%`;
      li.append(b, h.text);
      return li;
    }));
    await run(tab.id, markHeadlines, [flagged.map(({ id, probability }) => ({ id, probability }))]);
  }
  sens.addEventListener("input", render);
  $("#clear").addEventListener("click", () => run(tab.id, clearMarks).then(() => window.close()));
  await render();
}

main();
