// In-browser port of the trained pipeline: normalize -> word 1-2grams ->
// sublinear TF-IDF -> L2 norm -> logistic regression.
// Mirrors baitnswitch/text.py and sklearn's TfidfVectorizer; tests/test_parity.py
// checks this file against the Python model.

const URL_RE = /https?:\/\/\S+|www\.\S+/g;
const RETWEET_RE = /^rt\s+@[\p{L}\p{N}_]+:?/u;
const MENTION_RE = /@[\p{L}\p{N}_]+/gu;
const TOKEN_RE = /[a-z0-9]+|[!?]/g;
const QUOTES = { "‘": "'", "’": "'", "“": '"', "”": '"' };

export function normalize(text) {
  return String(text)
    .normalize("NFKD")
    .replace(/\p{Mn}/gu, "")
    .replace(/[‘’“”]/g, (c) => QUOTES[c])
    .toLowerCase()
    .replace(URL_RE, " ")
    .replace(RETWEET_RE, " ")
    .replace(MENTION_RE, " ")
    .replaceAll("#", " ")
    .split(/\s+/)
    .filter(Boolean)
    .join(" ");
}

export function terms(text) {
  const tokens = normalize(text).match(TOKEN_RE) || [];
  const out = [...tokens];
  for (let i = 0; i + 1 < tokens.length; i++) out.push(`${tokens[i]} ${tokens[i + 1]}`);
  return out;
}

export function createScorer(weights) {
  const { features, intercept, threshold } = weights;
  function probability(text) {
    const counts = new Map();
    for (const t of terms(text)) {
      if (t in features) counts.set(t, (counts.get(t) || 0) + 1);
    }
    let norm = 0;
    const values = [];
    for (const [t, tf] of counts) {
      const [idf, coef] = features[t];
      const v = (1 + Math.log(tf)) * idf;
      values.push([v, coef]);
      norm += v * v;
    }
    norm = Math.sqrt(norm) || 1;
    let z = intercept;
    for (const [v, coef] of values) z += (v / norm) * coef;
    return 1 / (1 + Math.exp(-z));
  }
  return { probability, threshold };
}
