const state = { books: [], selected: null };
const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
}[char]));

function cover(url, title) {
  return url || `https://placehold.co/330x470/e3e7e1/0e7770?text=${encodeURIComponent(title.slice(0, 18))}`;
}

function coverMarkup(book, className) {
  const primary = cover(book.cover_url, book.title);
  const fallback = book.cover_fallback_url || `https://placehold.co/330x470/e3e7e1/0e7770?text=${encodeURIComponent(book.title.slice(0, 18))}`;
  return `<div class="${className}"><img src="${primary}" data-fallback="${fallback}" alt="Capa de ${escapeHtml(book.title)}" loading="lazy" onerror="this.onerror=null;this.src=this.dataset.fallback"></div>`;
}

function renderCard(book, className = "recommend-card") {
  return `<article class="${className}" data-id="${escapeHtml(book.id)}">
    ${coverMarkup(book, className === "mini-card" ? "mini-cover" : "recommend-cover")}
    <div class="${className === "mini-card" ? "mini-info" : "recommend-info"}"><strong>${escapeHtml(book.title)}</strong><small>${escapeHtml(book.author)}</small></div>
  </article>`;
}

function words(items, cloud = false) {
  return (items || []).map((item, index) =>
    `<span style="${cloud ? `font-size:${12 + Math.max(0, 10 - index)}px` : ""}">${escapeHtml(item.word)} <small>${item.count}</small></span>`
  ).join("");
}

function recommendations(book) {
  const genreSet = new Set(book.genres);
  return state.books.filter((candidate) => candidate.id !== book.id).map((candidate) => {
    const shared = candidate.genres.filter((genre) => genreSet.has(genre)).length;
    const sentimentDistance = Math.abs(candidate.sentiment_score - book.sentiment_score);
    return { candidate, score: shared * 4 - sentimentDistance };
  }).sort((a, b) => b.score - a.score || b.candidate.review_count - a.candidate.review_count)
    .slice(0, 4).map((item) => item.candidate);
}

function showBook(book) {
  state.selected = book;
  $("#empty-state").hidden = true;
  $("#book-view").hidden = false;
  $("#search").value = book.title;
  $("#suggestions").hidden = true;
  $("#cover").src = cover(book.cover_url, book.title);
  $("#cover").dataset.fallback = book.cover_fallback_url || `https://placehold.co/330x470/e3e7e1/0e7770?text=${encodeURIComponent(book.title.slice(0, 18))}`;
  $("#cover").onerror = function () { this.onerror = null; this.src = this.dataset.fallback; };
  $("#cover").alt = `Capa de ${book.title}`;
  $("#book-title").textContent = book.title;
  $("#book-author").textContent = book.author;
  $("#book-genres").innerHTML = book.genres.map((genre) => `<span>${escapeHtml(genre.replaceAll("_", " "))}</span>`).join("");
  $("#book-reviews").textContent = book.review_count.toLocaleString("pt-BR");
  $("#book-rating").textContent = book.average_rating ? book.average_rating.toFixed(2) : "—";
  $("#book-words").textContent = Math.round(book.average_words).toLocaleString("pt-BR");
  const positive = book.sentiment.positive * 100;
  const neutral = book.sentiment.neutral * 100;
  const negative = book.sentiment.negative * 100;
  $("#bar-positive").style.width = `${positive}%`;
  $("#bar-neutral").style.width = `${neutral}%`;
  $("#bar-negative").style.width = `${negative}%`;
  $("#positive-pct").textContent = `${positive.toFixed(0)}%`;
  $("#neutral-pct").textContent = `${neutral.toFixed(0)}%`;
  $("#negative-pct").textContent = `${negative.toFixed(0)}%`;
  $("#score").textContent = `${book.sentiment_score > 0 ? "+" : ""}${book.sentiment_score.toFixed(2)}`;
  $("#common-words").innerHTML = words(book.common_words, true);
  $("#positive-words").innerHTML = words(book.positive_words);
  $("#negative-words").innerHTML = words(book.negative_words);
  renderGraph(book);
  $("#recommendation-grid").innerHTML = recommendations(book).map((item) => renderCard(item)).join("");
  document.querySelectorAll("[data-id]").forEach((card) => card.addEventListener("click", () => showBook(state.books.find((item) => item.id === card.dataset.id))));
  window.scrollTo({ top: document.querySelector(".main").offsetTop - 20, behavior: "smooth" });
}

function relatedBooks(book, word) {
  return state.books.filter((candidate) => candidate.id !== book.id)
    .map((candidate) => {
      const match = candidate.common_words.find((item) => item.word === word.word);
      return match ? { book: candidate, count: match.count } : null;
    })
    .filter(Boolean)
    .sort((a, b) => b.count - a.count || b.book.review_count - a.book.review_count)
    .slice(0, 3);
}

function renderGraph(book) {
  const width = 900;
  const height = 560;
  const center = { x: width / 2, y: height / 2 };
  const wordsToShow = (book.common_words || []).slice(0, 6);
  const nodes = [{ type: "book", id: book.id, label: book.title, x: center.x, y: center.y }];
  const edges = [];
  const usedBooks = new Set([book.id]);
  wordsToShow.forEach((word, wordIndex) => {
    const angle = -Math.PI / 2 + (Math.PI * 2 * wordIndex) / Math.max(wordsToShow.length, 1);
    const wordNode = {
      type: "word", id: `word-${word.word}`, label: word.word, count: word.count,
      x: center.x + Math.cos(angle) * 145, y: center.y + Math.sin(angle) * 145
    };
    nodes.push(wordNode);
    edges.push({ source: nodes[0], target: wordNode, weight: word.count });
    relatedBooks(book, word).forEach(({ book: related, count }, relatedIndex) => {
      if (usedBooks.has(related.id)) return;
      usedBooks.add(related.id);
      const spread = (relatedIndex - 1) * 0.26;
      const relatedNode = {
        type: "related", id: related.id, label: related.title, book: related,
        x: center.x + Math.cos(angle + spread) * 255,
        y: center.y + Math.sin(angle + spread) * 255
      };
      nodes.push(relatedNode);
      edges.push({ source: wordNode, target: relatedNode, weight: count });
    });
  });
  const svg = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
    <defs><filter id="graph-shadow" x="-30%" y="-30%" width="160%" height="160%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-opacity=".15"/></filter></defs>
    <g class="graph-edges">${edges.map((edge) => `<line x1="${edge.source.x}" y1="${edge.source.y}" x2="${edge.target.x}" y2="${edge.target.y}" style="stroke-width:${Math.min(5, 1 + edge.weight / 6)}"></line>`).join("")}</g>
    <g class="graph-nodes">${nodes.map((node) => {
      const radius = node.type === "book" ? 43 : node.type === "word" ? 26 + Math.min(13, node.count / 3) : 25;
      const label = node.type === "word" ? node.label : "";
      const image = node.type !== "word"
        ? `<image class="node-cover" href="${escapeHtml(cover(node.book?.cover_url || book.cover_url, node.label))}" data-fallback="${escapeHtml(node.book?.cover_fallback_url || "")}" x="${-radius}" y="${-radius}" width="${radius * 2}" height="${radius * 2}" preserveAspectRatio="xMidYMid slice" clip-path="url(#clip-${node.type}-${node.id})" onerror="this.onerror=null;this.setAttribute('href',this.dataset.fallback || '${escapeHtml(cover("", node.label))}')"></image>`
        : "";
      const clip = node.type !== "word"
        ? `<clipPath id="clip-${node.type}-${node.id}"><circle r="${radius - 1}"></circle></clipPath>`
        : "";
      return `<g class="graph-node ${node.type}" transform="translate(${node.x} ${node.y})" data-node-id="${escapeHtml(node.id)}" tabindex="0">
        ${clip}<circle class="node-background" r="${radius}" filter="url(#graph-shadow)"></circle>${image}
        <circle class="node-outline" r="${radius}"></circle>
        ${label ? `<text class="node-label">${escapeHtml(label)}</text>` : ""}
        ${node.type === "word" ? `<text class="node-count">${node.count}x</text>` : ""}
        ${node.type === "related" ? `<title>${escapeHtml(node.label)}</title>` : ""}
      </g>`;
    }).join("")}</g>
  </svg>`;
  $("#book-graph").innerHTML = svg;
  $("#book-graph").querySelectorAll(".graph-node.related").forEach((node) => {
    const related = state.books.find((item) => item.id === node.dataset.nodeId);
    node.addEventListener("click", () => showBook(related));
    node.addEventListener("keydown", (event) => { if (event.key === "Enter") showBook(related); });
  });
}

function searchBooks(query) {
  const normalized = query.trim().toLocaleLowerCase();
  const results = normalized ? state.books.filter((book) => `${book.title} ${book.author}`.toLocaleLowerCase().includes(normalized)).slice(0, 8) : [];
  $("#result-count").textContent = normalized ? `${results.length} resultado${results.length === 1 ? "" : "s"}` : "";
  $("#suggestions").innerHTML = results.map((book) => `<button class="suggestion" data-id="${escapeHtml(book.id)}"><strong>${escapeHtml(book.title)}</strong><small>${escapeHtml(book.author)} · ${book.review_count} resenhas</small></button>`).join("");
  $("#suggestions").hidden = results.length === 0;
  document.querySelectorAll(".suggestion").forEach((item) => item.addEventListener("click", () => showBook(state.books.find((book) => book.id === item.dataset.id))));
}

async function init() {
  const response = await fetch("data/books.json");
  const payload = await response.json();
  state.books = payload.books;
  $("#featured").innerHTML = state.books.slice(0, 4).map((book) => renderCard(book, "mini-card")).join("");
  document.querySelectorAll("[data-id]").forEach((card) => card.addEventListener("click", () => showBook(state.books.find((item) => item.id === card.dataset.id))));
  $("#search").addEventListener("input", (event) => searchBooks(event.target.value));
}
init().catch(() => { $("#empty-state h2").textContent = "Não foi possível carregar os dados"; });
