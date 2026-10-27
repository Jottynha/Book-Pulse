const state = { books: [], selected: null, active: -1 };
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

function renderRatingChart(book) {
  const distribution = book.rating_distribution || {};
  const max = Math.max(...Object.values(distribution), 1);
  return [5, 4, 3, 2, 1].map((rating) => {
    const count = distribution[String(rating)] || 0;
    return `<div class="rating-row"><b>${rating}★</b><span class="rating-track"><i style="width:${count / max * 100}%"></i></span><small>${count}</small></div>`;
  }).join("");
}

function renderReviews(book) {
  if (!book.reviews?.length) {
    return `<p class="empty-reviews">Não há texto de resenha disponível para este livro.</p>`;
  }
  return book.reviews.map((review) => `<article class="review-card">
    <div class="review-meta"><span class="review-stars">${review.rating ? `${"★".repeat(Math.round(review.rating))}${"☆".repeat(5 - Math.round(review.rating))}` : "Sem nota"}</span><span>${review.words.toLocaleString("pt-BR")} palavras</span></div>
    <p>“${escapeHtml(review.text)}”</p>
    ${review.label ? `<small class="review-label ${escapeHtml(review.label)}">${review.label === "positive" ? "classificada como positiva" : "classificada como negativa"}</small>` : ""}
  </article>`).join("");
}

function recommendations(book) {
  const genreSet = new Set(book.genres);
  return state.books.filter((candidate) => candidate.id !== book.id).map((candidate) => {
    const shared = candidate.genres.filter((genre) => genreSet.has(genre)).length;
    // sentiment_score é a probabilidade positiva média (0 a 1); sem resenhas em
    // inglês, a distância fica neutra.
    const sentimentDistance = book.sentiment_score == null || candidate.sentiment_score == null
      ? 0.5
      : Math.abs(candidate.sentiment_score - book.sentiment_score);
    return { candidate, score: shared * 4 - sentimentDistance * 3 };
  }).sort((a, b) => b.score - a.score || b.candidate.review_count - a.candidate.review_count)
    .slice(0, 4).map((item) => item.candidate);
}

function notice(book) {
  if (book.review_count < 3) {
    return `Perfil baseado em ${book.review_count === 1 ? "uma única resenha" : "duas resenhas"} da amostra: sentimento e palavras refletem poucas pessoas.`;
  }
  if (!book.sentiment_reviews) return "Nenhuma resenha em inglês: o perfil de sentimento não foi calculado.";
  return "";
}

function updateHistory(book, replace = false) {
  const url = `?livro=${encodeURIComponent(book.id)}`;
  if (replace) history.replaceState({ bookId: book.id }, "", url);
  else history.pushState({ bookId: book.id }, "", url);
}

function showHome(replace = false) {
  state.selected = null;
  document.body.classList.add("home-mode");
  document.body.classList.remove("detail-mode");
  if (replace) history.replaceState({}, "", location.pathname);
  else history.pushState({}, "", location.pathname);
  $("#empty-state").hidden = false;
  $("#book-view").hidden = true;
  $("#search").value = "";
  $("#suggestions").hidden = true;
  $("#back-button").disabled = true;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showBook(book, { replace = false, scroll = true } = {}) {
  if (!book) return;
  state.selected = book;
  document.body.classList.add("detail-mode");
  document.body.classList.remove("home-mode");
  updateHistory(book, replace);
  $("#empty-state").hidden = true;
  $("#book-view").hidden = false;
  $("#back-button").disabled = history.length <= 1;
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
  const warning = notice(book);
  $("#book-notice").textContent = warning;
  $("#book-notice").hidden = !warning;
  const positive = book.sentiment.positive * 100;
  const negative = book.sentiment.negative * 100;
  $("#bar-positive").style.width = `${positive}%`;
  $("#bar-negative").style.width = `${negative}%`;
  $("#positive-pct").textContent = book.sentiment_reviews ? `${positive.toFixed(0)}%` : "—";
  $("#negative-pct").textContent = book.sentiment_reviews ? `${negative.toFixed(0)}%` : "—";
  $("#score").textContent = book.sentiment_score == null ? "—" : `${Math.round(book.sentiment_score * 100)}%`;
  $("#common-words").innerHTML = words(book.common_words, true);
  $("#positive-words").innerHTML = words(book.positive_words);
  $("#negative-words").innerHTML = words(book.negative_words);
  $("#rating-chart").innerHTML = renderRatingChart(book);
  $("#book-data-summary").innerHTML = `<div class="summary-number">${book.text_review_count.toLocaleString("pt-BR")} <small>com texto</small></div><p>${book.sentiment_reviews ? `${book.sentiment_reviews.toLocaleString("pt-BR")} resenhas em inglês alimentaram o perfil de sentimento.` : "Não houve resenhas em inglês suficientes para o perfil de sentimento."}</p><p>Média de ${Math.round(book.average_words).toLocaleString("pt-BR")} palavras por resenha.</p>`;
  $("#reviews-list").innerHTML = renderReviews(book);
  renderGraph(book);
  $("#recommendation-grid").innerHTML = recommendations(book).map((item) => renderCard(item)).join("");
  document.querySelectorAll("[data-id]").forEach((card) => card.addEventListener("click", () => showBook(state.books.find((item) => item.id === card.dataset.id))));
  if (scroll) window.scrollTo({ top: document.querySelector(".main").offsetTop - 20, behavior: "smooth" });
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
      return `<g class="graph-node ${node.type}" transform="translate(${node.x} ${node.y})" data-node-id="${escapeHtml(node.id)}" data-label="${escapeHtml(node.label)}" data-count="${node.count || ""}" data-related="${node.type === "related" ? "true" : "false"}" tabindex="0">
        ${clip}<circle class="node-background" r="${radius}" filter="url(#graph-shadow)"></circle>${image}
        <circle class="node-outline" r="${radius}"></circle>
        ${label ? `<text class="node-label">${escapeHtml(label)}</text>` : ""}
        ${node.type === "word" ? `<text class="node-count">${node.count}x</text>` : ""}
        ${node.type === "related" ? `<title>${escapeHtml(node.label)}</title>` : ""}
      </g>`;
    }).join("")}</g>
  </svg>`;
  $("#book-graph").innerHTML = svg;
  const graphNodes = $("#book-graph").querySelectorAll(".graph-node");
  graphNodes.forEach((node) => {
    node.addEventListener("mouseenter", (event) => showGraphTooltip(event, node, book));
    node.addEventListener("mousemove", (event) => moveGraphTooltip(event));
    node.addEventListener("mouseleave", hideGraphTooltip);
    node.addEventListener("focus", (event) => showGraphTooltip(event, node, book));
    node.addEventListener("blur", hideGraphTooltip);
  });
  $("#book-graph").querySelectorAll(".graph-node.related").forEach((node) => {
    const related = state.books.find((item) => item.id === node.dataset.nodeId);
    const open = () => showBook(related);
    node.addEventListener("click", open);
    node.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); } });
  });
  $("#book-graph").addEventListener("mouseleave", hideGraphTooltip);
  $("#book-graph").addEventListener("pointerleave", hideGraphTooltip);
}

function graphTooltipContent(node, selected) {
  if (node.classList.contains("word")) {
    return `<strong>${escapeHtml(node.dataset.label)}</strong><span>${node.dataset.count} ocorrências nas resenhas de ${escapeHtml(selected.title)}</span>`;
  }
  const book = node.classList.contains("book")
    ? selected
    : state.books.find((item) => item.id === node.dataset.nodeId);
  if (!book) return "";
  const genres = book.genres.slice(0, 2).map((genre) => genre.replaceAll("_", " ")).join(" · ");
  return `<img src="${escapeHtml(cover(book.cover_url, book.title))}" alt=""><div><strong>${escapeHtml(book.title)}</strong><span>${escapeHtml(book.author)}</span><small>${book.review_count} resenhas${genres ? ` · ${escapeHtml(genres)}` : ""}</small></div>`;
}

function showGraphTooltip(event, node, selected) {
  const tooltip = $("#graph-tooltip");
  tooltip.innerHTML = graphTooltipContent(node, selected);
  tooltip.hidden = false;
  moveGraphTooltip(event);
}

function moveGraphTooltip(event) {
  const tooltip = $("#graph-tooltip");
  if (tooltip.hidden) return;
  const gap = 12;
  const margin = 8;
  const width = tooltip.offsetWidth;
  const height = tooltip.offsetHeight;
  const candidates = [
    { left: event.clientX + gap, top: event.clientY + gap },
    { left: event.clientX - width - gap, top: event.clientY + gap },
    { left: event.clientX + gap, top: event.clientY - height - gap },
    { left: event.clientX - width - gap, top: event.clientY - height - gap },
  ].map((position) => ({
    left: Math.max(margin, Math.min(position.left, window.innerWidth - width - margin)),
    top: Math.max(margin, Math.min(position.top, window.innerHeight - height - margin)),
  }));
  const nodeRects = [...$("#book-graph").querySelectorAll(".graph-node")].map((node) => node.getBoundingClientRect());
  const overlaps = (position, rect) => nodeRects.some((node) =>
    position.left < node.right && position.left + width > node.left &&
    position.top < node.bottom && position.top + height > node.top
  );
  const best = candidates.find((position) => !overlaps(position));
  const chosen = best || candidates[0];
  tooltip.style.left = `${chosen.left}px`;
  tooltip.style.top = `${chosen.top}px`;
}

function hideGraphTooltip() {
  $("#graph-tooltip").hidden = true;
}

window.addEventListener("blur", hideGraphTooltip);
window.addEventListener("scroll", hideGraphTooltip, { passive: true });

function setCredits(open) {
  const modal = $("#credits-modal");
  modal.hidden = !open;
  document.body.classList.toggle("modal-open", open);
  if (open) $("#credits-close").focus();
}

function searchBooks(query) {
  const normalized = query.trim().toLocaleLowerCase();
  const results = normalized ? state.books.filter((book) => `${book.title} ${book.author}`.toLocaleLowerCase().includes(normalized)).slice(0, 8) : [];
  state.active = -1;
  $("#result-count").textContent = normalized ? `${results.length} resultado${results.length === 1 ? "" : "s"}` : "";
  $("#suggestions").innerHTML = results.map((book) => `<button class="suggestion" data-id="${escapeHtml(book.id)}"><strong>${escapeHtml(book.title)}</strong><small>${escapeHtml(book.author)} · ${book.review_count} resenha${book.review_count === 1 ? "" : "s"}</small></button>`).join("");
  $("#suggestions").hidden = results.length === 0;
  document.querySelectorAll(".suggestion").forEach((item) => item.addEventListener("click", () => showBook(state.books.find((book) => book.id === item.dataset.id))));
}

function navigateSuggestions(event) {
  const items = [...document.querySelectorAll(".suggestion")];
  if (!items.length || $("#suggestions").hidden) return;
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const step = event.key === "ArrowDown" ? 1 : -1;
    state.active = (state.active + step + items.length) % items.length;
    items.forEach((item, index) => item.classList.toggle("active", index === state.active));
    items[state.active].scrollIntoView({ block: "nearest" });
  } else if (event.key === "Enter") {
    event.preventDefault();
    const target = items[Math.max(0, state.active)];
    showBook(state.books.find((book) => book.id === target.dataset.id));
  } else if (event.key === "Escape") {
    $("#suggestions").hidden = true;
  }
}

async function init() {
  const response = await fetch("data/books.json");
  const payload = await response.json();
  document.body.classList.add("home-mode");
  state.books = payload.books;
  $("#footer-count").textContent = `Dados agregados de ${payload.review_count.toLocaleString("pt-BR")} resenhas confiáveis · ${payload.book_count.toLocaleString("pt-BR")} livros`;
  $("#featured").innerHTML = state.books.slice(0, 4).map((book) => renderCard(book, "mini-card")).join("");
  document.querySelectorAll("[data-id]").forEach((card) => card.addEventListener("click", () => showBook(state.books.find((item) => item.id === card.dataset.id))));
  $("#search").addEventListener("input", (event) => searchBooks(event.target.value));
  $("#search").addEventListener("keydown", navigateSuggestions);
  $("#home-button").addEventListener("click", () => showHome());
  $("#back-button").addEventListener("click", () => history.back());
  $("#credits-button").addEventListener("click", () => setCredits(true));
  $("#credits-close").addEventListener("click", () => setCredits(false));
  $("#credits-modal").addEventListener("click", (event) => {
    if (event.target.matches("[data-close-credits]")) setCredits(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("#credits-modal").hidden) setCredits(false);
  });
  window.addEventListener("popstate", () => {
    const linked = new URLSearchParams(location.search).get("livro");
    const book = linked ? state.books.find((item) => item.id === linked) : null;
    if (book) showBook(book, { replace: true, scroll: false });
    else showHome(true);
  });
  const linked = new URLSearchParams(location.search).get("livro");
  if (linked) showBook(state.books.find((book) => book.id === linked), { replace: true });
}
init().catch(() => { $("#empty-state h2").textContent = "Não foi possível carregar os dados"; });
