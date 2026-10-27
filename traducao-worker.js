// Tradução EN→PT no navegador com transformers.js (OPUS-MT em ONNX).
// Procura primeiro uma cópia local em ./models/ e, se não houver, baixa do
// Hugging Face na revisão fixada. O modelo fica no cache do navegador.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";

const MODELS = {
  big: { id: "Douglasrambo/opus-mt-tc-big-en-pt-onnx", revision: "738ee4cce801ce7c519954e4b78d02d5057b6740", prefix: ">>pob<<" },
  compact: { id: "Xenova/opus-mt-en-ROMANCE", revision: "9d2ba69ac80c8e8453c3d9a1e2323a0e7b8ca3cd", prefix: ">>pt_BR<<" },
};
const MAX_TOKENS = 400;
const CHUNK_WORDS = 80;

env.allowLocalModels = true;
env.allowRemoteModels = true;
env.localModelPath = new URL("./models/", import.meta.url).href;
env.useBrowserCache = true;
env.backends.onnx.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const sentenceSplitter = new Intl.Segmenter("en", { granularity: "sentence" });
let translator = null;
let spec = null;
let loadedVariant = null;
const queue = [];
let running = false;
let currentId = null;

const status = (message) => self.postMessage({ type: "status", message });

async function load(variant) {
  if (translator && loadedVariant === variant) return;
  await translator?.dispose?.();
  translator = null;
  spec = MODELS[variant] || MODELS.big;
  let last = 0;
  translator = await pipeline("translation", spec.id, {
    device: "wasm",
    dtype: "q8",
    revision: spec.revision,
    progress_callback: (event) => {
      if (event.status === "progress" && performance.now() - last > 300) {
        status(`baixando modelo ${Math.round(event.progress || 0)}% (${event.file})`);
        last = performance.now();
      }
    },
  });
  loadedVariant = variant;
}

async function translatePiece(source) {
  const input = await translator.tokenizer(`${spec.prefix} ${source}`, { truncation: false });
  const tokenCount = input.input_ids.data.length;
  if (tokenCount > MAX_TOKENS) {
    for (const value of Object.values(input)) value.dispose?.();
    // Frases muito longas são divididas por palavras antes de traduzir.
    const words = source.split(/\s+/);
    const parts = [];
    for (let start = 0; start < words.length; start += CHUNK_WORDS) {
      parts.push(await translatePiece(words.slice(start, start + CHUNK_WORDS).join(" ")));
    }
    return parts.join(" ");
  }
  const maximum = Math.min(512, Math.max(64, tokenCount * 3));
  const output = await translator.model.generate({ ...input, max_new_tokens: maximum, num_beams: 1, do_sample: false });
  try {
    return translator.tokenizer.batch_decode(output, { skip_special_tokens: true })[0]?.trim() || "";
  } finally {
    output.dispose?.();
    for (const value of Object.values(input)) value.dispose?.();
  }
}

async function translate(job) {
  await load(job.variant);
  const paragraphs = job.text.split(/\n+/).map((item) => item.trim()).filter(Boolean);
  const done = [];
  for (const paragraph of paragraphs) {
    const sentences = [...sentenceSplitter.segment(paragraph)].map((item) => item.segment.trim()).filter(Boolean);
    const translated = [];
    for (const sentence of sentences) {
      try {
        translated.push(await translatePiece(sentence));
      } catch {
        translated.push(`[não traduzido] ${sentence}`);
      }
      self.postMessage({ type: "partial", id: job.id, text: [...done, translated.join(" ")].join("\n\n") });
    }
    done.push(translated.join(" "));
  }
  return done.join("\n\n");
}

async function drain() {
  if (running) return;
  running = true;
  while (queue.length) {
    const job = queue.shift();
    currentId = job.id;
    try {
      const text = await translate(job);
      self.postMessage({ type: "done", id: job.id, variant: job.variant, text });
    } catch (error) {
      self.postMessage({ type: "error", id: job.id, message: String(error?.message || error) });
    }
  }
  currentId = null;
  running = false;
}

self.onmessage = ({ data }) => {
  if (data.type === "translate") {
    if (data.id === currentId) return;
    // Pedido prioritário (cartão atual) entra na frente da fila.
    const index = queue.findIndex((job) => job.id === data.id);
    if (index >= 0) queue.splice(index, 1);
    if (data.priority) queue.unshift(data); else queue.push(data);
    drain();
  } else if (data.type === "clear") {
    queue.length = 0;
  }
};
