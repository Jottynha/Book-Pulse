/**
 * BookPulse — recebe as anotações de anotar.html numa planilha Google.
 *
 * Instalação (uma vez):
 * 1. Crie uma planilha no Google Sheets.
 * 2. Menu Extensões > Apps Script; apague o conteúdo e cole este arquivo.
 * 3. Implantar > Nova implantação > tipo "App da Web";
 *    Executar como: "Eu"; Quem pode acessar: "Qualquer pessoa".
 * 4. Autorize e copie a URL terminada em /exec para SHEETS_URL em anotar.js.
 *
 * A aba "anotacoes" é um log só de inclusão: reclassificar uma resenha cria
 * outra linha, e vale a mais recente por anotador e resenha.
 */

const SHEET_NAME = "anotacoes";
const HEADER = ["review_id", "label", "uncertain", "annotator", "annotated_at", "received_at"];
const LABELS = ["positive", "negative", "neutral", "mixed", "unclear"];

function sheet_() {
  const book = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = book.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = book.insertSheet(SHEET_NAME);
    sheet.appendRow(HEADER);
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function json_(value) {
  return ContentService.createTextOutput(JSON.stringify(value)).setMimeType(ContentService.MimeType.JSON);
}

function clean_(value, maxLength) {
  return String(value == null ? "" : value).replace(/[\r\n\t]/g, " ").slice(0, maxLength);
}

function doPost(e) {
  let items;
  try {
    const body = JSON.parse(e.postData.contents);
    items = Array.isArray(body.items) ? body.items : [body];
  } catch (error) {
    return json_({ ok: false, error: "JSON inválido" });
  }
  const now = new Date().toISOString();
  const rows = items
    .filter((item) => item && LABELS.indexOf(item.label) >= 0 && /^[0-9a-f]{8,64}$/i.test(item.review_id || "") && item.annotator)
    .slice(0, 200)
    .map((item) => [
      // Prefixo "'" impede o Sheets de converter ids e datas.
      "'" + clean_(item.review_id, 64),
      item.label,
      item.uncertain ? 1 : 0,
      "'" + clean_(item.annotator, 30).toLowerCase(),
      "'" + clean_(item.annotated_at, 40),
      now,
    ]);
  if (rows.length) {
    const lock = LockService.getScriptLock();
    lock.waitLock(20000);
    try {
      const sheet = sheet_();
      sheet.getRange(sheet.getLastRow() + 1, 1, rows.length, HEADER.length).setValues(rows);
    } finally {
      lock.releaseLock();
    }
  }
  return json_({ ok: true, saved: rows.length, rejected: items.length - rows.length });
}

function doGet(e) {
  const params = (e && e.parameter) || {};
  const values = sheet_().getDataRange().getDisplayValues();
  const header = values.shift() || HEADER;
  let rows = values.map((row) => Object.fromEntries(header.map((name, index) => [name, row[index]])));
  if (params.annotator) rows = rows.filter((row) => row.annotator === String(params.annotator).toLowerCase());
  if (params.format === "csv") {
    const escape = (value) => (/[",\n]/.test(value) ? '"' + value.replace(/"/g, '""') + '"' : value);
    const csv = [header.join(",")].concat(rows.map((row) => header.map((name) => escape(String(row[name] || ""))).join(","))).join("\n");
    return ContentService.createTextOutput(csv + "\n").setMimeType(ContentService.MimeType.CSV);
  }
  return json_({ ok: true, rows });
}
