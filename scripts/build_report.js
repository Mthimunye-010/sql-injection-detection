// Builds the NCOS734 mini-project report as a .docx matching the module template
// (A4, Arial 10pt, template margins, departmental footer with page numbers).
//
//   node scripts/build_report.js
//
// Every figure referenced here is read from results/figures/ and every number in
// the prose comes from results/tables/.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, Footer, PageNumber, PageBreak, TabStopType, convertInchesToTwip,
  LevelFormat, Numbering,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "results", "figures");

// ---------------------------------------------------------------- style atoms
const FONT = "Arial";
const SZ = 20;        // 10pt body, matching the template
const SZ_SMALL = 18;  // 9pt for tables and captions
const CONTENT_W = 9602;   // 11906 - 1152 - 1152

const P = (text, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after === undefined ? 120 : opts.after, line: 252 },
  indent: opts.indent,
  children: [new TextRun({ text, font: FONT, size: opts.size || SZ,
                           bold: opts.bold, italics: opts.italics })],
});

// Rich paragraph: array of [text, {bold, italics}] fragments
const RP = (frags, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after === undefined ? 120 : opts.after, line: 252 },
  children: frags.map(([t, o = {}]) => new TextRun({
    text: t, font: FONT, size: opts.size || SZ,
    bold: o.bold, italics: o.italics, subScript: o.sub, superScript: o.sup,
  })),
});

const H1 = (text, marks) => new Paragraph({
  spacing: { before: 280, after: 160 },
  children: [
    new TextRun({ text, font: FONT, size: 24, bold: true }),
    ...(marks ? [new TextRun({ text: `\t[${marks}]`, font: FONT, size: 24, bold: true })] : []),
  ],
  tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
});

const H2 = (text) => new Paragraph({
  spacing: { before: 200, after: 100 },
  children: [new TextRun({ text, font: FONT, size: SZ, bold: true })],
});

const BULLET = (text, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  spacing: { after: 80, line: 252 },
  children: [new TextRun({ text, font: FONT, size: SZ })],
});

const NUM = (text, level = 0) => new Paragraph({
  numbering: { reference: "numbers", level },
  spacing: { after: 80, line: 252 },
  children: [new TextRun({ text, font: FONT, size: SZ })],
});

// Monospace block for pseudocode / code
const CODE = (text) => new Paragraph({
  spacing: { after: 0, line: 216 },
  children: [new TextRun({ text, font: "Consolas", size: 16 })],
});

// Centred italic equation line
const EQ = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 100, after: 100 },
  children: [new TextRun({ text, font: "Cambria Math", size: SZ, italics: true })],
});

const CAPTION = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 200 },
  children: [new TextRun({ text, font: FONT, size: SZ_SMALL, italics: true })],
});

// ------------------------------------------------------------------- tables
function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const w = widths.map((x) => Math.round(x * scale));

  const cell = (text, i, { bold = false, shade = null, align = null } = {}) =>
    new TableCell({
      width: { size: w[i], type: WidthType.DXA },
      shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: "auto" } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({
        alignment: align || (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER),
        spacing: { after: 0, line: 240 },
        children: [new TextRun({ text: String(text), font: FONT,
                                 size: opts.size || SZ_SMALL, bold })],
      })],
    });

  return new Table({
    columnWidths: w,
    width: { size: CONTENT_W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "808080" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "808080" },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "C0C0C0" },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => cell(h, i, { bold: true, shade: "EDEDED" })),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((c, i) => cell(c, i, { bold: opts.boldRows && r[0] === opts.boldRows })),
      })),
    ],
  });
}

// -------------------------------------------------------------------- images
function figure(file, widthIn, caption) {
  const p = path.join(FIG, file);
  if (!fs.existsSync(p)) {
    console.warn("  missing figure: " + file);
    return [P(`[figure ${file} not found]`, { italics: true })];
  }
  // Preserve aspect ratio by reading the PNG header.
  const buf = fs.readFileSync(p);
  const pxW = buf.readUInt32BE(16), pxH = buf.readUInt32BE(20);
  const wPt = widthIn * 72, hPt = Math.round(wPt * pxH / pxW);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 0 },
      children: [new ImageRun({ type: "png", data: buf,
                                transformation: { width: Math.round(wPt), height: hPt } })],
    }),
    CAPTION(caption),
  ];
}

module.exports = { P, RP, H1, H2, BULLET, NUM, CODE, EQ, CAPTION, table, figure,
                   FONT, SZ, SZ_SMALL, CONTENT_W, Document, Packer, Paragraph,
                   TextRun, AlignmentType, Footer, PageNumber, PageBreak, ROOT };
