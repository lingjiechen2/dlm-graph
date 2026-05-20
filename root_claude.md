# Writing Rules (Local)

These rules apply to all academic writing tasks (papers, drafts, rebuttals) in this directory.

## Font / Emphasis Conventions

Use the following LaTeX commands strictly. Do not mix or substitute.

### `\texttt{}` — monospace
Use for things that are literally code-like or system-level:
- Code identifiers: function names, variable names, class names (e.g., `\texttt{forward()}`, `\texttt{batch\_size}`)
- File paths, command names, flags (e.g., `\texttt{--lr}`, `\texttt{train.py}`)
- Model checkpoint / API names when referring to the artifact, not the method (e.g., `\texttt{gpt-4o-mini}`, `\texttt{Llama-3-8B}`)
- Dataset / benchmark identifiers when treated as a tag (e.g., `\texttt{MMLU}` only if you want the tag form; otherwise plain text is fine — be consistent within a paper)
- Token / string literals (e.g., `\texttt{<eos>}`)

Do **not** use `\texttt{}` for: method names, general English emphasis, or headings.

### `\textit{}` — italic
Use for:
- **First introduction of a new term** that the paper defines: "we call this the \textit{reflection step}". After the first use, switch to plain text.
- Foreign / Latin phrases: \textit{i.e.}, \textit{e.g.}, \textit{et al.}, \textit{a priori}, \textit{vs.}
- Paper / book titles when cited inline (rare; usually handled by bibtex)
- Light emphasis on a single word where bold would be too loud

Do **not** use `\textit{}` for: math variables in text (use `$x$` instead), method names (plain text or small caps), or whole sentences.

### `\textbf{}` — bold
Use sparingly, only for:
- Paragraph-leading labels in lists / inline headings: "\textbf{Setup.} We train ..."
- Best results in tables (numbers only)
- Key takeaway sentence in an abstract / takeaways box (at most one per section)

Do **not** use `\textbf{}` for: in-text emphasis, term introduction, or method names.

### Method names
Use plain text or `\textsc{}` (small caps) consistently within a paper. Pick one and stick to it. Do not switch between `\textit{}`, `\texttt{}`, and `\textbf{}` for the same method.

## Abbreviations
- **First mention**: full form followed by abbreviation in parentheses, e.g., "Large Language Model (LLM)".
- **All subsequent mentions**: abbreviation only. Never re-spell.
- Apply per-document, not per-section. Once defined in the introduction, do not redefine later.
- Common abbreviations that need no expansion: GPU, CPU, API, JSON, NLP. When in doubt, expand.

## Formulas
- **Max length: half a page.** ACL papers use a two-column layout, so a formula must never exceed half of one column-page in vertical extent. If it does, refactor: introduce named intermediate symbols, split across multiple numbered equations, or move detail to an appendix.
- Keep single-line equations within (single) column width. If an equation overflows the column, break it across lines using `align` / `split`, aligned at `=` or a binary operator.
- Inline math should be short (a few symbols). Anything with fractions, sums, or multi-term products goes into display math.
- Define every symbol the first time it appears. Do not reuse symbols for different quantities.

## Citations
**Always verify on Google Scholar before inserting a citation.** Process:
1. Search Google Scholar for the paper by title or authors.
2. Confirm: authors, year, venue, exact title.
3. Prefer the published venue version over arXiv when both exist.
4. Only after verification, insert the `\cite{}` entry and update the `.bib` file with the correct fields.

Never insert a `\cite{}` from memory or training data without checking — citation hallucinations are unacceptable.

## Tables
- **Best result per column**: bold (`\textbf{}`).
- **Second-best result per column**: underline (`\underline{}`).
- Apply per column unless the user specifies otherwise (e.g., per row, per group of columns).
- Ties: bold all tied best values; underline all tied second-best.
- Add a short note in the caption: "Best in \textbf{bold}, second-best \underline{underlined}."
- Keep numerical precision consistent within a column (same number of decimal places).

## Consistency
- Pick one spelling per term (e.g., "fine-tuning" vs "finetuning") and use it everywhere.
- Pick one capitalization per method/dataset name and use it everywhere.
- Pick one notation per symbol (e.g., $\theta$ for parameters) and do not reuse the symbol for another quantity.

## Build / Compile
**After making any change to LaTeX source files, recompile the PDF before reporting the task as complete.** Do not assume the source compiles — run the build, check for errors, and fix them. If the project uses `pdflatex` + `bibtex`, run `pdflatex → bibtex → pdflatex → pdflatex`. If it uses `latexmk`, run `latexmk -pdf`. Report the compile result (success / errors / warnings) along with the diff.

## Prose Style

**Prefer flowing prose over bullet points.** Bullet lists are appropriate only for genuinely enumerable items (final contributions list, hyperparameter values, dataset specs). Do not use bullets to dodge the work of writing connective sentences. If three points belong together, write a paragraph that links them with "first… second… finally…" or with logical connectives ("because", "therefore", "in contrast"). A page full of bullets reads as notes, not as a paper.

**Assume nothing about the reader.** Do not write "as is well known", "obviously", "it is clear that", or "the standard X". If a term, symbol, dataset, or technique is needed to follow the sentence, define or cite it the first time it appears. When in doubt, give one extra sentence of context rather than leaving a gap. The reader is a competent researcher in an adjacent field, not a specialist who has read the same papers you have.

**Be concise, but complete.** Concise means no filler ("In order to" → "To"; "due to the fact that" → "because"; "it should be noted that" → delete). Complete means every claim has the information needed to evaluate it: what was measured, on what data, against what baseline, with what variance. Cutting words is good; cutting load-bearing information is not. If a sentence can be shortened without losing a fact, shorten it; if shortening drops a fact, keep the fact and rewrite the sentence.

**Self-contained sentences.** A reader who jumps to the middle of a section should be able to parse the sentence in front of them. Avoid pronouns ("this", "it", "they") whose antecedent sits two paragraphs back — repeat the noun.
