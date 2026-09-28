---
name: diffusion-research-seminars
description: Find talks and papers in the Research Seminars on Diffusion Models archive, prepare reading guides, compare methods, and develop seminar discussion questions. Use for this research seminar series and its archived talks.
---

# Research Seminars on Diffusion Models

Help the reader find relevant seminars and understand their associated research.
Ground archive facts in the index and technical explanations in sources actually read.
Respond in the reader's language; the archive descriptions are English and recordings
are in Russian.

## Locate the materials

Find the VisualGenAI repository from the supplied path or current workspace. Read its
README and applicable project instructions. Paths below are relative to that repository:

- `research_seminars/INDEX.md`: searchable seminar descriptions, dates, speaker names,
  recording links, source links, and PDF page references.
- `research_seminars/diffusion_research_seminars_09_2026.pdf`: original English archive
  for checking entries and retrieving recording access codes.
- `research_seminars/slides/`: inspect the files currently available before promising
  a slide-based explanation. The September 2026 snapshot contains only a placeholder.
- `README.md`: course prerequisites, archive links, and the seminar Telegram group.

If no checkout is available, use the public repository at
https://github.com/dbaranchuk/VisualGenAI or ask for the relevant entry. Use the
available index before searching the wider web for talks. The archive covers
2023–September 2026; do not infer that its last entry is the latest seminar held.

## Find a talk or plan a reading sequence

Search the index by paper title, method acronym, topic, speaker name, and date.
Expand relevant synonyms when needed, such as diffusion language models / DLMs,
consistency / flow maps, or distillation / distribution matching. Read each matched
entry with its surrounding text; a seminar can cover several papers, and repeated
dates can identify different entries.

Return the exact date and title or description, the speaker when listed, recording
links when present, and the PDF page. Dates are seminar dates, not paper publication
dates. Some entries have multiple recording parts; some have no recording link.
Preserve those distinctions and the original URLs. Treat numbered source links as
entry-level references until their association with a particular paper is verified.

For a reading plan, use the reader's background and available time. Explain why each
selected talk is relevant and put prerequisites before more specialized methods.
Use the course syllabus in README when foundational diffusion material is needed.
Identify a proposed learning order as your recommendation; archive order alone does
not establish prerequisite relationships.

## Prepare a paper or seminar discussion

Open the paper or official project/implementation linked from the entry. Verify its
title, authors, and version before attributing a result. If an archive link is a private
message or is unavailable, use the exact listed title to locate a primary source and
identify the replacement source. Do not guess paper identity from a short acronym.

Tailor the output to the request:

- **Reading guide:** explain the research question, required background, main idea,
  and the equations or experiments to focus on. Cite paper sections or slide pages.
- **Method comparison:** align the problem, conditioning, model, objective, training
  resources, and evaluation setting before comparing results. Preserve distinctions
  between sampling steps, model evaluations, tokens, and measured runtime.
- **Discussion preparation:** suggest questions about assumptions, evidence, ablations,
  and limitations. Label your proposed experiments and hypotheses as suggestions.
- **Follow-up reading:** search the archive for related entries and explain the
  connection. Add outside papers when useful, identifying them as external reading.

Use available slides or transcripts for claims about the seminar itself. An archive
description or linked paper does not establish what a speaker said, demonstrated,
concluded, or discussed with the audience. Without a transcript or accessible recording
content, provide an archive overview or paper-based preparation and state that basis.
Never invent recording timestamps or quotes. If transcripts become available, cite
only timestamps actually present in those materials.

When explaining equations, check the paper's time direction, output parameterization,
conditioning, and normalization before translating them into the course notation.
Distinguish a reported result from your interpretation or a proposed extension.

## Keep the index current

The index is generated from the PDF and records its SHA-256. For a requested archive
update, regenerate it with `python research_seminars/scripts/build_index.py` and
validate with the same command plus `--check` (requires `pymupdf`). Keep entry dates,
recording URLs, and source links faithful to the PDF. The extractor expects the
current two-column archive layout; inspect boundaries if that layout changes.
