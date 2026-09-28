# Research Seminars on Diffusion Models

[Repository overview](../README.md) · [Visual GenAI course](../visual_genai_course_2026/README.md)

A research seminar series on diffusion models and related generative methods.
Use the archive to find talks, recordings, and the papers discussed. The archive
descriptions are in English; recordings are in Russian.

## Materials

* [Searchable seminar index](INDEX.md) — browse 151 entries by topic, paper title, speaker, or date.
* [Seminar recording archive (2023–2026)](diffusion_research_seminars_09_2026.pdf) — English guide with recording links and access codes.
* [Seminar slides](slides/)
* [Research seminar Telegram group](https://t.me/+gE2ERaknHecyMjZi)

## Research seminar assistant

The [Diffusion Research Seminars skill](../.agents/skills/diffusion-research-seminars/SKILL.md)
helps you find relevant talks, plan prerequisite reading, compare papers, and prepare
discussion questions. You can ask questions in English or Russian.

### Get started

1. Clone this repository, or update your existing checkout, and open its root folder
   in **Codex CLI or the Codex IDE extension**.
2. Codex discovers the included skill in
   [`.agents/skills/diffusion-research-seminars/`](../.agents/skills/diffusion-research-seminars/SKILL.md);
   no separate skill installation is needed. See the
   [official skill documentation](https://learn.chatgpt.com/docs/build-skills).
3. Type `$` and select `diffusion-research-seminars`, or include its name in your prompt:

```text
$diffusion-research-seminars
Find seminars on diffusion language models and suggest a reading order.
I know image diffusion but am new to text generation.
Link the recordings and explain which papers I should read first.
```

If the skill does not appear, restart Codex after opening the repository.

### What to ask

* “Find the seminar covering MeanFlow and prepare a reading guide from its paper.”
* “Find the DMD and consistency-model talks, compare their objectives, and cite the papers.”
* “Suggest discussion questions and possible follow-up experiments for this seminar.”

The skill uses the archive to locate talks and reads linked papers for technical
explanations. Claims about what a speaker said, and recording timestamps, require
accessible recording content or transcripts. The current archive includes recording
links and paper references; the slides folder is a placeholder.

For foundational lectures and exercises, start with the [Visual GenAI course](../visual_genai_course_2026/README.md).
