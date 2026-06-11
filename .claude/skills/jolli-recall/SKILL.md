---
name: jolli-recall
description: Recall prior development context from Jolli for the current branch
argument-hint: "[branch or keyword]"
user-invocable: true
jolli-skill-version: 0.99.1
---

# Jolli Recall

> Every commit deserves a Memory. Every memory deserves a Recall.

Load the structured development context for a branch — commits with their
distilled topics (trigger / response / decisions / files), plus any plans
and notes that the work referenced. Synthesize a grounded answer to the
user's prompt about that branch.

## Step 1: Parse the argument

The user's input is either a branch name (exact or fragment) or empty (use
the current git branch). Quote it when constructing bash to prevent shell
injection.

## Step 2: Run the CLI

```
"$HOME/.jolli/jollimemory/run-cli" recall "${ARGUMENTS}" --format json
```

If `~/.jolli/jollimemory/run-cli` does not exist, tell the user:
"Jolli not installed. Please install via `npm install -g @jolli.ai/cli && jolli enable` or install the Jolli VS Code extension."
Do not attempt further processing.

## Step 3: Handle the response

The output is JSON with a `type` field. Three cases:

### type: "recall" — full payload returned

You have a `RecallPayload` with these fields:

- `branch`, `period: { start, end }`, `commitCount`, `totalFilesChanged`,
  `totalInsertions`, `totalDeletions` — branch-level facts.
- `commits[]` — per-commit projection. Each carries:
  - identity (always present): `hash` (8-char display), `fullHash`, `branch`,
    `commitDate`, `commitAuthor`, `commitMessage`; optional `commitType?`,
    `ticketId?`.
  - `diffStats?` — `{ filesChanged, insertions, deletions }`.
  - `recap?` — 1-3 paragraphs of plain-English narrative.
  - `topics[]` — each with **always present**: `title`, **`decisions` (★)**;
    **may be absent**: `trigger?`, `response?`, `todo?`, `filesAffected?`,
    `category?`, `importance?`. `trigger` and `response` may be dropped by
    budget trimming; `decisions` is never dropped from a kept commit (if the
    budget can't fit it, the whole commit is omitted from `commits[]`).
  - `plans?` — `{ slug, title }[]` refs only; `slug` is the **normalized
    base slug** that always resolves to an entry in payload-level `plans`.
  - `notes?` — `{ id, title }[]` refs only; `id` always resolves to an
    entry in payload-level `notes`. (Notes use `id`, not `slug` — they
    have no archive-suffix mechanism.)
- `plans[]` — branch-deduplicated plan bodies: `{ slug, title, content? }`.
  `content` may be absent under tight budget — when absent, the entry is
  still a valid grounding anchor but you can't quote from it.
- `notes[]` — same shape and trimming rule as plans.
- `stats`, `estimatedTokens`, `truncated?`.

Render in two parts (in order):

#### Part A — Forced fact opener (no paraphrase, no interpretation)

Render the loaded confirmation as a heading + bullet block (not a prose
line). **Facts only — do not interpret what the branch is "about" here.**
The mandated shape:

```markdown
### Loaded `feature/auth`

- **Period:** 2026-04-10 → 2026-04-15 (5 days)
- **Commits:** 8 (+312 −89, 24 files)
- **Captured:** 12 topics, 5 key decisions, 2 plans, 3 notes
```

The heading + bullet shape is required — a single prose line blends into
the synthesis below and the user loses the visual anchor for verification.
Save interpretation for Part B.

#### Part B — Free-form synthesis

Pick whatever shape best serves the user's prompt: prose narrative,
chronological timeline, decision-focused bullet list, per-theme
`###` sections, side-by-side comparison, mixed. When multiple
distinct themes emerge across the commits, prefer `###` per theme —
inline-bold paragraph prefixes blend into a wall under markdown
rendering. The principles below are the only constraints.

#### Universal principles (apply regardless of shape)

1. **Lead with the answer.** No "Let me analyze..." or "Found N commits..."
   preamble.

2. **Ground every concrete claim** to a hash and/or file. Use `(abc12345)`
   for hashes and `[middleware/auth.ts](middleware/auth.ts)` for files.

3. **Synthesize, don't dump — but DO use verbatim quotes from stored
   data.** Read everything; fold into coherent prose or bullets.
   Whenever a phrase from `decisions` / `recap` / `plans[].content` /
   `notes[].content` captures the answer more compactly than your
   paraphrase, quote it verbatim in **bold** with attribution.

   Quote **complete clauses (typically 10-30 words)** — not 2-3 word
   fragments that depend on your surrounding paraphrase to mean
   anything. The reader should be able to skim the bold quote alone
   and understand its claim. Format, embedded in narrative:

   *The design chose JWT because* **"the stateless model lets us scale
   horizontally without a shared session store across regions"**
   *(decisions, abc12345)*; *per the auth-redesign plan,* **"all session
   tokens must be opaque, with no client-readable claims, so rotation
   never breaks the API"** *(plan: auth-redesign)*.

   **Bold = verbatim from stored data.** Never use bold for general
   emphasis. Quotes belong inside running prose or bullets that carry
   their own narrative — never as bare bullets stripped of context.
   Stringing bare quotes is the wall-of-fragments failure mode.

4. **Reply in the user's language.** Template is English; user-visible
   output matches the user.

5. **Don't expose machinery.** No "RecallPayload" / "commits array" /
   "JSON field" / "SearchHit" mentions.

6. **Brief by default — synthesize, don't dump every commit.** Skip
   routine commits and merge overlapping themes; aim for ~500 words
   on a typical branch, but favor section structure over compression.
   Never collapse `###` themes into inline-bold paragraph prefixes
   just to hit a word count — that produces a wall and defeats the
   structure's purpose. Branches with many distinct themes may
   legitimately run longer; a "deep dive" on a specific theme is
   opt-in.

#### Plan / note stubs on commits

When a commit carries `plans?` / `notes?` stubs, use the stub title as a
grounding anchor for narrative ("the auth-redesign plan guides this work").

**To quote from a plan or note body**, look up the matching entry in the
top-level `plans` / `notes` array by its `slug` (plans) or `id` (notes):

- If the entry has `content`: quote verbatim with `(plan: <slug>)` /
  `(note: <id>)` attribution if relevant to the user's prompt.
- If `content` is absent (budget trimming dropped the body): use **only**
  the title as a citation anchor — never fabricate a quote from a body
  you cannot see.

#### Empty / partial handling

- Empty `commits`: tell the user no records were found; suggest running
  `jolli enable` if they expected records.
- `truncated: true`: budget enforcement dropped fields or commits. Mention
  it with a one-liner if the user asks for deeper detail; otherwise stay
  silent.

### type: "catalog" — branch lookup needed

Returned when no exact branch match was found. Has a `branches[]` array
with `branch`, `commitCount`, `period`, `commitMessages`, `topicTitles?`.
If a `query` field is present, semantic-match the user's input against
`branch`, `commitMessages`, and `topicTitles` (the highest-signal source);
support cross-language matching and time-relative queries.

- One match: re-run `"$HOME/.jolli/jollimemory/run-cli" recall "<branch>" --format json`
  and continue from Step 3.
- Multiple matches: list candidates, ask user to choose.
- No matches: show the catalog, ask user to clarify.

### type: "error" — CLI returned a hard error

Has a `message` string. Common cases:

- Branch matched but its summaries failed to load.
- No records in the repo at all.
- Invalid argument or internal failure.

Surface the message verbatim to the user (translated into their language if
non-English). For "no records in this repo" specifically, suggest running
`jolli enable` if they expected records. Do NOT retry or fabricate a recall
payload from nothing.
