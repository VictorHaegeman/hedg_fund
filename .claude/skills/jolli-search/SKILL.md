---
name: jolli-search
description: Search structured commit memories across all branches — decisions, topics, files
argument-hint: "<keyword> [--since 2w]"
user-invocable: true
jolli-skill-version: 0.99.1
---

# Jolli Search

Search structured commit memories across every branch in this repo.
Uses LLM-distilled summaries (decisions, topic titles, recap, filesAffected) —
not raw markdown — so semantic / cross-language / synonym matching is a fit
for the chat LLM that runs this skill.

## When to use

- "Has anyone dealt with X before?" / "How have we handled Y previously?"
- Looking for a past decision: "why did we choose X over Y?"
- Finding the commit related to a half-remembered ticket / file / topic.

## When NOT to use

- Need full context of a known branch → `/jolli-recall <branch>`.
- Looking at the current code → grep / read files directly.

## Step 1: Parse ${ARGUMENTS} into query + flags

The user can include flags inline, e.g. `/jolli-search auth --since 2w`. Before
invoking the CLI, split the argument string into two parts:

1. **Query**: the user's keyword / sentence (everything that is NOT a CLI flag).
   The query may be in any human language and contain natural punctuation
   (`?`, `#`, `(`, etc.).
2. **Flags**: any of `--since <date>`, `--limit <n>`, `--budget <tokens>`,
   `--output <path>`. Pass these through verbatim. Ignore any other token starting
   with `--`.

Quote ONLY the query when constructing bash; pass flags as separate unquoted
tokens. Examples:

| User input                                  | Bash you should run                                                                                |
|---------------------------------------------|----------------------------------------------------------------------------------------------------|
| `auth`                                    | `"$HOME/.jolli/jollimemory/run-cli" search "auth" --format json`                                |
| `auth --since 2w`                         | `"$HOME/.jolli/jollimemory/run-cli" search "auth" --since 2w --format json`                     |
| `why did we choose X over Y? --since 1m`  | `"$HOME/.jolli/jollimemory/run-cli" search "why did we choose X over Y?" --since 1m --format json` |

Always include `--format json`. Never put flags inside the query quotes.

## Step 2: Get the catalog

Run the bash command you constructed in Step 1.

**Failure handling**:
- If `~/.jolli/jollimemory/run-cli` does not exist: tell the user
  "Jolli not installed. Please install via `npm install -g @jolli.ai/cli && jolli enable`
  or install the Jolli VS Code extension." Do not attempt further processing.
- If the command output starts with `error:` or contains `unknown command 'search'`:
  the installed CLI is older than this skill. Tell the user
  "Your installed Jolli CLI is older than this skill — please run
  `npm update -g @jolli.ai/cli` (or update your VS Code extension), then retry."
  Do not attempt further processing.

The output is a JSON object with `type: "search-catalog"` containing one entry per
recent root commit, each with `branch`, `date`, `recap`, `ticketId`, and `topics`
(title / decisions / category / importance / filesAffected).

### What the output fields mean (don't conflate them)

- `totalCandidates`: number of root commits matching the time-window filter
  (i.e. catalog universe size after `--since` / `--limit`).
- `entries`: the actual array you'll scan — usually fewer than `totalCandidates`
  because `--budget` may have trimmed less-recent ones to fit the token cap.
- `truncated: true` means at least one candidate didn't make it into `entries`.

**The catalog is NOT pre-filtered by the user's query**. It's a recent-commits
window — your job in Step 3 is to do the semantic filter. Don't be surprised
if many entries look unrelated; that's expected.

## Step 3: Pick relevant commit hashes (semantic, not literal)

**Read each entry's `title`, `recap`, `decisions`, and `filesAffected` and judge
semantic relevance to the user's intent.** The query and the entries may be in
different human languages. Cross-language and synonym matching is YOUR job here.

- Pick **5-10** commits whose meaning relates to the query. The Phase 2 payload
  carries full topic content per hit (trigger / response / decisions / files);
  picking more than 10 risks blowing the chat context budget for ambitious
  queries.
- Don't worry if no entry contains the literal query token — that's exactly what
  semantic picking is for.

**If `truncated` is `true` and the entries you see don't include obvious matches**,
silently retry Step 2 once with `--budget 50000` to see more of the corpus before
asking the user to narrow `--since`. Only ask the user to narrow time if the
larger budget still doesn't surface relevant commits.

### Internal constraints

- **DO** read the JSON inline from the previous tool result.
- **DO NOT** process programmatically — no temp files, no jq/python/grep
  scoring scripts. Semantic picking by reading is the whole point.

## Step 4: Load full content for the picks

Construct the Phase 2 bash with the same query quoting + flag separation rule
from Step 1, plus `--hashes <fullHash1,fullHash2,fullHash3>`:

```
"$HOME/.jolli/jollimemory/run-cli" search "<the query>" --hashes <fullHash1>,<fullHash2>,<fullHash3> --format json
```

**Use `hit.fullHash` (40-char SHA), NOT `hit.hash` (8-char display)**. The
CLI rejects abbreviated hashes — the 8-char display `hash` is for showing in
your output to the user, but Phase 2 lookup needs the unambiguous full SHA
(otherwise cherry-pick / rebase chains can resolve to the wrong commit silently).

Output is a `SearchResult` with `results: SearchHit[]`. See Step 5 for the schema and how to render.

## Step 5: Render to the user

The CLI gave you structured data — full distilled content per commit.
**Output shape is entirely your call.** Pick whatever serves the user's
query: prose, table, timeline, side-by-side, mixed. **The principles below
are the only constraints.**

### A. The data you have (per hit)

Each `results[i]` is a `SearchHit`:

**Identity / provenance**:
- `hash` — 8-char short SHA (plain text, no link)
- `fullHash` — 40-char full SHA
- `commitMessage` — raw subject (fallback for label; `recap` is usually better)
- `commitAuthor` — for "who worked on X" queries
- `commitDate` — ISO 8601
- `branch`
- `commitType?` — `"commit"` / `"amend"` / `"squash"` / `"rebase-pick"` / `"cherry-pick"` / `"revert"`; helps distinguish routine commits from consolidated ones
- `ticketId?` — render as `[TICKET-1234]` badge

**Change scale**:
- `diffStats?` — `{ filesChanged, insertions, deletions }`

**Narrative**:
- `recap?` — 1-3 paragraphs of plain-English narrative. Highest-quality prose; primary source for "what is X" / "explain X".

**Topics** — `topics: SearchHitTopic[]` (★ the meat):

  - `title` — one-sentence label
  - `trigger?` — 1-2 sentences, what prompted the work
  - `response?` — implementation summary, may include code; longest field
  - `decisions` ★ **THE STAR FIELD** — design choices + *why*, as markdown bullets. Primary source for "why did we choose X" / "what alternatives" / "rationale". Not in the diff; only here.
  - `todo?` — residual work the LLM flagged (rare)
  - `filesAffected?` — per-topic file list. Render as markdown links: `[cli/src/Types.ts](cli/src/Types.ts)`.
  - `category?` — `feature` / `bugfix` / `refactor` / `tech-debt` / `docs` / `test` / `devops` / `ux`
  - `importance?` — `major` / `minor`

**Plan / note stubs**:
- `plans?` — `{ slug, title }[]` — plan refs this commit declared. Search ships only stubs (no plan body); use the title as a grounding anchor in your narrative ("the decision is consistent with the auth-redesign plan referenced by this commit"). **Do NOT promise the user they can navigate to the plan body from search** — search Phase 2 carries no plan content.
- `notes?` — `{ id, title }[]` — same shape and rule as plans.

### B. Universal principles (apply regardless of shape)

1. **Lead with the answer.** No "Let me analyze..." or "Found N commits..." preamble.

2. **Ground every concrete claim** to a hash and/or file. Use `(abc1234)` for hashes and `[cli/src/Types.ts](cli/src/Types.ts)` for files.

3. **Synthesize, don't dump — but DO use verbatim quotes from stored data.** Read everything; fold into coherent prose or bullets. Whenever a phrase from `recap` or `decisions` captures the answer more compactly than your paraphrase, quote it verbatim in **bold** with attribution.

   Quote **complete clauses (typically 10-30 words)** — not 2-3 word fragments that depend on your surrounding paraphrase to mean anything. The reader should be able to skim the bold quote alone and understand its claim. Format, embedded in narrative: *the design chose JWT because* **"the stateless model lets us scale horizontally without a shared session store across regions"** *(decisions, abc1234)*.

   **Bold = verbatim from stored data.** Never use bold for general emphasis. Quotes belong inside running prose or bullets that carry their own narrative — never as bare bullets stripped of context. Stringing bare quotes is the wall-of-fragments failure mode.

4. **Reply in the user's language.** Template is English; user-visible output matches the user.

5. **Don't expose machinery.** No "Phase 1" / "Phase 2" / "catalog" / "SearchCatalog" / "SearchHit" / "JSON field" mentions.

### C. Output shape

Your call. The only hard rule: every concrete claim must be groundable to a hash or file (principle 2). If the picks share an obvious unifying theme (same `branch` / `ticketId` / initiative), name it.

### D. Empty / partial / failed-hash handling

This is the **only** place where it is appropriate to mention search-machinery
state (catalog size, truncation, hash load failures). Stay silent about it
when results are healthy.

- If `results` is empty: tell the user no usable content was found and suggest
  broader keywords or a wider `--since`. Coverage chatter is appropriate here:
  *"Scanned N candidates from the last <window>; none matched."*
- If the catalog was truncated (`truncated: true` from Phase 1) **and** the
  picks feel thin: *"Catalog hit the token budget; rerun with `--budget 50000`
  to widen the search."*
- If `failedHashes` is non-empty: mention which picks couldn't be loaded so the
  user knows the search isn't complete.
