# NeuroLady_Final

## Before writing code or starting development

Whenever the user asks to **code, implement, build, or start developing** anything, first
**re-read and keep in context** the project's core guide docs before doing the work:

- `developer files/feature_description_guide.md` — how features are specified.
- `developer files/test_driven_development.md` — how tests are designed (a whole set of tests
  per requirement, and how tests bridge requirements to the architecture).
- `developer files/architecture.md` — the system architecture across six levels (UX, API,
  services, AI services, data ERD/DFD, infrastructure).
- The relevant feature file(s) in `developer files/features/` and their test spec(s) in
  `developer files/tests/`.
- For product context: `developer files/Project Concept.md`, `developer files/Audience.md`,
  `developer files/user_metrics.md`.

Do not start implementing from memory — reload these each time so the work follows the agreed
format (feature documented → requirements with IDs → full set of tests → then code).

### Docs-first, always (change requests too)

When the user requests a **new behavior, UX change, or fix** to how something already works, the
order is **always: update the documentation first, then implement from it** — never patch the code
directly and backfill the docs (or skip them). Concretely, for any such request:

1. **Refine the docs to capture the request precisely** — the relevant part of `architecture.md`
   (e.g. the UX flow in §1), the affected `developer files/features/F-<NNN>-*.md` (user flows,
   Gherkin, `FR-`/`NFR-` requirements), and the test spec. Encode exact screen order, transitions,
   screen contents, and copy so the doc alone is enough to build from.
2. **Then implement to match the updated docs**, and update the runnable tests accordingly.

The documentation is the source of truth; code follows the docs, not the other way around. This
applies even to small tweaks — write the intended behavior down first, then make the code match it.

**Docs-first is the DEFAULT, never a question.** Do not ask "docs-first or straight to code?" and do
not wait to be told — the user should never have to repeat it. On any reported defect or change
request, start by writing: the `ISS-<NNN>` entry (for defects), the `FR-`/`NFR-` requirements in the
owning feature file, the `TC-` cases in its mirror test spec, and any `architecture.md` change —
**then** implement. If a request seems too small for docs, it still gets the requirement it was
missing: every defect found so far in this project traced to a **missing requirement**, not a typo.

**Tests must EXECUTE the path, never grep the source.** A test that asserts on implementation *text*
(e.g. "does the handler mention `foo` before `bar`") stays green while that code path raises on its
first line — this exact mistake shipped a bug where every photo request died with a `TypeError` and
the user got **silence**, with 766 tests passing. Behavioural tests invoke the real function/handler
with fakes and assert on **observable outcomes**. A structural check may only ever be *additive* to
an executing test, never a substitute. Likewise, if a defect class cannot occur in the test
environment (e.g. lock contention under an in-memory DB with no concurrency), build the environment
that can reproduce it rather than assuming it is covered.

**Silence is always a defect.** No path — including an unhandled exception — may leave the user with
zero outbound messages. Any turn either answers or degrades in character.

## ⛔ NEVER destroy the heavy assets (READ THIS BEFORE ANY git checkout/merge/pull)

The image checkpoint (`image/models`, ~30G), its venv (`image/.venv`, ~6G) and `image/comfyui` have
been **destroyed twice** by the same mechanism, costing the developer real money and hours each time.
This is the single most expensive class of mistake in this project. The mechanism, exactly:

> A **tracked symlink** (git mode `120000`) that points back inside the repo gets committed. Later a
> `git checkout` / `git merge` / `git pull` **materializes that symlink over the real directory**, and
> git **silently deletes the gitignored contents** underneath to make room. The second time, the
> trigger was a **stale local `master`** that predated the symlink-removal commit — `origin/master`
> was clean, but checking out the behind-by-N local ref restored the landmine.

**Enforced guards are installed (run `sh scripts/install-git-hooks.sh` once per clone/worktree):**
- **`pre-commit`** refuses to commit *any* symlink → the landmine can never enter a ref again. This is
  the root-cause fix. Escape hatch only for a genuinely intended link: `ALLOW_SYMLINK_COMMIT=1`.
- **`post-checkout` / `post-merge`** abort loudly if a guarded heavy path became a symlink or vanished.
- **`scripts/safe-checkout.sh <ref>`** inspects the **TARGET** tree for symlinks *before* switching —
  use it (or `git ls-tree -r <ref> | awk '$1==120000'`) instead of a bare `git checkout`.

**Non-negotiable rules (the guards enforce these, but you follow them regardless):**
1. **Never commit a symlink.** Never create a symlink whose target is inside this repo. Share heavy
   assets via absolute paths in config/env (`IMAGE_COMFY_DIR`, `IMAGE_COMFY_PYTHON`,
   `IMAGE_MEDIA_ROOT`, `CHAT_MODEL_PATH`) — never via in-repo links.
2. **Before any `checkout`/`merge`/`pull`: `git fetch` first, then inspect the ref you are moving TO**
   (`git ls-tree -r <target> | awk '$1==120000'`), not the ref you are on. A `git ls-files` check on
   HEAD is worthless — it describes where you are, not where you are going. A local branch that is
   **behind** its remote is not "old", it is a **restored snapshot of a deleted landmine**.
3. **After any tree-touching git op, verify the heavy dirs are still real directories** (the guards do
   this automatically). A symlink where a real dir belongs = STOP, do not run another git command.
4. **Never `git add -A` / `git add <dir>` blindly** — `git status --short` first, stage explicit files.
5. **Before any `rm -rf`/`reset --hard`/`clean -fdx` recovery, look at what is actually on disk first.**
6. When you report a safety check as green, **state which ref you inspected.**

## Git workflow: commit and push after every change

After **any** change made to the project (new file, code edit, config change, etc.),
automatically:

1. `git add` the changed files.
2. `git commit` with a message following the versioning template below.
3. `git push origin master`.

No need to ask for permission for the commit/push itself — this is a standard step after
every unit of work on the project. Confirmation is only needed for unusual/dangerous git
operations (force push, reset --hard, history rewrite, etc.).

### Commit message format

```
v{MAJOR.MINOR.PATCH} [{type}]: {short description}
```

- `{MAJOR.MINOR.PATCH}` — project version, stored in `developer files/VERSION`
  (create it with value `0.1.0` if it doesn't exist yet).
- Update the version before every commit:
  - `fix` → PATCH +1
  - `add` / `feat` → MINOR +1, PATCH reset to 0
  - `refactor`, `docs`, `chore`, `style`, `test` → PATCH +1
  - Breaking/major changes → MAJOR +1, MINOR and PATCH reset to 0
- `{type}` — change type: `add`, `fix`, `refactor`, `docs`, `chore`, `style`, `test`.
- Description — specific, state exactly what was added/changed/fixed (no vague wording).

Example:
```
v0.2.0 [add]: added NeuroLady persona response generation module
v0.2.1 [fix]: fixed persona config loading error
```

## Feature branching and merging

When a separate, self-contained feature is being built (as opposed to a small doc/config edit
or a quick fix), it must be developed on its own dedicated branch rather than directly on
`master`:

1. Create a new branch for the feature (e.g. `feature/<short-name>`).
2. Commit and push work-in-progress changes to that branch as work proceeds (following the
   normal commit message/versioning rules above), not to `master`.
3. Before merging the feature branch into `master`, all tests located in the `tests/` folder
   must be run and must pass.
4. Only merge into `master` after all tests pass. If tests fail, fix the issues on the feature
   branch first — do not merge a branch with failing or skipped tests.

This rule applies going forward, once features start being implemented (application code,
not documentation-only changes).

Every feature is also documented as its own file in `developer files/features/`, following the
format defined in `developer files/feature_description_guide.md` (user stories → user flows →
Gherkin use cases → functional & non-functional requirements, each requirement with a stable ID).

Every requirement must be covered by **several tests** (never just one), designed per
`developer files/test_driven_development.md`.

**Test volume is proportional to feature granularity — do not chase a fixed huge number.** When
features are split finely (small, self-contained), aim for roughly **2-3 tests per requirement**
(≈100-150 tests for a typical feature), not thousands. Only broad, coarse-grained features warrant
very large suites. More is not automatically better; pick *varied, meaningful* cases (happy,
negative, boundary, error, concurrency, localization, integration, e2e) over sheer count.

**File-name matching + ID traceability (strict):**
- Each feature file `developer files/features/F-<NNN>-<slug>.md` has a **mirror test spec with the
  exact same file name** in `developer files/tests/F-<NNN>-<slug>.md`.
- **Every test must be addressed to a specific artifact ID from the feature file** — either a
  requirement (`FR-`/`NFR-`) or a user story (`US-`) — via its own ID:
  `TC-<FR|NFR|US-id>-<nn>` (e.g. `TC-FR-001-05-02`, `TC-NFR-001-01-01`, `TC-US-001-03-01`). The IDs
  on both sides must stay consistent so coverage is traceable both ways.

Runnable test code lives in the repo-root `tests/` folder, and all of it must pass before a feature
branch merges to `master`.

## Issue log (reported problems despite passing tests)

When the user reports that **a feature doesn't work, or the logic is wrong — even though all
tests pass** (or reports any behavior/logic problem regardless of test status), do the following:

1. Assign the next `ISS-<NNN>` id and log the report in `developer files/issue_log.md`, clearly
   formulated, with an unchecked `[ ]` "fixed" status and the current date.
2. Investigate *why the tests didn't catch it* — the gap is usually a missing/incorrect test, an
   architecture flaw, or a missing requirement.
3. Close the gap **at its source**: add/fix tests (new `TC-` cases) in the relevant test spec and
   `tests/` code, refine `architecture.md`, and/or add/adjust requirements (`FR-`/`NFR-`) in the
   feature file — whatever was under-covered.
4. Flip the issue's checkbox to `[x]`, record the resolution and date, and update the index table
   in `issue_log.md`.

This mechanism is how the architecture and coverage get modernized from real findings. Follow the
full format in `developer files/issue_log.md`.

## Automatically logging user feedback and preferences

If the user, during a conversation:
- points out that a mistake was made,
- says they want something done differently,
- says they don't like something (e.g. a specific UI type, code style, approach, etc.),

— this must be **immediately** appended to the "Preferences and feedback" section below in
this same file (CLAUDE.md), briefly and specifically, with a date. This is done so the same
undesired decisions aren't repeated in the future.

Entry format:
```
- [YYYY-MM-DD] <the essence of the preference/mistake — what to avoid, or what to do instead>
```

## Project status file

Maintain a `developer files/PROJECT_STATUS.md` file that describes, in technical detail,
everything that has been done in the project so far (setup steps, files/modules created,
architecture decisions, configuration, fixes, integrations, etc.). Its purpose is to preserve
context across sessions so work can be picked up without re-deriving history from `git log`.

Rules:
- Update `PROJECT_STATUS.md` every time a meaningful change is made to the project (new
  feature, fix, config change, integration, etc.) — not for trivial/no-op edits.
- Write concrete technical details (what was added, how it works, relevant file paths,
  decisions made and why), not vague summaries.
- Organize it by topic/component rather than strictly chronologically, so it stays readable
  as the project grows; keep a "Recent changes" section at the top for the latest updates.
- Include `PROJECT_STATUS.md` updates in the same commit as the change they describe,
  following the normal commit/push workflow above.

## Language

All `.md` files in this project (this file included) must be written in English.

## Developer files folder

All working/context documentation created for the project (product concept, audience,
research notes, planning docs, etc.) is stored inside `developer files/` — this is meant to be
the single place to look for developer context on the project. The only exception is this
`CLAUDE.md` file itself, which must stay at the repo root so it's auto-loaded by Claude Code.

When creating a new `.md` doc for the project, put it in `developer files/` unless the user
explicitly asks for it elsewhere.

## Preferences and feedback

- [2026-07-27] **A feature that spans both media must be delivered for both media — and a
  "boundary" is not an excuse to render nothing.** ISS-035's first cut wired the LLM scene author
  into the photo pipeline and left the clip authors on templates; the operator called the output
  unusable and named three things as part of acceptance, not extras: **(a)** one author,
  `medium="photo"|"video"`, used by F-011/F-014 *and* F-017/F-016; **(b)** the **served** intimacy
  level (post-FR-014-20 clamp) must actually drive wardrobe, framing, pose and action —
  `author_intimate_prompt`'s output was *"theoretical"*, and vague filler at a level the gate
  approved is a defect, not a safe default. The hard limits (never above the served level, never
  above `PLATFORM_MAX_INTIMACY_LEVEL`, and for video the F-016 FR-016-14 content tier **read from
  config, never hard-coded**) exist to make the levels mean something, not to avoid rendering them;
  **(c)** identity preservation is code-appended to **every** prompt on both media and is never
  omittable, and video negatives keep the identity-drift guards (ISS-032: a clip started as one
  woman and ended as another). Also: **the authoring instruction is the product** — write it long
  and structured with worked few-shot examples at more than one intimacy level; a thin instruction
  produces filler no matter how good the plumbing is.
- [2026-07-23] **Stop asking whether to do docs-first — just do it.** The user had to repeat
  "docs-first" on nearly every request despite the rule already existing. Docs-first is the default
  for *every* defect and change: ISS entry → requirements → TC cases → architecture → then code.
  Also reinforced from this session's real failures: (a) **tests must execute the path, not grep the
  source** — a source-grepping regression test stayed green while every photo request died with a
  `TypeError` and the user got silence; (b) **silence is always a defect** — every inbound message
  must end in a visible reply; (c) when a defect class cannot occur in the test environment (no
  concurrency under in-memory SQLite), build an environment that reproduces it instead of assuming
  coverage.
- [2026-07-17] **NEVER commit worktree/machine-local plumbing — it destroyed a 28G model and cost
  the developer real time/money.** Root cause chain (all must be avoided): (1) to share heavy
  gitignored assets (`image/models` = 28G v23 checkpoint, `image/comfyui`, `image/.venv`) across
  isolated worktrees, symlinks were created **inside the repo pointing back at the same repo's real
  asset paths** — a self-referencing landmine at the exact path of real data; (2) a subdirectory
  `.gitignore` used **repo-root-relative patterns** (`image/.venv` inside `image/.gitignore`), which
  are actually directory-relative → matched nothing → the symlinks were **not** ignored; (3) a broad
  `git add`/`git add -A` staged those symlinks as tracked files (mode 120000) and they were committed
  and pushed; (4) when that branch merged into another working tree, git **materialized the tracked
  self-symlink over the real directory, deleting the 28G checkpoint + ComfyUI + venv**. Binding rules
  going forward: **(a)** never create a symlink whose target is a path inside the same git repo,
  above all at a path holding real or gitignored heavy data — share cross-worktree assets via
  **absolute paths injected through config/env** (this project already has `IMAGE_COMFY_DIR`,
  `IMAGE_COMFY_PYTHON`, `IMAGE_MEDIA_ROOT`, `CHAT_MODEL_PATH` for exactly this), not via in-repo
  symlinks; **(b)** never `git add -A` or `git add <dir>` blindly — always `git status --short`
  first and stage **explicit files**, especially in a worktree; **(c)** `.gitignore` patterns are
  relative to the file's own directory — verify with `git check-ignore -v <path>` before trusting
  them; **(d)** treat gitignored heavy assets (models/checkpoints/venvs) as sacred: confirm
  `git ls-files -s | awk '$1==120000'` is empty before pushing, and after any merge/checkout that
  touches the tree, verify the heavy assets still exist (a symlink where a real dir belongs = STOP);
  **(d2) [2026-07-23 — this rule was not enough, and the 30G checkpoint was destroyed a SECOND
  time]** `git ls-files` describes the ref you are **currently on**, which is worthless as a
  pre-checkout safety check — the danger lives in the ref you are about to switch **to**. Before any
  `git checkout <ref>` / `git merge <ref>` / `git pull`, inspect the **target** tree:
  `git ls-tree -r <ref> | awk '$1==120000'` **and** `git ls-tree -r origin/<ref> | awk '$1==120000'`.
  Root cause the second time: a **stale local `master`** that predated the v0.53.1 symlink-removal
  commit. `origin/master` was clean, the feature branch was clean, and checking out the local
  `master` still materialized three self-referencing symlinks over `image/{models,comfyui,.venv}` —
  git deletes **gitignored** directories without complaint to make room for a tracked symlink.
  Always `git fetch` and confirm the local ref is not behind before checking it out; a local branch
  that is behind is not "old", it is a **restored snapshot of a deleted landmine**. And never trust
  a green check performed on the wrong ref — state explicitly which ref was inspected;
  **(e)** before any `rm -rf` "recovery", inspect what is actually there first — a wrong assumption
  compounds the loss.
- [2026-07-10] User wants CLAUDE.md and all future .md files in this project written in English, not Russian.
- [2026-07-10] User wants PROJECT_STATUS.md and VERSION kept inside a `developer files/`
  subfolder, but CLAUDE.md must stay at the repo root (not moved into that subfolder), so
  it's picked up automatically by Claude Code.
- [2026-07-10] User wants all project documentation files (e.g. Audience.md, Project
  Concept.md) stored inside `developer files/` so Claude can look there for developer context.
- [2026-07-12] Test volume must scale with feature granularity: for finely-split features, ~2-3
  tests per requirement (≈100-150 per feature), not a fixed 1000+. Test spec file names must mirror
  the feature file names, and every test must be addressed to a specific `FR-`/`NFR-`/`US-` id via a
  consistent `TC-` id. Prefer varied, meaningful cases over raw count.
- [2026-07-12] **Describe every feature maximally in detail** — do not compress important features.
  F-002 was under-detailed relative to F-001; going forward each feature file must be thorough:
  exhaustive user stories per relevant audience segment, complete user-flow diagrams, a rich set of
  Gherkin use cases, and a full, granular set of `FR-`/`NFR-` requirements covering every facet of
  the feature (not a minimal set). More detail in the feature spec is wanted, especially for the
  important features. (This is about spec thoroughness; the ~2-3-tests-per-requirement rule above
  still governs test count — more requirements simply yield more tests overall.)
- [2026-07-12] Don't stack two consecutive bot messages that both nudge the user toward the same
  action (e.g. an intro line asking "write me?" immediately followed by a separate "ready — say
  something" message). It reads as robotic/redundant. Combine into one message and attach whatever
  keyboard/markup is needed directly to it, rather than sending a second follow-up text.
- [2026-07-13] **Docs-first workflow (reaffirmed by the user):** on any behavior/UX/fix request,
  first update the documentation (architecture §1 UX flow, the feature file, tests) to capture the
  exact screen order/transitions/contents, then implement strictly from the updated docs. Never code
  first and document after. See "Docs-first, always" above.
- [2026-07-26] **Never report generated media as "done" without LOOKING at it — and never ship a
  one-line motion prompt.** Five clips were delivered as a green result on the strength of
  `mp4=ok` plus a format check (h264 / 16 fps / 65 frames / faststart / inside the 90 s budget).
  The user opened them and they were unusable: violent colour and exposure strobing between frames,
  the face changing identity almost frame to frame, and motion that ignored the prompt entirely
  (asked for a head turn, got arms flailing overhead; asked for walking, got a near-frozen subject).
  **The container was validated, the content never was.** Binding rules going forward:
  **(a)** before reporting any generated image/video, actually inspect it — extract a frame strip
  (`ffmpeg … tile=Nx1`) and *view* it; a passing format probe is not evidence of quality and must
  never be presented as if it were;
  **(b)** motion/scene prompts must be **long and structured** — subject and wardrobe, the action
  broken into beats, explicit camera behaviour (static / lens / no pan-zoom-shake), lighting held
  constant, and the film/grade style. Lazy fragments like "slowly turns her head, natural subtle
  motion, cinematic" give the model no control and are not acceptable;
  **(c)** the negative prompt must **name the artifacts actually observed** (flickering, exposure
  shift, colour shift, identity drift, morphing, warping, extra people…), not generic "blurry, bad
  quality";
  **(d)** check the generator's **native operating point** before choosing config — Wan 2.2 is a
  **24 fps** model and the bench muxed its output at 16 fps, stretching ~2.7 s of motion into 4.06 s,
  which is *why* everything looked sluggish; and 480×480 is roughly a quarter of the model's native
  resolution, which is *why* it strobed. Off-distribution settings were chosen purely to hit a speed
  target, and no requirement anywhere defined an acceptable-quality bar (see ISS-032);
  **(e)** do not squash a non-square reference into a square frame — match the source aspect.
