# README Visual Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the RSI GraphRAG README easier to scan and review by adding a visual overview, splitting the dense pipeline into three readable Mermaid diagrams, and restructuring the entry flow without changing research claims.

**Architecture:** Keep README as the single research entry point. Add one repository-local PNG concept map for non-technical orientation, while using Mermaid for exact technical flows. Preserve existing sections, anchors, links, metrics, and reproduction commands; improve the top-level order and add concise signposts.

**Tech Stack:** GitHub-flavored Markdown, Mermaid flowcharts, generated PNG asset, Python standard-library validation scripts.

**Spec:** `docs/superpowers/specs/2026-11-12-readme-visual-refresh-design.md`

## Global Constraints

- Generated images must be stored in the repository and referenced with relative paths.
- Technical flows should use Mermaid so nodes and code/data flow remain reviewable.
- Image text should be minimal; key terms remain in README prose.
- Existing Chinese README structure and links must remain usable.
- Do not change code, experiment results, or metric definitions.

## Review Focus

- Relative image path resolves from the README location; test that the referenced asset exists.
- Mermaid fences are balanced and each diagram has a readable number of primary nodes; validate fence pairing and diagram labels.
- Existing internal anchors and links remain present after reordering; scan all relative Markdown targets and anchor references.
- README remains legible when rendered without custom CSS; inspect the generated Markdown structure and headings.
- The concept image does not imply unsupported claims; inspect dimensions, alpha/background, and visual content before linking it.

### Task 1: Generate and validate the concept overview asset

**Files:**
- Create: `docs/assets/rsigraph-rag-overview.png`
- Inspect: generated image output before copying into the repository

**Interfaces:**
- Produces a repository-local PNG referenced by `README.md`.

- [ ] **Step 1: Generate the concept image**

  Use the built-in image generation tool with an infographic-diagram prompt: a clean horizontal research-system overview showing three connected stages—documents/read-time understanding, graph retrieval/question-time answering, and feedback/skill memory—with abstract document cards, graph nodes, question/answer markers, restrained blue/teal palette, no logos, no watermark, and no dense small text.

- [ ] **Step 2: Inspect the generated image**

  Confirm the three stages are visually distinct, the arrows read left to right, and there is no unreadable pseudo-text or accidental branding.

- [ ] **Step 3: Copy the selected image into `docs/assets/rsigraph-rag-overview.png`**

  Preserve the generated PNG and use a stable filename.

### Task 2: Restructure README entry and split the technical flow

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: existing README sections, metrics, and links.
- Produces: a GitHub-renderable README with a fast-start overview, one concept image, and three stage-specific Mermaid diagrams.

- [ ] **Step 1: Add a visual entry block after the title**

  Add a short “先看这里” block containing the project definition, audience path (结论 / 方法 / 证据 / 复现), the concept image, and compact status chips or bullets linking to the existing sections.

- [ ] **Step 2: Rewrite the current one-minute conclusion as scan-friendly cards**

  Preserve all existing numerical claims, but group them into three short subsections: what improved, what did not improve, and what should happen next. Keep links to DetectiveQA details and supporting sections.

- [ ] **Step 3: Replace the monolithic pipeline diagram**

  Introduce three diagrams under a clear “三段 pipeline” heading:
  - Read-Time: document → OpenIE → grouping → comprehension → verification → graph variant.
  - Question-Time: question/cutoff → decomposition → retrieval → sufficiency → reader → grounded answer.
  - RSI Memory: reflection → sanitization → four gates → versioned memory → dev snapshot.

  Limit each diagram to the main flow and add a one-line code pointer beneath it.

- [ ] **Step 4: Add lightweight reading signposts before the detailed tables**

  Add short lead sentences and links before the experiment progress, dataset comparison, diagnosis, and repository map sections so readers can choose depth without deleting existing detail.

- [ ] **Step 5: Preserve all existing detailed sections and commands**

  Keep the tables, evidence numbers, repository map, source/sync notes, excluded files, and reproduction commands intact except for whitespace and local heading adjustments needed by the new entry flow.

### Task 3: Validate the rendered Markdown structure

**Files:**
- Read: `README.md`
- Read: `docs/assets/rsigraph-rag-overview.png`

- [ ] **Step 1: Run a structural Markdown check**

  Verify the image path exists, Mermaid code fences are balanced, headings are ordered, and all relative Markdown links resolve to files or known anchors.

- [ ] **Step 2: Inspect image metadata**

  Use a local image probe to confirm the PNG opens and has a reasonable landscape size for README display.

- [ ] **Step 3: Review the diff**

  Confirm no experiment values, code paths, or reproduction commands were accidentally altered.

- [ ] **Step 4: Report validation evidence**

  Summarize the checks and identify any GitHub-only rendering behavior that cannot be fully reproduced locally.
