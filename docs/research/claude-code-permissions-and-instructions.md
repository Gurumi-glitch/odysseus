# Claude Code permissions and instruction files

Research for wayfinder ticket #22 (map #20). Sources: official Claude Code docs at code.claude.com (fetched 2026-10-10) and the open-source `anthropics/claude-agent-sdk-python` repo. No other sources were used. Version notes such as "v2.1.211" come from the docs as published; behaviour may drift.

Short links used below:

- PERM = https://code.claude.com/docs/en/permissions
- MODES = https://code.claude.com/docs/en/permission-modes
- SET = https://code.claude.com/docs/en/settings
- MEM = https://code.claude.com/docs/en/memory
- HOOKS = https://code.claude.com/docs/en/hooks
- TOOLS = https://code.claude.com/docs/en/tools-reference
- SDKPERM = https://code.claude.com/docs/en/agent-sdk/permissions
- SDKINPUT = https://code.claude.com/docs/en/agent-sdk/user-input
- SDKSYS = https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts
- SDKTYPES = https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/types.py

(The `iam` page is now about authentication only and has no permission-model content: https://code.claude.com/docs/en/iam.)

## Answer summary

1. Permissions are enforced by the harness, not the model. CLAUDE.md and the prompt shape what Claude tries but cannot grant or revoke access. [PERM#manage-permissions]
2. A rule is `Tool` or `Tool(specifier)`. It lives in one of three lists: `allow`, `ask` or `deny`. Evaluation is deny, then ask, then allow; the first match wins and specificity does not matter. A narrow allow cannot carve an exception out of a broad deny or ask. [PERM#manage-permissions]
3. Specifiers depend on the tool: a command pattern for Bash (`Bash(git push *)`, where `:*` is the same as ` *`), gitignore-style paths for Read and Edit, `domain:` for WebFetch, and a server or tool name for MCP. The harness splits compound shell commands, and each subcommand must match on its own. [PERM#permission-rule-syntax]
4. A bare-name deny (`Bash`) removes the tool from the model's context. A scoped deny (`Bash(rm *)`) keeps the tool and blocks matching calls. [PERM#manage-permissions]
5. Modes set the baseline, and rules layer on top. The modes are `default` (Manual), `acceptEdits`, `plan`, `auto`, `dontAsk` and `bypassPermissions`. Deny rules and explicit ask rules hold in every mode. [MODES#available-modes]
6. `auto` mode sends unresolved actions to a separate classifier model. That model sees user messages, tool calls and CLAUDE.md, but tool results are stripped out. It also reads boundaries and approvals the user states in conversation, but it does not store them as rules. [MODES#how-auto-mode-evaluates-actions]
7. Settings precedence, from highest: managed, then the command line (`--settings`), then local (`.claude/settings.local.json`), then project (`.claude/settings.json`), then user (`~/.claude/settings.json`). List keys such as `permissions.allow` merge across files. A deny at any level wins over an allow at any level. [SET#settings-precedence], [PERM#settings-precedence]
8. A project's `allow` rules apply only after the user accepts workspace trust. Its `deny` and `ask` rules always apply. [PERM#project-allow-rules-and-workspace-trust]
9. The ask prompt offers: Yes (once); "Yes, and don't ask again" (a rule); sometimes "switch to auto mode"; and No. "Don't ask again" persists a rule to `.claude/settings.local.json` for Bash and WebFetch, but lasts only until the session ends for file edits. A prompt offers that option only when it can show everything the rule would allow. [PERM#permission-system]
10. In the SDK, the ask becomes a `canUseTool(tool, input, context)` callback. It returns Allow (optionally with `updated_input` and `updated_permissions`) or Deny (`message`, `interrupt`). `context.suggestions` carries ready-made rule updates whose destination is `session`, `localSettings`, `projectSettings` or `userSettings`. [SDKINPUT], [SDKTYPES]
11. Hooks: `PreToolUse` can return allow, deny, ask or defer and can rewrite the input, but it cannot override deny or ask rules. `PermissionRequest` fires only when a prompt is about to appear. `Stop` can block the end of a turn, capped at 8 consecutive continuations. [HOOKS#pretooluse-decision-control], [HOOKS#permissionrequest], [HOOKS#stop]
12. How the model "decides to ask": it calls `AskUserQuestion`, a tool for clarifying questions. Permission prompts, by contrast, come from the harness. The `claude_code` system-prompt preset carries the "security and safety instructions", but the docs do not publish its text. [TOOLS#askuserquestion-tool-behavior], [SDKSYS]
13. CLAUDE.md files come at four scopes: managed, user (`~/.claude/CLAUDE.md`), project (`./CLAUDE.md` or `./.claude/CLAUDE.md`) and local (`./CLAUDE.local.md`), plus `.claude/rules/*.md`. Files in ancestor directories load at launch. Files in subdirectories load lazily, once Claude touches a file there. `@path` imports nest up to four hops. [MEM#choose-where-to-put-claude-md-files], [MEM#how-claude-md-files-load], [MEM#import-additional-files]
14. Instruction files have no authority over the system prompt. CLAUDE.md is "delivered as a user message after the system prompt" and treated "as context, not enforced configuration". All files are concatenated, broad scope first, and conflicts are resolved arbitrarily. [MEM#claude-isnt-following-my-claudemd], [MEM#claudemd-vs-auto-memory]
15. Auto memory is model-written notes in `~/.claude/projects/<repo>/memory/`. `MEMORY.md` is an index whose first 200 lines or 25 KB load every session; topic files load on demand. The directory is per repository and local to the machine. CLAUDE.md, by contrast, is user-written instructions. [MEM#auto-memory]

## Permissions

### Rule syntax

- Format: `Tool` or `Tool(specifier)`. `Bash` and `Bash(*)` are equivalent. [PERM#permission-rule-syntax]
- **Bash**: `*` matches any text, including spaces. A trailing ` *` also matches the bare command (`Bash(ls *)` matches `ls`, not `lsof`). `Bash(ls:*)` is the same as `Bash(ls *)`. Put the `*` after the subcommand: `Bash(git *)` allows every git command. [PERM#wildcard-patterns]
- **Compound commands**: the separators are `&&`, `||`, `;`, `|`, `|&`, `&` and newline. Every subcommand must match an allow rule. A deny or ask rule fires if *any* subcommand matches, including one nested in `$(...)`, a subshell or a loop body. [PERM#compound-commands]
- **Wrappers**: the harness strips `timeout`, `time`, `nice`, `nohup`, `stdbuf`, `command`, `builtin`, `noglob` and a flagless `xargs` before matching. Runners such as `npx`, `docker exec` and `devbox run` are *not* stripped. [PERM#process-wrappers]
- **Not a security boundary**: `Bash(git push *)` does not stop `git -C . push`, `/usr/bin/curl` or `sh -c '...'`. Enforcement that doesn't depend on command text needs the sandbox or a PreToolUse hook. [PERM#bash-rule-limits]
- **Built-in read-only commands** (`ls`, `cat`, `grep`, read-only `git`, and others) never prompt. The set is not configurable. [PERM#read-only-commands]
- **Read/Edit**: gitignore patterns with four anchors: `//abs`, `~/home`, `/relative-to-settings-file` and `relative-to-cwd`. `Edit(...)` covers every file-writing tool, and a `Write(...)` rule is never consulted. A Read deny also blocks Edit and Write on the same path. [PERM#read-and-edit]
- **WebFetch**: `WebFetch(domain:*.example.com)`. A mid-pattern `*` never crosses a dot. [PERM#webfetch]
- **MCP**: `mcp__server`, `mcp__server__*` or `mcp__server__tool`. An allow glob must be anchored to a literal server name. [PERM#mcp], [PERM#tool-name-wildcards]
- **Parameter rules** (deny and ask only): `Agent(model:opus)` and `Bash(run_in_background:true)`. A parameter rule cannot match a tool's primary content field, such as `command`, `file_path` or `url`. [PERM#match-by-input-parameter]

### Evaluation order (SDK view, the most explicit statement)

hooks, then deny rules, then ask rules, then permission mode, then allow rules (plus a tool's own "needs no approval" logic), then `canUseTool`. The details by step: [SDKPERM#how-permissions-are-evaluated]

- A hook `allow` does not skip deny or ask rules. A hook deny or exit code 2 blocks even in `bypassPermissions`.
- An ask rule always falls through to the callback, even in `bypassPermissions`.
- `plan` mode routes file edits and shell writes to the callback regardless of allow rules.
- Auto-approved calls never reach `canUseTool`, so checks that must see every call belong in a PreToolUse hook.
- "Actions no mode auto-approves": explicit ask rules, tools that require user interaction (`AskUserQuestion`, MCP `requiresUserInteraction`), `rm` or `rmdir` on critical paths such as `/` or `~`, and a few others. [MODES#actions-no-mode-auto-approves]
- Protected paths (`.git`, `.claude` and others) are never auto-approved except in `bypassPermissions`. [MODES#protected-paths]

### Modes

| Mode | Runs without asking | Source |
| :- | :- | :- |
| `default` (Manual) | reads only | [MODES#available-modes] |
| `acceptEdits` | reads, file edits, `mkdir/touch/rm/rmdir/mv/cp/sed` inside working dirs | [MODES#auto-approve-file-edits-with-acceptedits-mode] |
| `plan` | reads; edits blocked until the user approves a plan (the approve options switch the mode) | [MODES#analyze-before-you-edit-with-plan-mode] |
| `auto` | everything, each unresolved action checked by a classifier; drops broad allow rules (`Bash(*)`, `Bash(python*)`, `Agent`) while active | [MODES#how-auto-mode-evaluates-actions] |
| `dontAsk` | reads and pre-approved tools; anything that would prompt is denied | [MODES#allow-only-pre-approved-tools-with-dontask-mode] |
| `bypassPermissions` | everything except the "no mode auto-approves" list; deny rules still apply | [MODES#skip-all-checks-with-bypasspermissions-mode] |

Auto-mode details relevant to a chat runtime ([MODES](https://code.claude.com/docs/en/permission-modes)):

- A boundary the user states ("don't push") is a block signal until the user lifts it. "Claude's own judgment that a condition was met does not lift it." Compaction can lose it, and "for a hard guarantee, add a deny rule". [MODES#boundaries-you-state-in-conversation]
- An approval the user states must name the action and its specifics ("you can force-push" clears nothing). It covers one action unless the user grants it as standing. [MODES#approvals-you-state-in-conversation]
- Fallback: 3 blocks in a row or 20 in total pause auto mode and resume prompting. [MODES#when-auto-mode-falls-back]
- The classifier input omits tool results, so hostile content cannot steer the classifier directly. A separate probe scans incoming tool results. [MODES#how-auto-mode-evaluates-actions]

### Settings precedence

The order is managed, then `--settings`, then local, then project, then user. [SET#settings-precedence]

- Arrays merge across files. [SET#lists-merge-instead-of-overriding]
- A deny at any scope beats an allow at any scope, and nothing overrides a managed deny. [PERM#settings-precedence]
- `allowManagedPermissionRulesOnly` makes managed settings the only source of rules, and `disableBypassPermissionsMode` / `disableAutoMode` lock modes. [PERM#managed-settings]
- Project `allow` rules and `additionalDirectories` wait for workspace trust, while `deny` and `ask` always apply. A tracked or symlinked `settings.local.json` is treated as repository-supplied. [PERM#project-allow-rules-and-workspace-trust]

### How an "ask" is presented and what each answer grants

- Prompt options (Bash example): Yes; "Yes, and don't ask again for: `npm test *`"; "Yes, and switch to auto mode"; No. Tab adds a comment. [PERM#permission-system]
- What a grant covers varies by tool type:
  - Bash and WebFetch: saved permanently per repository and command or domain, to `.claude/settings.local.json` at the git root.
  - File edits: until the session ends.
  - WebSearch: permanently per repository.
- A compound command saves one rule per subcommand that needed approval, up to five. [PERM#compound-commands]
- If a prompt can't show everything a broader option would allow, it offers only the one-time approval. [PERM#permission-system]
- A comment on Yes is sent to Claude after the result, and a comment on No is sent as the denial reason. A bare **No** in the main conversation stops the turn. [PERM#add-a-comment-when-you-answer-a-permission-prompt]
- `/permissions` lists every rule with its source file. Edits apply from the next tool call. [PERM#manage-permissions]
- Grant shapes are machine-readable:
  - `PermissionUpdate{type: addRules|replaceRules|removeRules|setMode|addDirectories|removeDirectories, rules[{toolName, ruleContent}], behavior, destination: session|localSettings|projectSettings|userSettings}`. [HOOKS#permission-update-entries], [SDKTYPES]
  - `PermissionResultDeny` has `interrupt: bool`, which stops the agent. [SDKTYPES]
  - `ToolPermissionContext` carries `suggestions`, `decision_reason`, `title`, `display_name`, `description` and `blocked_path` for the UI. [SDKTYPES]

### How hooks and the system prompt tell the model to stop and ask

- `PreToolUse` returns `permissionDecision` with one of:
  - `allow`: skip the prompt.
  - `deny`: the reason is shown to Claude.
  - `ask`: the reason is shown to the user, and the prompt names the hook's source.
  - `defer`: `-p` only. The run pauses with `stop_reason: tool_deferred` and resumes later.

  When hooks disagree, the precedence is deny > defer > ask > allow. A hook's `ask` forces a prompt even in auto mode. [HOOKS#pretooluse-decision-control], [HOOKS#defer-a-tool-call-for-later]
- `PermissionRequest` runs only when a prompt is about to show. It can allow (with `updatedPermissions`) or deny (with `message` and `interrupt`). Where no prompt can be shown and no hook decides, the call is denied. [HOOKS#permissionrequest]
- `Stop` / `/goal`: a hook can block the end of a turn so the model keeps working toward a condition, with a cap of 8 consecutive continuations. [HOOKS#stop]
- The model asks clarifying questions with the `AskUserQuestion` tool, which is multiple choice with free text. No allow rule or mode auto-approves it. [TOOLS#askuserquestion-tool-behavior], [MODES#actions-no-mode-auto-approves]
- Auto mode "nudges Claude to keep working without stopping for clarifying questions", though it still asks when the prompt or a skill relies on asking. [MODES#eliminate-prompts-with-auto-mode]
- The system prompt: the `claude_code` preset includes "tool usage instructions and security and safety instructions". The SDK's default minimal prompt omits them. The docs do not publish the prompt text, so its exact "when to ask" wording is unknown from permitted sources. [SDKSYS]

## Instruction files & memory

### CLAUDE.md levels and load timing

| Scope | Location | Loads | Source |
| :- | :- | :- | :- |
| Managed | `/etc/claude-code/CLAUDE.md` (Linux), or `claudeMd` in managed settings | launch; cannot be excluded | [MEM#deploy-organization-wide-claudemd] |
| User | `~/.claude/CLAUDE.md`, `~/.claude/rules/*.md` | launch | [MEM#choose-where-to-put-claude-md-files], [MEM#user-level-rules] |
| Project | `./CLAUDE.md` or `./.claude/CLAUDE.md`, `.claude/rules/*.md` | launch, for cwd and every ancestor | [MEM#how-claude-md-files-load] |
| Local | `./CLAUDE.local.md` (gitignored) | launch, appended after that directory's CLAUDE.md | [MEM#how-claude-md-files-load] |
| Nested | `sub/CLAUDE.md` below cwd | lazily, when Claude reads or edits a file in `sub/` | [MEM#how-claude-md-files-load] |
| Path rules | `.claude/rules/*.md` with `paths:` frontmatter | only when Claude works with matching files | [MEM#path-specific-rules] |

- **Concatenation order**: files are concatenated, not overridden, from broad to specific (filesystem root down to cwd). Local comes last within each directory, and user rules come before project rules. Conflicts between files are resolved arbitrarily: "Claude may follow either one". [MEM#how-claude-md-files-load], [MEM#user-level-rules]
- **Imports**: `@path` imports are relative to the importing file and nest at most four hops; code spans are skipped. An import in a project file that points outside the working directory needs a one-time approval dialog. User-scope imports are trusted. [MEM#import-additional-files]
- **AGENTS.md**: read only when no CLAUDE.md or CLAUDE.local.md exists in cwd or its ancestors, unless the user sets `claude-md-and-agents-md`. [MEM#agents-md]
- **Exclusions and limits**: `claudeMdExcludes` skips files by glob. A file over 4 MiB is skipped, and files over 200 lines draw a warning. Block HTML comments are stripped. [MEM#exclude-specific-claude-md-files], [MEM#my-claudemd-is-too-large]
- **Compaction**: the project-root CLAUDE.md is re-read from disk after `/compact`, and nested files and path rules reload on demand. [MEM#instructions-seem-lost-after-compact]
- **SDK**: CLAUDE.md loads only when `setting_sources` includes `project` (or `user` for the user file), and it is "injected into the conversation", leaving the system prompt untouched. [SDKSYS], [SDKTYPES]

### Authority relative to the system prompt

- "CLAUDE.md content is delivered as a user message after the system prompt, not as part of the system prompt itself… there's no guarantee of strict compliance." [MEM#claude-isnt-following-my-claudemd]
- Both CLAUDE.md and memory are "context, not enforced configuration". To block an action, use a PreToolUse hook. [MEM#claudemd-vs-auto-memory]
- For system-prompt-level instructions, use `--append-system-prompt`. [MEM#claude-isnt-following-my-claudemd]
- CLAUDE.md does not change what Claude Code allows. [PERM#manage-permissions]
- The auto-mode classifier *does* read CLAUDE.md content as context for its decisions. [MODES#how-auto-mode-evaluates-actions]

### Auto memory vs instructions

| | CLAUDE.md | Auto memory |
| :- | :- | :- |
| Author | user | Claude |
| Content | instructions, rules | learnings: `user`, `feedback`, `project`, `reference` types |
| Scope | org, user, project, local | per git repo (shared across worktrees), machine-local |
| Load | whole file at launch | `MEMORY.md` index, first 200 lines or 25 KB; topic files read on demand |

Sources: [MEM#claudemd-vs-auto-memory], [MEM#auto-memory]

- **Location**: `~/.claude/projects/<project>/memory/`, holding `MEMORY.md` plus one topic file per memory with YAML frontmatter (`type`, plus a `modified` timestamp the harness adds). [MEM#storage-location], [MEM#how-it-works]
- **Size limits**: near a limit, the harness tells Claude to shorten `MEMORY.md`. Over a limit, the write succeeds but returns an error telling Claude to rewrite it. [MEM#how-it-works]
- **What gets saved**: Claude skips anything derivable from the code and anything CLAUDE.md already says. "Remember X" from the user goes to memory, while "add to CLAUDE.md" goes to the instruction file. [MEM#auto-memory], [MEM#view-and-edit-with-memory]
- **Controls**: the toggle is `autoMemoryEnabled`, and `autoMemoryDirectory` overrides the location. A repository-supplied directory is honoured only after workspace trust. [MEM#enable-or-disable-auto-memory], [MEM#storage-location]

## CLI-specific vs maps to a web chat

| Concept | Verdict | Why |
| :- | :- | :- |
| allow / ask / deny lists, deny > ask > allow first-match | Maps | Pure policy over (tool, input). Odysseus has its own tool names, but the `Tool(specifier)` shape transfers. |
| Bash command-prefix matching, compound split, wrapper stripping | Partly | Only if Odysseus exposes a shell tool. The matching is acknowledged as not a security boundary. |
| Read/Edit gitignore path anchors (`//`, `~/`, `/`, relative) | Partly | Maps to workspace file tools, but `~` and "settings-file-relative" anchors assume a local filesystem layout. |
| WebFetch `domain:` and MCP `mcp__server__tool` rules | Maps | Odysseus already has web and MCP tools. |
| Modes default / acceptEdits / plan / dontAsk / bypass | Maps | They are presets over the same rule engine. Plan mode needs a plan-approval card. |
| Auto mode classifier | Maps (costly) | It is provider-agnostic in principle (a second model call per unresolved action), but it is an extra model dependency and cost. |
| Ask prompt: once / don't-ask-again rule / deny with comment | Maps | This is a chat card. "Session" and "persist rule" destinations become per-chat and per-user/workspace stores. |
| `settings.local.json` vs project vs user files, git-root keying | CLI-specific | It relies on the filesystem and git. A web app would store rules in a DB per user and per workspace. |
| Managed settings, MDM, `allowManagedPermissionRulesOnly` | Partly | It maps to an admin or server policy tier in a multi-user deployment. |
| Workspace trust dialog gating project `allow` rules | Maps | It is the same idea as "repo-supplied grants need user consent", and it applies to workspace instruction files. |
| PreToolUse / PermissionRequest / Stop hooks as shell commands | CLI-specific | Hooks are out of scope on map #20. The *decision shape* (allow/deny/ask + reason + updatedInput) is reusable internally. |
| `defer` + resume | Maps conceptually | A web chat already "defers": it pauses at the card and resumes on the answer. |
| `AskUserQuestion` tool | Maps | A multiple-choice chat card. |
| Sandbox, protected paths `.git` / `.claude`, critical-path `rm` | Partly | Odysseus has its own sandbox. The protected-path idea maps to "the agent may not edit its own config or rules without asking". |
| CLAUDE.md user / project / local / nested lazy load | Partly | User and workspace levels map. "Local" (gitignored per-checkout) and directory-walk lazy loading assume a cwd. |
| `@path` imports, external-import approval | Partly | Maps only for workspace files. Approval maps to the trust card. |
| Auto memory dir + `MEMORY.md` index | Maps | Odysseus already has a memory store. The index-plus-topic-files layout and the size cap are the transferable parts. |
| `/permissions`, `/memory`, `/context` slash commands, `Shift+Tab` mode cycle | CLI-specific | Their UI equivalents are a settings page, a mode selector and a "loaded context" inspector. |

## Implications for Odysseus (facts, not decisions)

- Claude Code grants come from **rules plus mode plus an explicit user answer**, never from the wording of the request. The one exception is the auto-mode classifier, which reads stated boundaries and approvals, does not persist them, and requires an approval to name the action and its specifics. Odysseus today derives grants from request wording: `create_request_authority` in `src/agent_runtime/authority.py` calls `interpret_request`, and `requested_capabilities` lives in `src/turn_contract.py`.
- In Claude Code an "ask" is triggered by **rule or mode**: no allow match in `default`, an explicit ask rule, a protected path, or a tool that needs interaction. In Odysseus an approval card is raised only by the taint gate: `ToolRunSecurityContext.decision_for` in `src/tool_capabilities.py` returns allow whenever `external_untrusted_context_seen` is false.
- The answer scopes differ:
  - Claude Code offers once, session, persisted rule (local/project/user) and deny with an optional `interrupt`, plus a mode switch.
  - Odysseus's card in `src/tool_approvals.py` offers "Allow for this task", "Allow for this chat session" and "Deny". It has no once-only option and no persisted-rule option.
- Claude Code loads user-level and project-level instruction files as **user-message context** after the system prompt. Their authority is advisory, below the system prompt and with no enforcement power. Odysseus loads only workspace `AGENTS.md`, walking up to the root (max 8 files, 12000 chars each), wrapped as **untrusted** context in `_workspace_agents_context_message` in `src/agent_loop.py`. It has no user-level file.
- Claude Code separates concerns:
  - Instruction files steer.
  - Permission rules enforce.
  - Repo-supplied *grants* (allow rules, external imports, hooks) need workspace trust.
  - Repo-supplied *restrictions* (deny, ask) never need trust.
- The SDK exposes the whole permission model as plain data types (`PermissionUpdate`, `PermissionResultAllow/Deny`, `ToolPermissionContext`) under an MIT-licensed repo. These are a legitimate shape reference for a Python re-implementation. [SDKTYPES]
- The permitted sources do not reveal the system-prompt text that tells Claude when to stop and ask. Only its existence ("security and safety instructions") and the `AskUserQuestion` tool are documented.
