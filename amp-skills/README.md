# Amp Skills

A collection of specialized skills for [Amp](https://ampcode.com), the AI coding agent.

## What are Skills?

Skills are domain-specific instruction sets that extend Amp's capabilities. When you invoke a skill, it loads detailed workflows, patterns, and techniques into the conversation context.

## Available Skills

- **ralph** - Autonomous feature development (setup and execution)
- **agent-browser** - Browser automation for testing, form filling, screenshots
- **react-best-practices** - React/Next.js performance optimization (40+ rules from Vercel) *
- **web-design-guidelines** - UI/UX audit against 100+ best practices *

\* Symlinked from [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)

## Usage

Skills are automatically available in Amp. To use one:

```
Use the direct-response-copy skill to write a landing page for my SaaS product.
```

Or trigger naturally:
```
Help me write copy for my landing page.
```

Amp will load the appropriate skill based on your request.

## Adding Skills

Each skill is a folder containing a `SKILL.md` file with frontmatter:

```markdown
---
name: my-skill
description: "Brief description. Use when... Triggers on: keyword1, keyword2."
---

# Skill Title

Instructions, workflows, and patterns go here.
```

## License

MIT
