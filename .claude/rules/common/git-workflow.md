# Git Workflow

## Commit Message Format

```
<type>: <description>

<optional body>
```

Types: feat, fix, refactor, docs, test, chore, perf, ci

Note: Attribution disabled globally via ~/.claude/settings.json.

## Pull Request Workflow

When creating PRs:
1. Analyze full commit history (not just latest commit)
2. Use `git diff [base-branch]...HEAD` to see all changes
3. Draft comprehensive PR summary
4. Include test plan with TODOs
5. Push with `-u` flag if new branch

## Branching & PR Best Practices

Default branch (workflow): `develop`
Note: This workflow assumes `develop` as the integration branch even if the repo default branch is `main`.

Branch naming:
- Use one of: `feature/`, `chore/`, `bugfix/`
- Include the issue number in the branch name
- Format: `<type>/<issue>-<short-slug>`
- If there is no issue, use `no-issue` as the issue placeholder

Examples:
- `feature/1-add-github-rule`
- `bugfix/42-fix-login-redirect`
- `chore/7-update-deps`
- `chore/no-issue-fix-typo`

Standard flow (feature work):
1. Create branch from `develop`
2. Make changes
3. Stage changes (`git add ...`)
4. Commit with a conventional message
5. Push branch
6. Open PR targeting `develop`
7. Use `git log` and `git diff` to draft the PR summary

Release flow:
- Open PRs from `develop` to `main` when promoting changes for release

## Feature Implementation Workflow

1. **Plan First**
   - Use **planner** agent to create implementation plan
   - Identify dependencies and risks
   - Break down into phases

2. **TDD Approach**
   - Use **tdd-guide** agent
   - Write tests first (RED)
   - Implement to pass tests (GREEN)
   - Refactor (IMPROVE)
   - Verify 80%+ coverage

3. **Code Review**
   - Use **code-reviewer** agent immediately after writing code
   - Address CRITICAL and HIGH issues
   - Fix MEDIUM issues when possible

4. **Commit & Push**
   - Detailed commit messages
   - Follow conventional commits format
