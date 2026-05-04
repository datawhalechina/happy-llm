# GitHub 同步工作流

这个仓库的远端已经是：

```bash
origin https://github.com/Edward-Jing/happy-llm.git
```

建议把手搓 LLM 工程放在独立分支：

```bash
git switch scratch-llm-starter
git status --short --branch
git add scratch_llm scripts tests pyproject.toml .gitignore
git commit -m "Add scratch LLM learning scaffold"
git push -u origin scratch-llm-starter
```

日常节奏：

```bash
git pull --rebase
python3 -m unittest tests/test_contracts.py
git add scratch_llm tests
git commit -m "Implement RMSNorm"
git push
```

如果你想把学习工程合回 `main`，可以在 GitHub 上从 `scratch-llm-starter` 开 Pull Request。
