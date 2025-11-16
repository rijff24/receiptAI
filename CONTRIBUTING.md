# Contributing to ScannerAI

Thank you for taking the time to contribute to ScannerAI! This project is a community-maintained fork of the [receipt_scanner](https://github.com/datasciencecampus/receipt_scanner) project originally developed by the Data Science Campus at the Office for National Statistics. Together we can keep the tool robust, privacy-preserving, and easy to use.

## Ways to Help

- Report issues or enhancement ideas via GitHub Issues.
- Improve documentation (README, SETTINGS guide, tutorials).
- Extend the OCR / AI provider support.
- Improve classifier accuracy or add new features.

## Ground Rules

1. **Respect licensing**: All code remains under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). By contributing you agree your work is released under the same licence.
2. **Security first**: Never commit API keys, secrets, or private user data. Use the new in-app settings and local encrypted storage instead of hardcoding credentials.
3. **Discuss big changes early**: For substantial refactors or external dependencies, open an issue or discussion before investing significant time.
4. **Follow code style**: We target Python 3.11+, use `ruff` for linting and `black`-compatible formatting (PEP 8, 79 char soft limit). Include docstrings for new modules and public functions.

## Development Workflow

1. **Set up environment**
   ```bash
   git clone https://github.com/rijff24/receiptAI.git
   cd receipt_scanner
   python -m venv scanner-venv
   scanner-venv\Scripts\activate  # or source scanner-venv/bin/activate
   pip install -e ".[dev]"
   ```
2. **Run tests & linters**
   ```bash
   ruff check .
   pytest
   ```
3. **Add / Update documentation** for any user-facing change.
4. **Submit a PR**
   - Reference related issues
   - Describe testing performed
   - Ensure CI passes

## Reporting Security Issues

Do **not** open a public issue for security vulnerabilities. Instead, email `rijff@swan-computing.com` with as much detail as possible. See `SECURITY.md` for the full policy.


## Documentation Expectations

- Update the relevant `.md` files whenever you change behavior or configuration.
- Architecture, deployment, and troubleshooting guides live in the repository root; keep them current when altering infrastructure.
- Screenshots or GIFs are encouraged for UI/UX improvements.

We appreciate your help keeping ScannerAI reliable and safe!

We appreciate your help keeping ScannerAI reliable and safe!

