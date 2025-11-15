# Security Policy

## Supported Versions

We aim to keep the latest `main` branch patched and secure. Older tags or forks receive fixes on a best-effort basis only. Always upgrade to the newest release for the latest security updates.

## Reporting a Vulnerability

Please report security issues privately to `security@scannerai.app` (placeholder contact). Include:

- Description of the issue and potential impact
- Steps to reproduce or proof-of-concept
- Logs, screenshots, or traceback details (no sensitive data)

We will acknowledge receipt within 3 working days and keep you informed of progress. Responsible disclosure is appreciated; we will credit researchers unless anonymity is requested.

## Handling Sensitive Data

- Never commit API keys, access tokens, or personal receipt data.
- Use the in-app settings to store encrypted credentials locally.
- When filing issues, redact file paths or contents that might expose private information.

## Hardening Checklist for Contributors

- Validate all user inputs, especially file paths and API keys.
- Prefer least-privilege permissions when reading/writing local files.
- Keep dependencies patched (`pip install -r requirements.txt` regularly).
- Run `ruff`/`pytest` before submitting pull requests.

Thank you for helping keep ScannerAI secure for everyone.

