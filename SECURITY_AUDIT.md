# 🔒 Security Audit Report — news-trading-bot

**Date:** 2026-04-04  
**Tool:** pip-audit 2.10.0 (PyPI advisory database)  
**Python:** 3.12 | **Venv:** `.venv/`

---

## 1. pip-audit Scan Results

### ✅ Vulnerabilities Found & Fixed

| Package    | Version (before) | CVE               | Severity | Fix Version | Status     |
|------------|-------------------|-------------------|----------|-------------|------------|
| pip        | 24.0              | CVE-2025-8869     | HIGH     | ≥25.3       | ✅ Fixed (→26.0.1) |
| pip        | 24.0              | CVE-2026-1703     | HIGH     | ≥26.0       | ✅ Fixed (→26.0.1) |
| setuptools | 70.2.0            | CVE-2025-47273    | HIGH     | ≥78.1.1     | ✅ Fixed (→81.0.0) |

> **Note:** setuptools pinned to `<82` due to `torch 2.11.0+cpu` requiring `setuptools<82`.

### ✅ All Application Dependencies — No Known Vulnerabilities

After fixes, `pip-audit` reports **0 vulnerabilities** across all 80+ packages.

---

## 2. Key Dependency Version Check

| Package      | Installed    | Latest   | Known CVEs | Status |
|--------------|-------------|----------|------------|--------|
| torch        | 2.11.0+cpu  | 2.11.0   | None       | ✅ Current |
| transformers | 5.5.0       | 5.5.0    | None       | ✅ Current |
| alpaca-py    | 0.43.2      | 0.43.2   | None       | ✅ Current |
| aiohttp      | 3.13.5      | 3.13.5   | None       | ✅ Current |
| sqlalchemy   | 2.0.49      | 2.0.49   | None       | ✅ Current |
| pydantic     | 2.12.5      | 2.12.5   | None       | ✅ Current |
| numpy        | 2.4.4       | 2.4.4    | None       | ✅ Current |
| pandas       | 3.0.2       | 3.0.2    | None       | ✅ Current |
| jinja2       | 3.1.6       | 3.1.6    | None       | ✅ Current |
| urllib3      | 2.6.3       | 2.6.3    | None       | ✅ Current |
| certifi      | 2026.2.25   | 2026.2.25| None       | ✅ Current |
| requests     | 2.33.1      | 2.33.1   | None       | ✅ Current |

### Skipped by pip-audit (manual verification)
- **torch 2.11.0+cpu** — Skipped because `+cpu` suffix doesn't match PyPI. Manually verified against `torch==2.11.0`: **no known CVEs**.
- **news-trading-bot** — Local editable package, not on PyPI (expected).

---

## 3. pyproject.toml Version Pins Analysis

| Dependency         | Pin in pyproject.toml | Installed  | Assessment |
|--------------------|----------------------|------------|------------|
| alpaca-py          | `>=0.28.0`           | 0.43.2     | ✅ Good — floor pin allows updates |
| finnhub-python     | `>=2.4.0`            | 2.4.27     | ✅ Good |
| feedparser         | `>=6.0.0`            | 6.0.12     | ✅ Good |
| aiohttp            | `>=3.9.0`            | 3.13.5     | ✅ Good |
| transformers       | `>=4.40.0`           | 5.5.0      | ✅ Good |
| torch              | `>=2.2.0`            | 2.11.0+cpu | ✅ Good |
| pandas             | `>=2.2.0`            | 3.0.2      | ✅ Good |
| numpy              | `>=1.26.0`           | 2.4.4      | ✅ Good |
| sqlalchemy         | `>=2.0.0`            | 2.0.49     | ✅ Good |
| aiosqlite          | `>=0.20.0`           | 0.22.1     | ✅ Good |
| pydantic           | `>=2.6.0`            | 2.12.5     | ✅ Good |
| pydantic-settings  | `>=2.2.0`            | 2.13.1     | ✅ Good |
| pyyaml             | `>=6.0.0`            | 6.0.3      | ✅ Good |
| discord-webhook    | `>=1.3.0`            | 1.4.1      | ✅ Good |
| structlog          | `>=24.1.0`           | 25.5.0     | ✅ Good |
| cachetools         | `>=5.3.0`            | 7.0.5      | ✅ Good |
| python-dotenv      | `>=1.0.0`            | 1.2.2      | ✅ Good |

### Pin Strategy Assessment

**Approach used:** `>=minimum` floor pins (no upper bounds)  
**Verdict:** ✅ **Good practice for an application** (not a library)

- Floor pins prevent installing ancient vulnerable versions
- No upper caps means `pip install --upgrade` gets latest patches
- Minimum versions are reasonably recent (not too permissive)

### ⚠️ Minor Recommendations

1. **Consider a lockfile** — Use `pip freeze > requirements.lock` or `pip-compile` to pin exact versions for reproducible deployments while keeping `pyproject.toml` flexible.
2. **setuptools constraint** — torch requires `setuptools<82`. If torch drops this constraint in a future release, setuptools can be upgraded further. Currently at 81.0.0 (patched).

---

## 4. Transitive Dependency Highlights

| Transitive Dep   | Version     | Notes |
|-------------------|-------------|-------|
| jinja2            | 3.1.6       | ✅ Used by torch — no CVEs |
| requests          | 2.33.1      | ✅ Used by alpaca-py — no CVEs |
| urllib3           | 2.6.3       | ✅ No CVEs |
| certifi           | 2026.2.25   | ✅ Current CA bundle |
| cryptography      | 46.0.6      | ✅ No CVEs (installed via safety) |
| websockets        | 16.0        | ✅ Used by alpaca-py — no CVEs |

---

## 5. Summary

| Category                 | Status |
|--------------------------|--------|
| Known CVEs (app deps)    | ✅ **0 vulnerabilities** |
| Known CVEs (tooling)     | ✅ **Fixed** (pip→26.0.1, setuptools→81.0.0) |
| Key packages up-to-date  | ✅ All at latest versions |
| Version pin strategy     | ✅ Reasonable `>=minimum` floor pins |
| torch +cpu audit         | ✅ Manually verified — no CVEs |

**Overall Risk: LOW** — All dependencies are current and free of known vulnerabilities.

---

## Actions Taken

1. Upgraded `pip` from 24.0 → 26.0.1 (fixed CVE-2025-8869, CVE-2026-1703)
2. Upgraded `setuptools` from 70.2.0 → 81.0.0 (fixed CVE-2025-47273, respects torch's `<82` constraint)
3. Verified all 80+ packages via pip-audit — clean
4. Manually audited torch 2.11.0+cpu (skipped by pip-audit) — clean
