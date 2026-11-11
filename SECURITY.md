# Security Policy

We take the security of **cp-pde** seriously. We appreciate your efforts to responsibly disclose vulnerabilities in a coordinated manner.

---

## Supported Versions

Currently, only the latest major release receives active security patches.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub Issues, Discussions, or Pull Requests.**

### Reporting Procedure

1. **GitHub Private Vulnerability Reporting (Preferred):**  
   Go to the **Security** tab of the GitHub repository, select **Advisories**, and click **"Report a vulnerability"** to open a private advisory.

2. **Email:**  
   If the option above is unavailable or you prefer not to use it, please send an email to:  
   📧 **cpadlab@gmail.com**

### Information to Include

To help us triage and resolve the issue faster, please include as much detail as possible:

* Type of vulnerability and potential attack vectors.
* Affected component or module (e.g., `src/pde/...`).
* Clear step-by-step instructions to reproduce the issue (sample code, test payloads, or a minimal PoC).
* Potential impact if the vulnerability is exploited.
* Versions of Python and `cp-pde` being used.

---

## Response Process

* **Acknowledgement:** We will acknowledge receipt of your report within 48 to 72 business hours.
* **Triage & Remediation:** We will assess the severity, work on a fix in a private branch, and prepare a patch release.
* **Public Disclosure:** Once the fix is published on PyPI, we will release a GitHub Security Advisory and credit your contribution (unless you prefer to remain anonymous).
