# Mini-XDR v2 Repository Migration Report
## Files to EXCLUDE from GitHub Repository

**Generated on:** September 28, 2025  
**Target Repository:** https://github.com/chasemad/mini-xdr-v2.git

---

## 🚨 CRITICAL SECURITY EXCLUSIONS

### 1. **Environment Files with Credentials**
- `frontend/env.local` - **CONTAINS API KEYS AND IP ADDRESSES**
  - Contains: `NEXT_PUBLIC_API_KEY=demo-minixdr-api-key`
  - Contains: `NEXT_PUBLIC_API_URL=http://54.237.168.3:8000`
- `backend/.env` (if exists) - Would contain database URLs, API keys
- `docs/secrets.md` - Contains credential management documentation

**Risk Level:** 🔴 **CRITICAL** - Immediate security compromise

### 2. **Database Files**
- ~~`backend/xdr.db`~~ - **KEEP THIS** - Contains only demo data (10 test events, demo IPs)
- Other production database files should be excluded

**Risk Level:** ✅ **SAFE** - Demo database with sample data is valuable for repository

### 3. **Log Files**
- `backend/logs/` (7 files) - May contain API keys, IP addresses, user data
  - `backend_final_test.log`, `backend.log`, `mcp.log`, etc.
- `frontend/logs/frontend.log`
- `ngrok.log`

**Risk Level:** 🟠 **HIGH** - May contain sensitive runtime data

---

## 💾 LARGE FILES (>8GB Total)

### 1. **Virtual Environments (5.2GB)**
- `venv/` (2.4GB)
- `backend/.venv/` (2.4GB) 
- `ml-training-env/` (375MB)

**Reason:** Dependencies should be installed via requirements.txt/package.json

### 2. **Node.js Dependencies (651MB)**
- `frontend/node_modules/`
- `backend/node_modules/`

**Reason:** Dependencies should be installed via package-lock.json

### 3. **Datasets Directory (2.4GB)**
- `datasets/cicids2017_official/` - Large CSV and ZIP files
- `datasets/real_datasets/` - Large compressed datasets
- `datasets/working_downloads/` - Large downloaded files (kdd_full.csv, etc.)
- Large individual files (10MB+):
  - `cicids_train.csv`, `cicids_test.csv`
  - Multiple `.pcap_ISCX.csv` files
  - `test_chunk_004.csv`

**Reason:** Too large for Git, should be downloaded separately

### 4. **ML Models (2.5MB)**
- `models/` directory with .pth files
- `aws/feature_scaler.pkl`

**Reason:** Can be regenerated or should use Git LFS

---

## 🔧 GENERATED/BUILD ARTIFACTS

### 1. **Build Caches**
- `frontend/.next/` - Next.js build cache (multiple >10MB files)
- `frontend/tsconfig.tsbuildinfo`
- Python `__pycache__/` directories
- `*.pyc` files

**Reason:** Generated during build process

### 2. **Package Lock Files (Debatable)**
- `package-lock.json` files exist but should be kept for reproducible builds

---

## 📦 BACKUP FILES (20 files)

### SSH Backup Files
- `*.ssh-backup-20250926_213734` (7 files)
- `*.ssh-backup-20250926_213735` (2 files)
- Various deployment script backups

**Reason:** Temporary backup files, not needed in clean repository

---

## 🏗️ DEPRECATED/LEGACY CONTENT

### 1. **Deprecated Directories**
- `aws/_deprecated/` - Old deployment scripts and legacy ML training
- `aws/_legacy/` - Legacy credential cleanup scripts

**Recommendation:** Keep for historical reference but clearly marked as deprecated

### 2. **Multiple AWS Scripts**
Many similar AWS deployment scripts exist. Consider consolidating:
- `start-mini-xdr-aws-v3.sh`
- `start-mini-xdr-aws-v2.sh` (deprecated)
- Multiple deployment scripts with similar functionality

---

## 📁 EMPTY OR MINIMAL DIRECTORIES
- `evidence/` - Empty directory
- `honeypot_configs/` - Empty directory

**Recommendation:** Keep with README explaining their purpose

---

## ✅ RECOMMENDED MIGRATION PROCESS

### Phase 1: Repository Setup (SAFE - No File Deletion!)
1. ✅ **Created comprehensive `.gitignore`** - Prevents sensitive files from being committed
2. 🔲 **Initialize and push clean repository:**
   ```bash
   git init
   git add .gitignore
   git add .
   git commit -m "Initial commit: Clean Mini-XDR v2 repository"
   git remote add origin https://github.com/chasemad/mini-xdr-v2.git
   git push -u origin main
   ```

### Phase 2: Verification  
3. 🔲 **Verify no sensitive files were committed:**
   ```bash
   # Check what files were actually added
   git ls-files | grep -E "(env\.local|\.log|\.db)" | grep -v xdr.db
   
   # Run gitleaks to scan for secrets
   gitleaks detect --no-banner
   ```

### Phase 3: Documentation Update
4. 🔲 **Update README with setup instructions** for new developers to create their own:
   - `frontend/env.local` 
   - Development database setup
   - Local environment configuration

**🎯 ADVANTAGES OF THIS APPROACH:**
- ✅ **Preserves your working environment**
- ✅ **No risk of accidentally breaking development setup** 
- ✅ **Prevents future accidental commits**
- ✅ **Easily reversible if needed**

---

## 📊 SUMMARY STATISTICS

| Category | Count | Total Size | Risk Level |
|----------|-------|------------|------------|
| **Sensitive Files** | 10+ | ~50MB | 🔴 Critical |
| **Large Dependencies** | 4 dirs | ~5.2GB | ⚪ Low Risk |
| **Datasets** | 1 dir | ~2.4GB | ⚪ Low Risk |
| **Build Artifacts** | 100+ | ~700MB | ⚪ Low Risk |
| **Backup Files** | 20 | ~10MB | 🟡 Medium |

**Total Space Savings:** ~8.4GB  
**Security Risk Reduction:** Critical → Minimal

---

## 🛡️ POST-MIGRATION SECURITY CHECKLIST

- [ ] Run `gitleaks detect` to scan for remaining secrets
- [ ] Update documentation to reference new repository
- [ ] Set up proper CI/CD with secret scanning
- [ ] Create separate documentation for local development setup
- [ ] Consider using GitHub Secrets for deployment keys
- [ ] Set up Git LFS for any necessary large files
- [ ] Add security scanning to GitHub Actions

---

**✅ SAFE MIGRATION:** The comprehensive `.gitignore` will automatically exclude all sensitive files and large dependencies. Your local development environment stays intact - no files need to be deleted!
