 # MAS512

Repository documentation and setup instructions.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `.git/` | Git metadata; do not edit manually. |
| `.gitattributes` | Git LFS and repository attribute rules. |
| `.gitignore` | Files and directories excluded from Git. |
| `README.md` | Project documentation and setup instructions. |


The repository may not contain every optional directory above. Add new top-level directories to this table when they are introduced.

## Clone or import the repository

Install [Git](https://git-scm.com/) and [Git LFS](https://git-lfs.com/), then run:

```bash
git lfs install
git clone <repository-url> mas512
cd mas512
git lfs pull
```

If the repository has already been cloned, fetch LFS files with:

```bash
git lfs install
git lfs pull
```

## Create and activate a virtual environment

Python 3.10 or newer is recommended. From the repository root:

### Windows PowerShell

```powershell
py -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Windows Command Prompt

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies if a dependency file is present:

```bash
python -m pip install -r requirements.txt
```

For a package project, install it in editable mode instead:

```bash
python -m pip install -e .
```

The virtual environment is local to the checkout and should not be committed. Deactivate it with:

```bash
deactivate
```


