# Birdophile Monorepo

This repository contains three projects:

## Projects

- `legacy/` - Original Python-based bird detection and camera system
- `birdophile/` - Improved Python bird detection and camera service
- `web/` - Next.js web application for viewing and interacting with bird data

## Setup and Development

Each project has its own setup instructions in their respective directories.



### Birdophile Project

Installation
```bash
cd birdophile
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Running 
```
birdophile_run.sh
```

OR
```

python3 /home/timotej/dev/birdophile/camera_service_buffer.py
```

### Web Project

Installing
```bash
cd web
yarn install
yarn dev
```

Running 
```
pnpm dev
```


## Repository Structure

The repository uses project-specific `.gitignore` files, along with a root-level `.gitignore` for common exclusions. 