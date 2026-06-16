# Kappa

Static viewer and reproducibility scripts for the RR Lyrae RSP/kappa-mechanism batch analysis.

The deployable app is generated into `site/` and published at:

https://earlbellinger.github.io/apps/kappa/

## Build the static app

From this repository:

```powershell
python scripts/build_static_site.py --rre-root C:\Users\earlb\Downloads\rre --output site
```

The builder copies only portable analysis products: completed GIF/PNG summaries, light-curve CSVs, verification JSON, batch metadata, and cycle diagnostics. It does not copy MESA work directories, SDKs, profile dumps, or transient logs.

## Watch And Deploy

The long batch run can update the public app automatically when meaningful outputs change:

```powershell
python scripts/watch_and_deploy.py
```

The watcher ignores ordinary period-by-period progress and deploys only when high-level state changes, such as a new GIF, a verification change, a stage transition, or a final audit/quality-extension change.

## Source Layout

- `scripts/build_static_site.py`: creates the portable static app bundle.
- `scripts/watch_and_deploy.py`: refreshes and deploys the app when batch outputs materially change.
- `analysis/`: selected scripts used to prepare and verify the batch analysis.
- `site/`: generated app bundle suitable for GitHub Pages.
