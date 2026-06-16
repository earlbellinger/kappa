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

## Source Layout

- `scripts/build_static_site.py`: creates the portable static app bundle.
- `analysis/`: selected scripts used to prepare and verify the batch analysis.
- `site/`: generated app bundle suitable for GitHub Pages.
