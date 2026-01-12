# RDS Storage Permissions and Usage

## Your RDS Storage

From your quota output, you have access to:

1. **`rds-ePtR33Nsgi4`** (Project 90416)
   - Used: 12410.7 GB
   - Quota: 23000.0 GB
   - **Available: ~10,589 GB**
   - Path: `/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4` or `~/rds/rds-autism-research-ePtR33Nsgi4`

2. **`/rds-d7`** (Project 45718)
   - Used: 0.0 GB
   - Quota: 1099.5 GB
   - **Available: ~1100 GB**
   - Path: `/rds-d7/project/45718`

## Are You Allowed?

**Yes!** You're allowed to:
- ✅ Create directories in your project's RDS storage
- ✅ Store research data (EU emotions, CAM data, models, etc.)
- ✅ Use it for large datasets that don't fit in `/home`

**What you created:**
```
~/rds/rds-autism-research-ePtR33Nsgi4/data
```

This is **perfectly fine** and is the right place for your data!

## RDS Structure

Your RDS storage is shared with your research group (project 90416), so:
- ✅ You can create directories and store files
- ✅ Other project members can access it (if needed)
- ✅ It's backed up and persistent
- ⚠️  Be mindful of shared space (you have ~10TB available, so plenty of room)

## Recommended Structure

```
~/rds/rds-autism-research-ePtR33Nsgi4/
├── data/
│   ├── EU_emotions/          # EU emotions library
│   ├── CAM/                  # CAM dataset (if needed)
│   └── models/              # Trained models (optional)
├── venv/                    # Python virtual environment
└── results/                 # Experiment results (optional)
```

## Using the Data Folder You Created

Since you already created `data` in `rds-autism-research-ePtR33Nsgi4`, we'll use that:

```bash
# Full path
~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions

# Or absolute path
/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/data/EU_emotions
```

## Transferring to Your Data Folder

The transfer script has been updated to use:
```
/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/data/EU_emotions
```

If the path is slightly different, you can verify on HPC:

```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk
cd ~/rds/rds-autism-research-ePtR33Nsgi4
pwd  # This will show the full path
ls -la data  # Verify your data folder exists
```

## Summary

- ✅ **You're allowed** to use RDS storage
- ✅ **You created the right place** (`data` folder in your RDS project)
- ✅ **Plenty of space** (~10TB available)
- ✅ **Scripts updated** to use your RDS location

Proceed with transferring EU emotions to that location!





