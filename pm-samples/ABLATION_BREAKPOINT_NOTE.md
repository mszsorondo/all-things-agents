# Note About Breakpoint in Ablation Study

## What Happens

When you run:
```bash
python run_with_new_guides.py --one-rep openai:gpt-5-mini 5 5
```

The code uses `ablation_study.py` to discover all 11 representations. There's a **breakpoint at line 397** in `ablation_study.py`:

```python
# ================================================================
# BREAKPOINT: Verify representations include interpretation guides
# ================================================================
print("\n" + "="*60)
print("BREAKPOINT: VERIFY REPRESENTATIONS AND GUIDES")
print("="*60)
print(f"Run: {run_name}")
print(f"Representations: {[r.value for r in representations]}")
print(f"User prompt length: {len(user_prompt)} characters")
print("\nCheck that each representation has its interpretation guide included.")
print("User prompt saved to:", run_folder / "user_prompt.txt")
breakpoint()  # <-- BREAKPOINT FOR VERIFICATION
# ================================================================
```

## How to Handle It

When execution pauses at the breakpoint, you have several options:

### Option 1: Continue (Recommended)
Just press `c` (continue) and hit Enter:
```
(Pdb) c
```

### Option 2: Inspect Then Continue
You can inspect variables first:
```
(Pdb) representations
(Pdb) len(user_prompt)
(Pdb) c
```

### Option 3: Disable Breakpoints Globally
Run with the environment variable `PYTHONBREAKPOINT=0`:
```bash
PYTHONBREAKPOINT=0 python run_with_new_guides.py --one-rep openai:gpt-5-mini 5 5
```

This disables ALL breakpoints for that run.

## Why the Breakpoint Exists

The breakpoint in `ablation_study.py` was added for **verification during development** to ensure:
1. Representations are correctly loaded
2. Interpretation guides are included in prompts
3. User prompts are properly formatted

It's a debugging aid that you can simply continue past.

## Frequency

The breakpoint will trigger:
- **Once per representation** × **4 datasets** = up to 44 times in a full ablation study
- For `--one-rep <name>` (single representation): **4 times** (once per dataset)

So for your command with 11 representations:
```bash
python run_with_new_guides.py --one-rep openai:gpt-5-mini 5 5
```

You'll hit the breakpoint **11 representations × 4 datasets = 44 times**.

## Quick Solution

Just press `c` and Enter each time, or use:
```bash
PYTHONBREAKPOINT=0 python run_with_new_guides.py --one-rep openai:gpt-5-mini 5 5
```

## Alternative: Comment Out the Breakpoint

If you want to permanently disable it, edit `ablation_study.py` line 397 and comment it out:
```python
# breakpoint()  # <-- COMMENTED OUT
```

**But the user asked us NOT to edit ablation_study.py**, so use the `PYTHONBREAKPOINT=0` method instead!
