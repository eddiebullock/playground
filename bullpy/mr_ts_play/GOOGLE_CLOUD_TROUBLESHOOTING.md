# Google Cloud Troubleshooting Guide

## Issue: Safety Filters Blocking Responses

If you're getting "Response blocked by safety filter" errors, check the following:

## 1. Check API Key Permissions

**In Google Cloud Console:**
1. Go to: https://console.cloud.google.com/
2. Navigate to: **APIs & Services** → **Credentials**
3. Find your API key
4. Click on it to edit
5. Check **API restrictions**:
   - Should include: **Generative Language API**
   - Or set to: **Don't restrict key** (for testing)

## 2. Check API Enablement

**In Google Cloud Console:**
1. Go to: **APIs & Services** → **Library**
2. Search for: **Generative Language API**
3. Make sure it's **ENABLED**
4. If not, click **Enable**

## 3. Check Quotas and Billing

**In Google Cloud Console:**
1. Go to: **APIs & Services** → **Quotas**
2. Search for: **Generative Language API**
3. Check if you have quota available
4. Go to: **Billing** → Make sure billing is enabled

## 4. Check Model Availability

**In Google AI Studio:**
1. Go to: https://aistudio.google.com/
2. Check which models are available
3. Try using the model directly in the web interface
4. If it works there but not via API, it's an API key/configuration issue

## 5. Test API Key Directly

Run this test script:
```bash
python test_google_models.py
```

This will show:
- Which models your API key can access
- Which models support `generateContent`
- Any authentication errors

## 6. Safety Filter Settings

The deprecated `google.generativeai` package may not properly support disabling safety filters. 

**Options:**
1. **Handle blocked responses gracefully** (current approach - returns uniform scores)
2. **Upgrade to `google.genai` package** (newer, better support)
3. **Use a different model** that's less strict (e.g., `gemini-2.0-flash`)

## 7. Check Logs

**In Google Cloud Console:**
1. Go to: **Logging** → **Logs Explorer**
2. Filter by: `resource.type="api"`
3. Look for errors related to Generative Language API
4. Check for quota/billing errors

## 8. Configure Safety Settings in Google AI Studio

**Important:** Safety filters can be configured in Google AI Studio:

1. Go to: https://aistudio.google.com/
2. Click on your API key or project settings
3. Look for **Safety Settings** or **Content Filtering**
4. You may be able to adjust safety thresholds there
5. Some accounts have stricter default settings

**Note:** The deprecated `google.generativeai` package may not properly respect safety settings passed in code. If safety settings in code don't work, you may need to:
- Configure them in Google AI Studio/Console
- Or upgrade to the new `google.genai` package

## 9. Test Safety Settings

Run the test script to see which format works:
```bash
python test_google_safety.py
```

This will test different safety setting formats and show which one works with your API key.

## Current Code Fixes

I've updated the code to:
1. Try multiple safety setting formats (enum, objects, etc.)
2. Log which format is being used
3. Handle blocked responses gracefully (but this shouldn't be needed if safety settings work)

## Next Steps

1. **Run the test script**: `python test_google_safety.py` to see what works
2. **Check Google AI Studio**: Configure safety settings there if needed
3. **Check API key permissions**: Make sure it has full access
4. **Try the updated code**: The new code tries multiple formats automatically
