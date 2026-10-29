"""Channel 4 codebook labels.

Demographic labels are from the variable-key letter. Item response numbers
are not in that letter; the 1-4 map below stays inferred.
"""

from __future__ import annotations

SEX = {
    1: "Male",
    2: "Female",
    3: "Other",
    4: "Prefer not to say",
}
HANDEDNESS = {
    1: "Right",
    2: "Left",
    3: "Ambidextrous",
    4: "Prefer not to say",
}
EDUCATION = {
    1: "No high school",
    2: "High school",
    3: "Undergraduate",
    4: "Postgraduate",
    5: "Prefer not to say",
}
REGION = {
    1: "Wales",
    2: "Scotland",
    3: "Northern Ireland",
    4: "London (England)",
    5: "North East (England)",
    6: "North West (England)",
    7: "Yorkshire and Humber (England)",
    8: "West Midlands (England)",
    9: "East Midlands (England)",
    10: "South East (England)",
    11: "South West (England)",
    12: "Outside UK",
    13: "Other UK",
    14: "Prefer not to say",
}
REGION_UK_CODES = tuple(range(1, 12))
DIAGNOSIS = {
    1: "ADHD",
    2: "Autism Spectrum Disorder",
    3: "Bipolar",
    4: "Depression",
    5: "Learning disability",
    6: "OCD",
    7: "Schizophrenia",
    8: "Prefer not to say",
    9: "None of these",
}
ASD_SUBTYPE = {
    1: "Classical autism",
    2: "Asperger syndrome",
    3: "Other",
}
OCCUPATION = {
    1: "Artist",
    2: "Civil Engineering",
    3: "Computers & I.T.",
    4: "Director",
    5: "Engineering",
    6: "Entrepreneur",
    7: "Financial Banking",
    8: "Food & Drinks",
    9: "Healthcare",
    10: "Hospitality",
    11: "Legal",
    12: "Leisure",
    13: "Musician",
    14: "Office Administration",
    15: "Other",
    16: "Public Sector",
    17: "Services",
    18: "Publishing & Media",
    19: "Retail",
    20: "Sales",
    21: "Scientific & Technical",
    22: "Supply chain",
    23: "Teaching & Interpretation",
    24: "Transport",
    25: "Other",
    26: "Prefer not to say",
}
OCCUPATION_CONFIRMED = OCCUPATION

# Inferred, not in the codebook. Confirm with the ARC.
# Only this direction reproduced autistic > non-autistic AQ-10 and SQ-R-10
# and autistic < non-autistic EQ-10 in the earlier pass.
ITEM_CODING_INFERRED = {
    1: "strongly agree",
    2: "slightly agree",
    3: "slightly disagree",
    4: "strongly disagree",
}
ITEM_CODING_STATUS = "inferred"
AGREE_CODES = (1, 2)
DISAGREE_CODES = (3, 4)
