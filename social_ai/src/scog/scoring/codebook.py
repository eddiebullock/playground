"""Channel 4 codebook labels that are stated in the project brief.

Occupation labels other than 15, 25 and 26, and the numeric order of the
11 UK regions, are not in the brief and are not filled in here.
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
    12: "Outside UK",
    13: "Other UK",
    14: "Prefer not to say",
}
# Codes 1-11 are UK regions. Names are not assigned to codes here.
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
OCCUPATION_CONFIRMED = {
    15: "Other",
    25: "Other",
    26: "Prefer not to say",
}

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
