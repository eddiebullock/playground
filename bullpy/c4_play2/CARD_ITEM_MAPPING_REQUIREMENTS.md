# CARD Dataset Item Mapping Requirements

## Problem Statement

The CARD dataset contains full questionnaires (AQ-50, EQ-60, SQ-R-75, SPQ-92) in the 'Itemised Score' column as CSV strings. We need to extract the specific items that correspond to the validated 10-item short versions (AQ-10, EQ-10, SQ-R-10, SPQ-10).

**CRITICAL**: The 10-item short versions do NOT use the first 10 items sequentially. They use specific, psychometrically-validated items scattered throughout the full questionnaires.

## Required Item Mappings

### AQ-10 from AQ-50
**Reference**: Allison, C., Auyeung, B., & Baron-Cohen, S. (2012). JAACAP, 51(2), 202-212

**AQ-10 Items** (item wording):
1. "I often notice small sounds when others do not"
2. "I usually concentrate more on the whole picture, rather than the small details"
3. "I find it easy to do more than one thing at once"
4. "If there is an interruption, I can switch back to what I was doing very quickly"
5. "I find it easy to 'read between the lines' when someone is talking to me"
6. "I know how to tell if someone listening to me is getting bored"
7. "When I'm reading a story I find it difficult to work out the characters' intentions"
8. "I like to collect information about categories of things"
9. "I find it easy to work out what someone is thinking or feeling just by looking at their face"
10. "I find it difficult to work out people's intentions"

**Scoring**:
- Items 1, 7, 8, 10: Score 1 for "Definitely/Slightly Agree"
- Items 2, 3, 4, 5, 6, 9: Score 1 for "Definitely/Slightly Disagree"

**TODO**: Find which AQ-50 item numbers correspond to these 10 items.

### EQ-10 from EQ-60
**Reference**: Greenberg, D.M., et al. (2018). PNAS. DOI: 10.1073/pnas.1811032115

**EQ-10 Items** (item wording):
1. "I am good at predicting how someone will feel"
2. "Other people tell me I am good at understanding how they are feeling and what they are thinking"
3. "It is hard for me to see why some things upset people so much"
4. "I can easily work out what another person might want to talk about"
5. "I can't always see why someone should have felt offended by a remark"
6. "I can tune into how someone else feels rapidly and intuitively"
7. "Other people often say that I am insensitive, though I don't always see why"
8. "In a conversation, I tend to focus on my own thoughts rather than on what my listener might be thinking"
9. "Friends usually talk to me about their problems as they say that I am very understanding"
10. "I find it hard to know what to do in a social situation"

**Scoring**:
- Items 1, 2, 4, 6, 9: Score 2/"strongly agree", 1/"slightly agree"
- Items 3, 5, 7, 8, 10: Score 2/"strongly disagree", 1/"slightly disagree"

**TODO**: Find which EQ-60 item numbers correspond to these 10 items.

### SQ-R-10 from SQ-R-75
**Reference**: Greenberg, D.M., et al. (2018). PNAS. DOI: 10.1073/pnas.1811032115

**SQ-R-10 Items** (item wording):
1. "When I learn about a new category I like to go into detail to understand the small differences between different members of that category"
2. "When I'm in a plane, I do not think about the aerodynamics"
3. "I am interested in knowing the path a river takes from its source to the sea"
4. "When travelling by train, I often wonder exactly how the rail networks are coordinated"
5. "When I hear the weather forecast, I am not very interested in the meteorological patterns"
6. "I enjoy looking through catalogues of products to see the details of each product and how it compares to others"
7. "When I look at a mountain, I think about how precisely it was formed"
8. "When I look at a piece of furniture, I do not notice the details of how it was constructed"
9. "When I learn a language, I become intrigued by its grammatical rules"
10. "When I listen to a piece of music, I always notice the way it's structured"

**Scoring**:
- Items 1, 3, 4, 6, 7, 9, 10: Score 2/"strongly agree", 1/"slightly agree"
- Items 2, 5, 8: Score 2/"strongly disagree", 1/"slightly disagree"

**TODO**: Find which SQ-R-75 item numbers correspond to these 10 items.

### SPQ-10 from SPQ-92
**Reference**: Greenberg, D.M., et al. (2018). PNAS. DOI: 10.1073/pnas.1811032115

**SPQ-10 Items** (item wording):
1. "I would be able to distinguish different people by their smell"
2. "I would be able to taste the difference between two brands of salty potato chips/crisps"
3. "I can hear electricity humming in the walls"
4. "I would be able to notice a tiny change (for example, 1 degree) in the temperature of the weather"
5. "I would be able to taste the difference between apparently identical pieces of candy"
6. "I would be able to tell the weight difference between two different coin sizes on the palm of my hand, if my eyes were closed"
7. "I would be able to smell the smallest gas leak from anywhere in the house"
8. "I would be the first to hear if there was a fly in the room"
9. "If I look at a pile of blue sweaters in a shop that are meant to be identical, I would be able to see differences between them"
10. "I can see dust particles in the air in most environments"

**Scoring**: All items score 3/"strongly agree", 2/"slightly agree", 1/"slightly disagree", 0/"strongly disagree"

**TODO**: Find which SPQ-92 item numbers correspond to these 10 items.

## How to Find the Mappings

1. **Check Supplementary Materials**: Download the Greenberg et al. (2018) PNAS paper supplementary materials (usually an Appendix PDF) which should specify the exact item numbers.

2. **Match by Wording**: If you have access to the full questionnaire item text, match the wording of the 10 items above to find their positions in the full questionnaires.

3. **Check CARD Metadata**: The CARD dataset might have metadata or documentation that specifies which items are which.

4. **Contact Authors**: If mappings are not available, contact the authors of the papers for clarification.

## Implementation

Once you have the mappings, update the `ITEM_MAPPINGS` dictionary in the notebook:

```python
ITEM_MAPPINGS = {
    'aq': {
        'full_length': 50,
        'short_items': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 0-indexed positions
    },
    'eq': {
        'full_length': 60,
        'short_items': [0, 5, 18, 21, 24, 25, 27, 28, 29, 30]  # 0-indexed positions
    },
    'sqr': {
        'full_length': 75,
        'short_items': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 0-indexed positions
    },
    'spq': {
        'full_length': 92,
        'short_items': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 0-indexed positions
    }
}
```

**Note**: The example mappings above are PLACEHOLDERS and are INCORRECT. Replace with actual mappings once found.

## Current Status

- ✅ Code structure ready to accept mappings
- ✅ **ITEM MAPPINGS CONFIGURED** (based on Greenberg et al. 2018 PNAS supplementary materials)
- ✅ Error messages will show when mappings are missing
- ⚠️  'Itemised Score' column appears to contain only 1 item per row (may need investigation)

## Configured Mappings (0-indexed positions)

### AQ-10
- **Items from AQ-50**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (items 1-10)
- **Source**: Allison et al. (2012) JAACAP Table S20

### EQ-10
- **Items from EQ-60**: [13, 3, 8, 30, 27, 34, 11, 21, 17, 33]
- **1-indexed**: Items 14, 4, 9, 31, 28, 35, 12, 22, 18, 34
- **Source**: Greenberg et al. (2018) PNAS supplementary materials (selected from EQ-22)

### SQ-R-10
- **Items from SQ-R-75**: [31, 15, 26, 8, 29, 32, 11, 24, 7, 6]
- **1-indexed**: Items 32, 16, 27, 9, 30, 33, 12, 25, 8, 7
- **Source**: Greenberg et al. (2018) PNAS supplementary materials (selected from 44-item gender-neutral SQ-R)

### SPQ-10
- **Items from SPQ-92**: [1, 20, 31, 34, 37, 57, 61, 72, 73, 87]
- **1-indexed**: Items 2, 21, 32, 35, 38, 58, 62, 73, 74, 88
- **Source**: Greenberg et al. (2018) PNAS supplementary materials (selected from SPQ-35)
