# EU-Emotion vs CAM Label Mapping Analysis

## Summary

**Critical Finding**: There is **ZERO direct overlap** between EU-Emotion emotions and CAM concepts.

- **EU-Emotion**: 27 emotions (mostly basic emotions: afraid, angry, happy, sad, etc.)
- **CAM**: 20 complex concepts (appalled, appealing, confronted, distaste, etc.)
- **Direct matches**: 0
- **Partial matches**: 0

## Detailed Comparison

### EU-Emotion Emotions (27 total)
```
afraid, afraid low intensity, angry, angry low intensity, ashamed, bored, 
disappointed, disgusted, disgusted low intensity, excited, frustrated, 
happy, happy low intensity, hurt, interested, jealous, joking, kind, 
neutral, proud, sad, sad low intensity, sneaky, surprised, 
surprised low intensity, unfriendly, worried
```

### CAM Concepts (20 total)
```
appalled, appealing, confronted, distaste, empathic, exonerated, 
grave, guarded, insincere, intimate, lured, mortified, nostalgic, 
reassured, resentful, stern, subdued, subservient, uneasy, vibrant
```

## Semantic Relationships (Potential Mappings)

While there are no direct matches, there may be semantic relationships:

| EU-Emotion | CAM Concept | Relationship |
|------------|-------------|--------------|
| ashamed | mortified | Both involve shame/embarrassment |
| disgusted | appalled, distaste | All involve disgust |
| afraid | confronted, guarded, uneasy | All involve fear/anxiety |
| unfriendly | insincere, stern | All involve negative social interaction |
| interested | lured, appealing | All involve attraction/interest |
| sad | grave, nostalgic, subdued | All involve sadness/melancholy |
| happy | vibrant | Both involve positive energy |
| kind | empathic | Both involve positive social connection |

## Impact on Model Performance

**Current Results:**
- EU-Emotion validation accuracy: 51.11% (learning EU-Emotion emotions)
- CAM test accuracy: 21.43% (worse than 37% zero-shot baseline)

**Why Performance is Poor:**
1. **Label Mismatch**: Model learned 27 EU-Emotion emotions, but CAM tests 20 completely different concepts
2. **Domain Gap**: EU-Emotion uses basic emotions; CAM uses complex, nuanced mental states
3. **Transfer Learning Challenge**: Model needs to generalize from basic emotions to complex concepts

## Recommendations

### Option 1: Two-Stage Fine-Tuning (Recommended)
1. **Stage 1**: Fine-tune on EU-Emotion (current step) - teaches general emotion recognition
2. **Stage 2**: Fine-tune on CAM training split - adapts to CAM-specific concepts

This approach:
- Uses EU-Emotion to learn general emotion features
- Uses CAM to learn the specific 20 concepts needed for evaluation
- Matches the original plan for two-stage fine-tuning

### Option 2: Direct CAM Fine-Tuning
- Skip EU-Emotion entirely
- Fine-tune directly on CAM training split
- Simpler but may have less data

### Option 3: Semantic Mapping
- Create a mapping from EU-Emotion emotions to CAM concepts
- Use this mapping during training or evaluation
- More complex, may not capture nuances

## Conclusion

The label mismatch explains the poor CAM performance. The model is learning the wrong labels. **Two-stage fine-tuning is essential** - EU-Emotion provides general emotion features, but CAM fine-tuning is needed to learn the specific 20 concepts.




