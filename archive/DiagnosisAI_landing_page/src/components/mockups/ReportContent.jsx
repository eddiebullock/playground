// src/components/mockups/ReportContent.jsx
import React from 'react';
import { 
  Box, 
  Typography, 
  Divider
} from '@mui/material';

const ReportContent = () => {
  // Shortened report sections with ellipses
  const reportSections = [
    {
      title: "Background and Concerns",
      content: "The parents sought an autism assessment for John due to concerns about his developmental milestones and social behaviors, particularly in relation to his communication skills. Initially, they noticed that John's communication was limited and observed a lack of gestures such as pointing or waving, which appeared subtle at first..."
    },
    {
      title: "Key Mental Health Topics",
      content: "During the assessment, several key mental health topics were identified, highlighting areas of concern that may warrant further exploration. The individual's communication style appears to be a notable area of focus. It was noted that when interested in a particular subject, he would focus intensely on it..."
    },
    {
      title: "Challenging Behavior",
      content: "The assessment identified several behavioral challenges that may impact John's daily functioning. His preferences regarding food suggest potential sensitivity and rigidity in his behavior. John shows aversion to certain food textures which may indicate a level of discomfort with unpredictability..."
    },
    {
      title: "Academic History/Scholarly Skills",
      content: "The academic history and scholarly skills of the individual assessed provide insights into their performance across various educational settings. The transcript does not contain specific references to the client's academic performance or specific strengths and weaknesses in learning..."
    },
    {
      title: "Family History",
      content: "The family history reveals pertinent insights into potential genetic and environmental factors influencing the client's development. There is no specific mention of a direct family history of autism or ADHD in the provided information..."
    },
    {
      title: "Past Medical History",
      content: "The individual in this assessment has exhibited several noteworthy medical and behavioral characteristics throughout early development. His sensory responses have been particularly pronounced, as evidenced by reactions to clothing and food preferences..."
    },
    {
      title: "Developmental History",
      content: "John's language development has raised concerns due to noted difficulties in matching his tone to various social situations. The observation regarding his language use highlights a potential challenge in social communication..."
    },
    {
      title: "Gesture",
      content: "The subject displays a limited use of gestures in communication, which may be indicative of atypical developmental patterns commonly associated with autism spectrum disorders. The information provided indicates that he did not exhibit typical gestures such as pointing or waving..."
    },
    {
      title: "Social Imaginative Play",
      content: "The assessment of social imaginative play reveals significant characteristics and challenges in the child's ability to engage in pretend play and interact meaningfully with peers. Observations indicate that the child demonstrates a preference for solitary play..."
    },
    {
      title: "Conversational Interchange",
      content: "The child exhibits notable challenges in conversational interchange, characterized by a predominantly one-sided communication style. He engages in minimal reciprocal interaction, tending to respond to questions with brief answers but rarely initiates topics..."
    },
    {
      title: "Repetitive or Unusual Speech",
      content: "The assessment revealed significant aspects of repetitive and unusual speech patterns in the child's communication style. Observations indicated that he does not typically engage in back-and-forth conversations..."
    },
    {
      title: "Nonverbal Behaviors to Regulate Emotion",
      content: "John exhibits distinct challenges in utilizing nonverbal behaviors to express and regulate his emotions. Throughout the assessment, it was noted that he tends not to employ facial expressions, gestures, or vocal intonations in ways that align with typical emotional communication..."
    },
    {
      title: "Developing Peer Relationships",
      content: "John's ability to make and maintain friendships appears to be significantly impacted by his unique social communication style. Observations indicate a tendency to prefer solitary play, suggesting limited engagement with peers..."
    },
    {
      title: "Shared Enjoyment",
      content: "The assessment of shared enjoyment reveals significant insights into the child's social engagement and interpersonal interactions. It has been noted that the child displays a clear preference for solitary play, indicating challenges in engaging with peers..."
    },
    {
      title: "Socioemotional Reciprocity",
      content: "The assessment of socioemotional reciprocity is crucial in understanding the child's interactions with others. This area examines the child's responses to others' emotions or their capacity for empathy..."
    },
    {
      title: "Preoccupation",
      content: "The assessment did not yield specific details regarding the child's preoccupations or intense interests. Further exploration of the child's focus areas would enhance understanding of their engagement patterns..."
    },
    {
      title: "Routines",
      content: "The observed rigidity in the child's routines is pronounced, particularly concerning his bedtime routine. The child exhibits a significant emotional response to any alterations, indicating a strong reliance on predictable patterns..."
    },
    {
      title: "Repetitive Movements",
      content: "The assessment of repetitive movements and behaviors in the child indicates a significant presence of stimming behaviors, which are often characteristic of individuals on the autism spectrum..."
    },
    {
      title: "Sensory Sensitivities",
      content: "The assessment revealed notable sensory sensitivities that may be characteristic of the individual in question. Sensory experiences can significantly influence behavior and social interaction, particularly in children on the autism spectrum..."
    },
    {
      title: "Development at and Before 36 Months",
      content: "John's developmental milestones in several key areas prior to 36 months present a mixed picture. His mother reported that he began crawling at around 10 months and walked at approximately 14 months, which aligns with typical physical developmental expectations..."
    }
  ];

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <Typography variant="h4" sx={{ mb: 3, color: '#1A7B8C', fontWeight: 600 }}>
        Assessment Report
      </Typography>
      
      {reportSections.map((section, index) => (
        <Box key={index} sx={{ mb: 4 }}>
          <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
            {section.title}
          </Typography>
          <Typography variant="body2" paragraph>
            {section.content}
          </Typography>
          {index < reportSections.length - 1 && <Divider sx={{ mt: 2 }} />}
        </Box>
      ))}
      
      <Box sx={{ mt: 4, bgcolor: 'rgba(26, 123, 140, 0.1)', p: 2, borderRadius: 1 }}>
        <Typography variant="body2">
          <strong>Note:</strong> This report was generated from a text-based interview and includes data from school reports. All diagnostic recommendations should be reviewed by a qualified healthcare professional.
        </Typography>
      </Box>
    </Box>
  );
};

export default ReportContent;