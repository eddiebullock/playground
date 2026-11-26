// src/components/mockups/InterviewMockup.jsx
import React from 'react';
import { Box, Typography, Chip, Divider, IconButton } from '@mui/material';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const InterviewMockup = () => (
  <Box sx={{ p: 3, bgcolor: '#fff', borderRadius: 2, boxShadow: '0 2px 10px rgba(0,0,0,0.08)' }}>
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
      <Typography variant="h6" gutterBottom sx={{ color: '#1A7B8C', fontWeight: 600, mb: 0 }}>
        Clinical Interview Assessment
      </Typography>
      <Chip 
        size="small" 
        label="Comprehensive" 
        sx={{ bgcolor: 'rgba(26, 123, 140, 0.1)', color: '#1A7B8C', fontWeight: 500 }} 
      />
    </Box>
    
    <Box sx={{ mb: 1 }}>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
        <Chip size="small" label="DAWBA" />
        <Chip size="small" label="Autism Quotient (AQ)" />
        <Chip size="small" label="Conners Rating Scale" />
        <Chip 
          size="small" 
          icon={<AttachFileIcon fontSize="small" />} 
          label="School Reports" 
          variant="outlined" 
        />
      </Box>
    </Box>
    <Divider sx={{ mb: 2 }} />
    
    <Box sx={{ maxHeight: 400, overflow: 'auto', pr: 1 }}>
      {/* Topic List */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1.5, color: 'text.secondary' }}>
          Comprehensive Interview Topics:
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {[
            'Background and Concerns',
            'Challenging Behavior',
            'Key Mental Health Topics',
            'Academic History/Scholarly Skills',
            'Family History',
            'Past Medical History',
            'Developmental History',
            'Gesture',
            'Social Imaginative Play',
            'Conversational Interchange',
            'Repetitive or Unusual Speech',
            'Nonverbal Behaviors to Regulate Emotion',
            'Developing Peer Relationships',
            'Shared Enjoyment',
            'Socioemotional Reciprocity',
            'Preoccupation',
            'Routines',
            'Repetitive Movements',
            'Sensory Sensitivities',
            'Development at and Before 36 Months'
          ].map((topic, idx) => (
            <Chip 
              key={idx} 
              size="small" 
              label={topic} 
              icon={<CheckCircleIcon fontSize="small" sx={{ fontSize: '14px !important' }} />}
              sx={{ 
                bgcolor: 'rgba(100, 180, 100, 0.1)', 
                color: '#64B464', 
                fontWeight: 500,
                '& .MuiChip-icon': { color: '#64B464' }
              }} 
            />
          ))}
        </Box>
      </Box>
      
      {/* Sample Section */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Background and Concerns</Typography>
        <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
          Please describe the primary concerns that led to seeking this assessment:
        </Typography>
        <Box sx={{ p: 2, bgcolor: 'rgba(0,0,0,0.04)', borderRadius: 1, mb: 2 }}>
          <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
            Text response field...
          </Typography>
        </Box>
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Developmental History</Typography>
        <Typography variant="body2" sx={{ mb: 1, color: 'text.secondary' }}>
          At what age did the child first walk independently?
        </Typography>
        <Box sx={{ p: 2, bgcolor: 'rgba(0,0,0,0.04)', borderRadius: 1, mb: 2 }}>
          <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
            Text response field...
          </Typography>
        </Box>
        <Typography variant="body2" sx={{ mb: 1, color: 'text.secondary' }}>
          At what age did the child speak their first words?
        </Typography>
        <Box sx={{ p: 2, bgcolor: 'rgba(0,0,0,0.04)', borderRadius: 1, mb: 2 }}>
          <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
            Text response field...
          </Typography>
        </Box>
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
          Autism Quotient (AQ) Assessment
          <Chip 
            size="small" 
            label="Standardized" 
            sx={{ ml: 1, bgcolor: 'rgba(230, 181, 74, 0.1)', color: '#E6B54A', fontWeight: 500 }} 
          />
        </Typography>
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" sx={{ mb: 1, color: 'text.secondary' }}>
            I prefer to do things with others rather than on my own:
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            {['Definitely agree', 'Slightly agree', 'Slightly disagree', 'Definitely disagree'].map((label, i) => (
              <Box key={i} sx={{ textAlign: 'center', width: '25%' }}>
                <Box sx={{ 
                  width: 20, 
                  height: 20, 
                  borderRadius: '50%', 
                  border: '2px solid #1A7B8C',
                  display: 'inline-flex',
                  mb: 0.5
                }}></Box>
                <Typography variant="caption" display="block">{label}</Typography>
              </Box>
            ))}
          </Box>
        </Box>
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
          Conners Rating Scale (ADHD)
          <Chip 
            size="small" 
            label="Standardized" 
            sx={{ ml: 1, bgcolor: 'rgba(230, 181, 74, 0.1)', color: '#E6B54A', fontWeight: 500 }} 
          />
        </Typography>
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" sx={{ mb: 1, color: 'text.secondary' }}>
            Inattention to details or makes careless mistakes:
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            {['Not at all', 'Just a little', 'Pretty much', 'Very much'].map((label, i) => (
              <Box key={i} sx={{ textAlign: 'center', width: '25%' }}>
                <Box sx={{ 
                  width: 20, 
                  height: 20, 
                  borderRadius: '50%', 
                  border: '2px solid #1A7B8C',
                  display: 'inline-flex',
                  mb: 0.5
                }}></Box>
                <Typography variant="caption" display="block">{label}</Typography>
              </Box>
            ))}
          </Box>
        </Box>
      </Box>
      
      <Box sx={{ display: 'flex', alignItems: 'center', mt: 3, py: 2, px: 3, bgcolor: 'rgba(26, 123, 140, 0.05)', borderRadius: 1 }}>
        <IconButton size="small" sx={{ mr: 1, bgcolor: 'rgba(26, 123, 140, 0.1)' }}>
          <AttachFileIcon fontSize="small" sx={{ color: '#1A7B8C' }} />
        </IconButton>
        <Box>
          <Typography variant="body2" sx={{ fontWeight: 500 }}>School Reports Included</Typography>
          <Typography variant="caption" color="text.secondary">
            Teacher evaluations, academic performance records, and behavior notes
          </Typography>
        </Box>
      </Box>
    </Box>
  </Box>
);

export default InterviewMockup;