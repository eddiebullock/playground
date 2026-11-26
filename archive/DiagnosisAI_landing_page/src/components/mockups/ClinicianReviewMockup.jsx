// src/components/mockups/ClinicianReviewMockup.jsx
import React from 'react';
import { Box, Typography } from '@mui/material';

const ClinicianReviewMockup = () => (
  <Box sx={{ p: 3, bgcolor: '#fff', borderRadius: 2, boxShadow: '0 2px 10px rgba(0,0,0,0.08)' }}>
    <Typography variant="h6" gutterBottom sx={{ color: '#1A7B8C', fontWeight: 600 }}>
      Abbreviated Clinical Session
    </Typography>
    
    <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 3 }}>
      <Box sx={{ 
        minWidth: 8, 
        height: 8, 
        borderRadius: '50%', 
        bgcolor: '#64B464',
        mt: 1.5,
        mr: 1.5
      }}></Box>
      <Box>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Pre-populated Clinical Notes</Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          Key observations and history already documented from the AI analysis
        </Typography>
      </Box>
    </Box>
    
    <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 3 }}>
      <Box sx={{ 
        minWidth: 8, 
        height: 8, 
        borderRadius: '50%', 
        bgcolor: '#64B464',
        mt: 1.5,
        mr: 1.5
      }}></Box>
      <Box>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Focused Assessment</Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          Clinician can focus on confirming diagnostic indicators rather than full assessment
        </Typography>
      </Box>
    </Box>
    
    <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 3 }}>
      <Box sx={{ 
        minWidth: 8, 
        height: 8, 
        borderRadius: '50%', 
        bgcolor: '#64B464',
        mt: 1.5,
        mr: 1.5
      }}></Box>
      <Box>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Diagnostic Confidence</Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          Review AI-identified indicators and add clinical judgment
        </Typography>
      </Box>
    </Box>
    
    <Box sx={{ p: 2, bgcolor: 'rgba(100, 180, 100, 0.1)', borderRadius: 1, mt: 4 }}>
      <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', fontWeight: 500 }}>
        <Box sx={{ 
          width: 24, 
          height: 24, 
          borderRadius: '50%', 
          bgcolor: '#64B464',
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          mr: 1,
          fontWeight: 'bold'
        }}>✓</Box>
        Most analysis work completed before the session
      </Typography>
    </Box>
  </Box>
);

export default ClinicianReviewMockup;