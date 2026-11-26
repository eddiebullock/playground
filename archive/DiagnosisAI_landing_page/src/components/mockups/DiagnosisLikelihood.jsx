// src/components/mockups/DiagnosisLikelihood.jsx
import React from 'react';
import { Box, Typography, Divider } from '@mui/material';

const DiagnosisLikelihood = () => (
  <Box sx={{ p: 4, bgcolor: '#fff', borderRadius: 2, boxShadow: '0 2px 10px rgba(0,0,0,0.08)' }}>
    <Typography variant="h6" gutterBottom sx={{ color: '#1A7B8C', fontWeight: 600, mb: 3 }}>
      AI-Powered Analysis Results
    </Typography>
    
    <Box sx={{ textAlign: 'center', mb: 3 }}>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
        Diagnostic Likelihood
      </Typography>
      <Box sx={{ 
        position: 'relative',
        width: 160,
        height: 160,
        margin: '0 auto',
        borderRadius: '50%',
        background: `conic-gradient(#1A7B8C 0% 94%, #f0f0f0 94% 100%)`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        mb: 2
      }}>
        <Box sx={{ 
          width: 140, 
          height: 140, 
          borderRadius: '50%', 
          bgcolor: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column'
        }}>
          <Typography variant="h3" sx={{ fontWeight: 600, color: '#1A7B8C' }}>94%</Typography>
          <Typography variant="caption" color="text.secondary">Confidence</Typography>
        </Box>
      </Box>
      
      <Box sx={{ p: 2, bgcolor: 'rgba(26, 123, 140, 0.1)', borderRadius: 2, mb: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, color: '#1A7B8C' }}>
          ASD Diagnostic Criteria Met
        </Typography>
      </Box>
    </Box>
    
    <Divider sx={{ my: 2 }} />
    
    <Box>
      <Typography variant="subtitle2" sx={{ mb: 1.5 }}>Diagnostic Indicators</Typography>
      
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="body2">DSM-5 Alignment:</Typography>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>High</Typography>
      </Box>
      
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="body2">Interview Analysis:</Typography>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>21/23 Indicators</Typography>
      </Box>
      
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="body2">DAWBA Score Interpretation:</Typography>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>Severe Range</Typography>
      </Box>
      
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="body2">AQ Assessment:</Typography>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>32/50 (Above Threshold)</Typography>
      </Box>
    </Box>
    
    <Box sx={{ mt: 3, p: 2, bgcolor: 'rgba(230, 181, 74, 0.1)', borderRadius: 1 }}>
      <Typography variant="body2" sx={{ color: '#E6B54A', display: 'flex', alignItems: 'center' }}>
        <Box component="span" sx={{ fontWeight: 600, mr: 1 }}>Note:</Box>
        Secondary ADHD indicators detected. Recommend further assessment.
      </Typography>
    </Box>
  </Box>
);

export default DiagnosisLikelihood;