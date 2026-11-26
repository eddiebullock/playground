// src/components/HowItWorks.jsx
import React from 'react';
import {
  Box,
  Container,
  Grid,
  Typography,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Paper,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import { motion } from 'framer-motion';

// Import mockup components
import InterviewMockup from './mockups/InterviewMockup';
import DiagnosisLikelihood from './mockups/DiagnosisLikelihood';
import ReportContent from './mockups/ReportContent';
import ClinicianReviewMockup from './mockups/ClinicianReviewMockup';

// Updated steps with rearranged order and mockup components
const steps = [
  {
    label: 'Initial Data Collection',
    description: 'Patient information and assessment questionnaires are collected through a secure portal. Includes both standardized rating scales and a comprehensive clinical interview.',
    mockupComponent: <InterviewMockup />,
  },
  {
    label: 'Automated Processing',
    description: 'Our AI analyzes responses against established clinical frameworks, identifying key diagnostic indicators and calculating likelihood of various conditions.',
    mockupComponent: <DiagnosisLikelihood />,
  },
  {
    label: 'Report Generation',
    description: 'Comprehensive assessment reports are automatically generated with all sections pre-populated based on the collected data and AI analysis.',
    mockupComponent: <Box sx={{ maxHeight: 400, overflow: 'auto', bgcolor: '#fff', p: 2, borderRadius: 2 }}><ReportContent /></Box>,
  },
  {
    label: 'Clinician Review',
    description: 'Clinicians conduct an abbreviated assessment session to confirm findings, focusing only on key areas that require professional judgment. The bulk of the documentation work is already done.',
    mockupComponent: <ClinicianReviewMockup />,
  },
];

const HowItWorks = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <Box
      id="how-it-works"
      sx={{
        py: { xs: 8, md: 12 },
        backgroundColor: theme.palette.background.paper,
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ textAlign: 'center', mb: 8 }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <Typography
              variant="h2"
              gutterBottom
              sx={{
                fontWeight: 700,
                position: 'relative',
                display: 'inline-block',
                mb: 3,
                '&::after': {
                  content: '""',
                  position: 'absolute',
                  bottom: -10,
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: 80,
                  height: 4,
                  backgroundColor: theme.palette.primary.main,
                },
              }}
            >
              How It Works
            </Typography>
            <Typography
              variant="h5"
              sx={{
                color: theme.palette.text.secondary,
                fontWeight: 400,
                maxWidth: 800,
                mx: 'auto',
                mb: 2,
              }}
            >
              Our streamlined assessment workflow enhances clinician efficiency
            </Typography>
          </motion.div>
        </Box>

        {isMobile ? (
          // Mobile view - vertical stepper
          <Stepper orientation="vertical" activeStep={-1}>
            {steps.map((step, index) => (
              <Step key={index} expanded>
                <StepLabel
                  StepIconProps={{
                    style: {
                      color: theme.palette.primary.main,
                    },
                  }}
                >
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {step.label}
                  </Typography>
                </StepLabel>
                <StepContent>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body1" color="text.secondary">
                      {step.description}
                    </Typography>
                  </Box>
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.6 }}
                  >
                    <Paper
                      sx={{
                        overflow: 'hidden',
                        borderRadius: 2,
                        border: `1px solid ${theme.palette.divider}`,
                        mb: 3,
                      }}
                    >
                      {step.mockupComponent}
                    </Paper>
                  </motion.div>
                </StepContent>
              </Step>
            ))}
          </Stepper>
        ) : (
          // Desktop view - horizontal presentation with mockups
          <Grid container spacing={8}>
            {steps.map((step, index) => (
              <React.Fragment key={index}>
                <Grid item xs={12}>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                  >
                    <Box sx={{ display: 'flex', flexDirection: index % 2 === 0 ? 'row' : 'row-reverse' }}>
                      <Box
                        sx={{
                          flex: 1,
                          display: 'flex',
                          flexDirection: 'column',
                          justifyContent: 'center',
                          p: 4,
                        }}
                      >
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            mb: 2,
                          }}
                        >
                          <Box
                            sx={{
                              width: 48,
                              height: 48,
                              borderRadius: '50%',
                              backgroundColor: theme.palette.primary.main,
                              color: theme.palette.primary.contrastText,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              mr: 2,
                              fontWeight: 700,
                              fontSize: '1.25rem',
                            }}
                          >
                            {index + 1}
                          </Box>
                          <Typography variant="h4" sx={{ fontWeight: 600 }}>
                            {step.label}
                          </Typography>
                        </Box>
                        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                          {step.description}
                        </Typography>
                      </Box>
                      <Box sx={{ flex: 1 }}>
                        <motion.div
                          initial={{ opacity: 0, scale: 0.95 }}
                          whileInView={{ opacity: 1, scale: 1 }}
                          viewport={{ once: true }}
                          transition={{ duration: 0.6, delay: 0.2 + index * 0.1 }}
                        >
                          <Box
                            sx={{
                              position: 'relative',
                              '&::before': {
                                content: '""',
                                position: 'absolute',
                                width: '100%',
                                height: '100%',
                                top: index % 2 === 0 ? '8%' : '-8%',
                                left: index % 2 === 0 ? '8%' : '-8%',
                                borderRadius: '12px',
                                backgroundColor: theme.palette.primary.main,
                                opacity: 0.1,
                                zIndex: 0,
                              },
                            }}
                          >
                            <Paper
                              sx={{
                                overflow: 'hidden',
                                borderRadius: 2,
                                border: `1px solid ${theme.palette.divider}`,
                                position: 'relative',
                                zIndex: 1,
                                maxHeight: 500,
                                overflowY: 'auto'
                              }}
                            >
                              {step.mockupComponent}
                            </Paper>
                          </Box>
                        </motion.div>
                      </Box>
                    </Box>
                  </motion.div>
                </Grid>
                {index < steps.length - 1 && (
                  <Grid item xs={12}>
                    <Box
                      sx={{
                        height: 2,
                        width: '100%',
                        backgroundColor: `${theme.palette.primary.main}15`,
                        my: 2,
                      }}
                    />
                  </Grid>
                )}
              </React.Fragment>
            ))}
          </Grid>
        )}
      </Container>
    </Box>
  );
};

export default HowItWorks;