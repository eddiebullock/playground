// src/components/Features.jsx
import React from 'react';
import {
  Box,
  Container,
  Grid,
  Typography,
  Card,
  CardContent,
  IconButton,
  useTheme,
} from '@mui/material';
import { motion } from 'framer-motion';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import SecurityIcon from '@mui/icons-material/Security';
import SettingsIcon from '@mui/icons-material/Settings';
import AssignmentIcon from '@mui/icons-material/Assignment';

const featuresList = [
  {
    title: 'Automated Screening',
    description: 'Intelligently processes assessment data to identify patterns consistent with ASD and ADHD, saving clinicians hours per case.',
    icon: <AutoGraphIcon fontSize="large" />,
    delay: 0,
  },
  {
    title: 'Comprehensive Reports',
    description: 'Generates detailed clinical reports that clinicians can review, edit, and finalize in a fraction of the time.',
    icon: <AssignmentIcon fontSize="large" />,
    delay: 0.1,
  },
  {
    title: 'Time Efficiency',
    description: 'Reduces assessment time by up to 60%, allowing clinicians to evaluate three times more patients and significantly decrease waitlists.',
    icon: <AnalyticsIcon fontSize="large" />,
    delay: 0.2,
  },
  {
    title: 'Advanced AI Technology',
    description: 'Leverages state-of-the-art large language models and machine learning algorithms trained on clinical data to deliver accurate assessment insights.',
    icon: <SmartToyIcon fontSize="large" />,
    delay: 0.3,
  },
  {
    title: 'Data Privacy & Security',
    description: 'Prioritizes patient confidentiality with end-to-end encryption, HIPAA compliance, and secure data storage with no third-party access.',
    icon: <SecurityIcon fontSize="large" />,
    delay: 0.4,
  },
  {
    title: 'Customizable Assessments',
    description: 'Allows clinicians to select preferred questionnaires, and interview questions to tailor the evaluation process to their practice.',
    icon: <SettingsIcon fontSize="large" />,
    delay: 0.5,
  },
];

const Features = () => {
  const theme = useTheme();

  return (
    <Box
      id="features"
      sx={{
        py: { xs: 8, md: 12 },
        backgroundColor: theme.palette.background.default,
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
              Key Features
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
              Streamlining the assessment process while maintaining clinical excellence
            </Typography>
          </motion.div>
        </Box>

        <Grid container spacing={4}>
          {featuresList.map((feature, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: feature.delay }}
              >
                <Card
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      boxShadow: '0 12px 20px rgba(0, 0, 0, 0.1)',
                    },
                  }}
                >
                  <CardContent sx={{ flexGrow: 1, p: 4 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
                      <IconButton
                        sx={{
                          backgroundColor: `${theme.palette.primary.main}15`,
                          color: theme.palette.primary.main,
                          '&:hover': {
                            backgroundColor: `${theme.palette.primary.main}25`,
                          },
                        }}
                        size="large"
                        disableRipple
                      >
                        {feature.icon}
                      </IconButton>
                    </Box>
                    <Typography variant="h5" component="h3" gutterBottom sx={{ fontWeight: 600, textAlign: 'center' }}>
                      {feature.title}
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center' }}>
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default Features;