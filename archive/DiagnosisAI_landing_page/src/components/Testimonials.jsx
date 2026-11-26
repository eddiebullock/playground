// src/components/Testimonials.jsx
import React from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  Avatar,
  Grid,
  useTheme,
} from '@mui/material';
import { motion } from 'framer-motion';
import FormatQuoteIcon from '@mui/icons-material/FormatQuote';

const testimonials = [
  {
    id: 1,
    name: 'Dr. Sarah Johnson',
    title: 'Developmental Pediatrician',
    image: '/api/placeholder/60/60',
    quote:
      'This platform has transformed our clinic\'s assessment process. We\'ve reduced our waitlist by 40% while maintaining clinical rigor. The time savings on documentation alone make it worthwhile.',
  },
  {
    id: 2,
    name: 'Dr. Michael Chen',
    title: 'Child Psychiatrist',
    image: '/api/placeholder/60/60',
    quote:
      'The clinical insights provided by the platform help me quickly focus on the areas that need my expertise. I can now see twice as many patients while spending more meaningful time with each one.',
  },
  {
    id: 3,
    name: 'Dr. Emily Rodriguez',
    title: 'Clinical Psychologist',
    image: '/api/placeholder/60/60',
    quote:
      'Integration with our EHR was seamless. The automated reports align perfectly with our clinical standards, and the time saved allows us to provide additional support services to families.',
  },
];

const Testimonials = () => {
  const theme = useTheme();

  return (
    <Box
      id="testimonials"
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
              Clinician Feedback
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
              Here&apos;s what healthcare professionals are saying about our platform
            </Typography>
          </motion.div>
        </Box>

        <Grid container spacing={4}>
          {testimonials.map((testimonial, index) => (
            <Grid item xs={12} md={4} key={testimonial.id}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <Card
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    position: 'relative',
                    overflow: 'visible',
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: -20,
                      left: 20,
                      width: 40,
                      height: 40,
                      borderRadius: '50%',
                      backgroundColor: theme.palette.primary.main,
                      opacity: 0.1,
                      zIndex: 0,
                    },
                  }}
                >
                  <CardContent sx={{ p: 4, flexGrow: 1 }}>
                    <Box sx={{ position: 'relative' }}>
                      <FormatQuoteIcon
                        sx={{
                          position: 'absolute',
                          top: -30,
                          left: -10,
                          fontSize: 60,
                          color: `${theme.palette.primary.main}30`,
                          transform: 'scaleX(-1)',
                        }}
                      />
                      <Typography
                        variant="body1"
                        color="text.secondary"
                        paragraph
                        sx={{ position: 'relative', zIndex: 1 }}
                      >
                        &quot;{testimonial.quote}&quot;
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
                      <Avatar
                        src={testimonial.image}
                        alt={testimonial.name}
                        sx={{ width: 60, height: 60, mr: 2 }}
                      />
                      <Box>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {testimonial.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {testimonial.title}
                        </Typography>
                      </Box>
                    </Box>
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

export default Testimonials;