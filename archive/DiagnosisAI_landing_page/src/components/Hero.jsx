// src/components/Hero.jsx
import React from 'react';
import {
  Box,
  Button,
  Container,
  Grid,
  Typography,
  useTheme,
  Card,
  CardContent,
  useMediaQuery,
} from '@mui/material';
import { motion } from 'framer-motion';
import { Link as ScrollLink } from 'react-scroll';
import DashboardMockup from './mockups/Dashboard';

const Hero = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <Box
      id="hero"
      sx={{
        py: { xs: 6, md: 12 },
        background: `linear-gradient(135deg, ${theme.palette.background.default} 0%, ${theme.palette.background.paper} 100%)`,
      }}
    >
      <Container maxWidth="lg">
        {/* Top section with title and buttons */}
        <Box sx={{ textAlign: 'center', mb: { xs: 4, md: 6 } }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Typography
              variant="h1"
              gutterBottom
              sx={{
                color: theme.palette.primary.main,
                fontWeight: 700,
                fontSize: { xs: '2rem', sm: '2.5rem', md: '3.5rem' },
                mb: 2,
              }}
            >
              Streamlining ASD & ADHD Assessments
            </Typography>
            <Typography
              variant="h5"
              sx={{
                color: theme.palette.text.secondary,
                fontWeight: 400,
                mb: 4,
                lineHeight: 1.6,
                fontSize: { xs: '1rem', sm: '1.25rem' },
                maxWidth: '800px',
                mx: 'auto'
              }}
            >
              Reduce waitlists and increase assessment capacity with our clinician-focused AI application.
            </Typography>
            
            {/* Statistics cards - reduced size on mobile and arranged in a row */}
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              flexWrap: 'wrap', 
              gap: { xs: 2, md: 3 },
              mb: 4
            }}>
              <Card 
                elevation={0} 
                sx={{ 
                  backgroundColor: 'rgba(26, 123, 140, 0.1)', 
                  borderLeft: `3px solid ${theme.palette.primary.main}`,
                  minWidth: { xs: '70px', sm: '180px' },
                  maxWidth: { xs: '35%', sm: 'none' }
                }}
              >
                <CardContent sx={{ py: { xs: 0.5, sm: 2 }, px: { xs: 0.8, sm: 2 } }}>
                  <Typography 
                    variant="h3" 
                    color="primary" 
                    sx={{ 
                      fontWeight: 700, 
                      mb: 0.3,
                      fontSize: { xs: '1.1rem', sm: '2rem', md: '2.5rem' },
                      lineHeight: 1
                    }}
                  >
                    60%
                  </Typography>
                  <Typography 
                    variant="body2" 
                    color="text.secondary"
                    sx={{ fontSize: { xs: '0.6rem', sm: '0.8rem', md: '0.875rem' }, lineHeight: 1.2 }}
                  >
                    Reduction in assessment time
                  </Typography>
                </CardContent>
              </Card>
              <Card 
                elevation={0} 
                sx={{ 
                  backgroundColor: 'rgba(230, 181, 74, 0.1)', 
                  borderLeft: `3px solid ${theme.palette.secondary.main}`,
                  minWidth: { xs: '70px', sm: '180px' },
                  maxWidth: { xs: '35%', sm: 'none' }
                }}
              >
                <CardContent sx={{ py: { xs: 0.5, sm: 2 }, px: { xs: 0.8, sm: 2 } }}>
                  <Typography 
                    variant="h3" 
                    color="secondary" 
                    sx={{ 
                      fontWeight: 700, 
                      mb: 0.3,
                      fontSize: { xs: '1.1rem', sm: '2rem', md: '2.5rem' },
                      lineHeight: 1
                    }}
                  >
                    3x
                  </Typography>
                  <Typography 
                    variant="body2" 
                    color="text.secondary"
                    sx={{ fontSize: { xs: '0.6rem', sm: '0.8rem', md: '0.875rem' }, lineHeight: 1.2 }}
                  >
                    More assessments per clinician
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            
            {/* Call to action buttons - always in a row for mobile */}
            <Box sx={{ 
              display: 'flex', 
              gap: 2, 
              justifyContent: 'center',
              mb: { xs: 4, md: 6 },
              flexDirection: 'row',  // Always show buttons in a row
              px: { xs: 2, sm: 0 }
            }}>
              <Button
                variant="contained"
                color="primary"
                size={isMobile ? "small" : "large"}  // Smaller buttons on mobile
                component={ScrollLink}
                to="contact"
                spy={true}
                smooth={true}
                offset={-70}
                duration={500}
                sx={{ 
                  fontSize: { xs: '0.75rem', sm: '0.875rem' },
                  py: { xs: 1, sm: 1.5 }
                }}
              >
                Request Demo
              </Button>
              <Button
                variant="outlined"
                color="primary"
                size={isMobile ? "small" : "large"}  // Smaller buttons on mobile
                component={ScrollLink}
                to="features"
                spy={true}
                smooth={true}
                offset={-70}
                duration={500}
                sx={{ 
                  fontSize: { xs: '0.75rem', sm: '0.875rem' },
                  py: { xs: 1, sm: 1.5 }
                }}
              >
                Learn More
              </Button>
            </Box>
          </motion.div>
        </Box>

        {/* Dashboard mockup section */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <Box
            sx={{
              position: 'relative',
              maxWidth: '100%',
              mx: 'auto',
              mb: { xs: 4, md: 0 },
              '&::before': {
                content: '""',
                position: 'absolute',
                width: '100%',
                height: '100%',
                top: '8%',
                left: '8%',
                borderRadius: '12px',
                backgroundColor: theme.palette.primary.main,
                opacity: 0.1,
                zIndex: 0,
              },
            }}
          >
            <Card 
              sx={{ 
                overflow: 'hidden', 
                borderRadius: '12px',
                border: `1px solid ${theme.palette.divider}`,
                position: 'relative',
                zIndex: 1,
              }}
            >
              {/* Force aspect ratio for the dashboard to maintain landscape orientation */}
              <Box sx={{ 
                position: 'relative', 
                width: '100%',
                height: 0,
                // 16:9 aspect ratio for landscape orientation on all screen sizes
                paddingBottom: '56.25%', 
                overflow: 'hidden'
              }}>
                <Box sx={{ 
                  position: 'absolute', 
                  top: 0, 
                  left: 0, 
                  width: '100%', 
                  height: '100%',
                  overflowY: 'auto'
                }}>
                  <DashboardMockup />
                </Box>
              </Box>
            </Card>
          </Box>
        </motion.div>
      </Container>
    </Box>
  );
};

export default Hero;