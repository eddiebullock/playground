// src/components/Pricing.jsx
import React from 'react';
import {
  Box,
  Button,
  Card,
  CardActions,
  CardContent,
  Container,
  Divider,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography,
  useTheme,
} from '@mui/material';
import { motion } from 'framer-motion';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import { Link as ScrollLink } from 'react-scroll';

const pricingPlans = [
  {
    title: 'Starter',
    price: '$299',
    period: 'per month',
    description: 'For small practices starting to optimize their assessment workflow',
    features: [
      'Up to 3 clinician accounts',
      'Digital intake & questionnaire processing',
      'Basic assessment report generation',
      'Email support',
      '10 assessments per month',
    ],
    buttonText: 'Get Started',
    buttonVariant: 'outlined',
    highlighted: false,
  },
  {
    title: 'Professional',
    price: '$699',
    period: 'per month',
    description: 'For growing practices seeking to significantly reduce waitlists',
    features: [
      'Up to 10 clinician accounts',
      'Advanced analytics dashboard',
      'Custom assessment templates',
      'EHR integration',
      'Priority support',
      'Unlimited assessments',
    ],
    buttonText: 'Request Demo',
    buttonVariant: 'contained',
    highlighted: true,
  },
  {
    title: 'Enterprise',
    price: 'Custom',
    period: 'pricing',
    description: 'For large clinics and healthcare networks with complex needs',
    features: [
      'Unlimited clinician accounts',
      'Dedicated account manager',
      'Custom integration services',
      'Advanced data analytics',
      'Staff training & certification',
      'SLA & uptime guarantees',
    ],
    buttonText: 'Contact Sales',
    buttonVariant: 'outlined',
    highlighted: false,
  },
];

const Pricing = () => {
  const theme = useTheme();

  return (
    <Box
      id="pricing"
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
              Pricing Plans
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
              Affordable solutions that scale with your practice
            </Typography>
          </motion.div>
        </Box>

        <Grid container spacing={4} alignItems="stretch">
          {pricingPlans.map((plan, index) => (
            <Grid item xs={12} md={4} key={index}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                style={{ height: '100%' }}
              >
                <Card
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    border: plan.highlighted
                      ? `2px solid ${theme.palette.primary.main}`
                      : `1px solid ${theme.palette.divider}`,
                    borderRadius: 3,
                    ...(plan.highlighted && {
                      backgroundColor: `${theme.palette.primary.main}05`,
                      boxShadow: `0 8px 24px rgba(26, 123, 140, 0.15)`,
                      position: 'relative',
                      '&::before': {
                        content: '"Recommended"',
                        position: 'absolute',
                        top: -12,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        backgroundColor: theme.palette.primary.main,
                        color: theme.palette.primary.contrastText,
                        fontSize: '0.75rem',
                        fontWeight: 600,
                        padding: '4px 12px',
                        borderRadius: 12,
                        textTransform: 'uppercase',
                        letterSpacing: 1,
                      },
                    }),
                  }}
                >
                  <CardContent sx={{ p: 4, flexGrow: 1 }}>
                    <Typography variant="h5" component="h3" gutterBottom sx={{ fontWeight: 600 }}>
                      {plan.title}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'baseline', mb: 2 }}>
                      <Typography variant="h3" component="span" sx={{ fontWeight: 700 }}>
                        {plan.price}
                      </Typography>
                      <Typography variant="subtitle1" component="span" color="text.secondary" sx={{ ml: 1 }}>
                        {plan.period}
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      {plan.description}
                    </Typography>
                    <Divider sx={{ my: 3 }} />
                    <List sx={{ mb: 2 }}>
                      {plan.features.map((feature, idx) => (
                        <ListItem key={idx} disableGutters>
                          <ListItemIcon sx={{ minWidth: 36 }}>
                            <CheckCircleOutlineIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText primary={feature} />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                  <CardActions sx={{ p: 4, pt: 0 }}>
                    <Button
                      fullWidth
                      variant={plan.buttonVariant}
                      color="primary"
                      size="large"
                      component={ScrollLink}
                      to="contact"
                      spy={true}
                      smooth={true}
                      offset={-70}
                      duration={500}
                    >
                      {plan.buttonText}
                    </Button>
                  </CardActions>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default Pricing;