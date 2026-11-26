// src/components/FAQ.jsx
import React from 'react';
import {
  Box,
  Container,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  Link,
  useTheme,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { motion } from 'framer-motion';
import { Link as ScrollLink } from 'react-scroll';

const faqItems = [
  {
    question: 'Is this platform clinically validated?',
    answer:
      'Our platform is designed to assist clinicians, not replace their judgment. We work with leading clinical advisors to ensure our algorithms align with diagnostic standards and best practices. The clinician always maintains final control over the assessment and diagnosis process.',
  },
  {
    question: 'How does the platform ensure patient data security?',
    answer:
      'We take data security seriously. Our platform is HIPAA compliant, uses end-to-end encryption, and undergoes regular security audits. All patient data is stored securely, and we never use identifiable patient information to train our algorithms without explicit consent.',
  },
  {
    question: 'What evidence frameworks do you support?',
    answer:
      'Our platform supports major evidence frameworks including DSM-5, ICD-10, ADOS-2, ADI-R, and CASD for autism assessments, as well as DIVA, CAARS, and Conners for ADHD assessments. We can also customize frameworks based on your clinical practice needs.',
  },
  {
    question: 'Can the platform integrate with our existing EHR system?',
    answer:
      'Yes, we offer integration capabilities with major EHR systems including Epic, Cerner, Allscripts, and others. Our team will work with you to establish the appropriate integration level, from simple data export to full bidirectional integration.',
  },
  {
    question: 'How much training do clinicians need?',
    answer:
      'Most clinicians become comfortable with the platform after a 1-hour training session. We provide comprehensive documentation, video tutorials, and dedicated support during the onboarding process. Additionally, our Professional and Enterprise plans include personalized training sessions.',
  },
  {
    question: 'What kind of support is available?',
    answer:
      'All plans include access to our knowledge base and email support. Professional plans include priority support with faster response times. Enterprise customers receive dedicated account management and technical support, including emergency response protocols.',
  },
];

const FAQ = () => {
  const theme = useTheme();

  return (
    <Box
      id="faq"
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
              Frequently Asked Questions
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
              Common questions about our assessment automation platform
            </Typography>
          </motion.div>
        </Box>

        <Box sx={{ mb: 6 }}>
          {faqItems.map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Accordion
                sx={{
                  mb: 2,
                  boxShadow: 'none',
                  '&:before': { display: 'none' },
                  backgroundColor: theme.palette.background.paper,
                  borderRadius: 2,
                  border: `1px solid ${theme.palette.divider}`,
                  overflow: 'hidden',
                }}
              >
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls={`panel${index}-content`}
                  id={`panel${index}-header`}
                  sx={{
                    px: 3,
                    py: 2,
                    '&.Mui-expanded': {
                      borderBottom: `1px solid ${theme.palette.divider}`,
                    },
                  }}
                >
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {item.question}
                  </Typography>
                </AccordionSummary>
                <AccordionDetails sx={{ p: 3 }}>
                  <Typography variant="body1" color="text.secondary">
                    {item.answer}
                  </Typography>
                </AccordionDetails>
              </Accordion>
            </motion.div>
          ))}
        </Box>

        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Still have questions?
          </Typography>
          <Typography variant="body1" sx={{ mb: 3 }}>
            Contact our team for more information.{' '}
            <Link
              component={ScrollLink}
              to="contact"
              spy={true}
              smooth={true}
              offset={-70}
              duration={500}
              sx={{ color: theme.palette.primary.main, cursor: 'pointer', fontWeight: 600 }}
            >
              Get in touch
            </Link>
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default FAQ;