// src/components/Footer.jsx
import React from 'react';
import {
  Box,
  Container,
  Grid,
  Link,
  Typography,
  IconButton,
  Divider,
  useTheme,
} from '@mui/material';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import TwitterIcon from '@mui/icons-material/Twitter';
import FacebookIcon from '@mui/icons-material/Facebook';
import InstagramIcon from '@mui/icons-material/Instagram';
import { Link as ScrollLink } from 'react-scroll';

const Footer = () => {
  const theme = useTheme();
  const year = new Date().getFullYear();

  const footerLinks = [
    {
      title: 'Product',
      links: [
        { name: 'Features', to: 'features' },
        { name: 'How It Works', to: 'how-it-works' },
        { name: 'Pricing', to: 'pricing' },
        { name: 'Case Studies', to: '#' },
        { name: 'Documentation', to: '#' },
      ],
    },
    {
      title: 'Company',
      links: [
        { name: 'About Us', to: '#' },
        { name: 'Careers', to: '#' },
        { name: 'Blog', to: '#' },
        { name: 'Contact', to: 'contact' },
      ],
    },
    {
      title: 'Resources',
      links: [
        { name: 'Support', to: '#' },
        { name: 'FAQ', to: 'faq' },
        { name: 'Privacy Policy', to: '#' },
        { name: 'Terms of Service', to: '#' },
        { name: 'Security', to: '#' },
      ],
    },
  ];

  return (
    <Box
      component="footer"
      sx={{
        py: 6,
        backgroundColor: theme.palette.background.default,
        borderTop: `1px solid ${theme.palette.divider}`,
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
              DiagnosisAI
            </Typography>
            <Typography 
              variant="body2" 
              color="text.secondary" 
              sx={{ mb: 3, maxWidth: 300 }}
            >
              Streamlining ASD & ADHD assessments through intelligent automation, 
              helping clinicians reduce waitlists and focus on what matters most - their patients.
            </Typography>
            <Box sx={{ mb: 3 }}>
              <IconButton
                aria-label="LinkedIn"
                sx={{ 
                  color: theme.palette.text.secondary,
                  '&:hover': { color: theme.palette.primary.main }, 
                  mr: 1 
                }}
              >
                <LinkedInIcon />
              </IconButton>
              <IconButton
                aria-label="Twitter"
                sx={{ 
                  color: theme.palette.text.secondary,
                  '&:hover': { color: theme.palette.primary.main }, 
                  mr: 1 
                }}
              >
                <TwitterIcon />
              </IconButton>
              <IconButton
                aria-label="Facebook"
                sx={{ 
                  color: theme.palette.text.secondary,
                  '&:hover': { color: theme.palette.primary.main }, 
                  mr: 1 
                }}
              >
                <FacebookIcon />
              </IconButton>
              <IconButton
                aria-label="Instagram"
                sx={{ 
                  color: theme.palette.text.secondary,
                  '&:hover': { color: theme.palette.primary.main }
                }}
              >
                <InstagramIcon />
              </IconButton>
            </Box>
          </Grid>

          {footerLinks.map((section) => (
            <Grid item xs={6} sm={4} md={2} key={section.title}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                {section.title}
              </Typography>
              <Box component="ul" sx={{ pl: 0, listStyle: 'none' }}>
                {section.links.map((link) => (
                  <Box component="li" sx={{ mb: 1 }} key={link.name}>
                    <Link
                      component={ScrollLink}
                      to={link.to}
                      spy={true}
                      smooth={true}
                      offset={-70}
                      duration={500}
                      sx={{
                        color: theme.palette.text.secondary,
                        textDecoration: 'none',
                        '&:hover': {
                          color: theme.palette.primary.main,
                          textDecoration: 'none',
                        },
                        cursor: 'pointer',
                      }}
                    >
                      {link.name}
                    </Link>
                  </Box>
                ))}
              </Box>
            </Grid>
          ))}

          <Grid item xs={12} sm={4} md={2}>
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              Contact
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              123 Medical Plaza, Suite 500
              <br />
              Boston, MA 02115
            </Typography>
            <Typography variant="body2" color="text.secondary">
              info@assessmentai.com
              <br />
              +1 (555) 123-4567
            </Typography>
          </Grid>
        </Grid>

        <Divider sx={{ my: 4 }} />

        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            justifyContent: 'space-between',
            alignItems: { xs: 'center', sm: 'flex-start' },
          }}
        >
          <Typography variant="body2" color="text.secondary">
            © {year} DiagnosisAI. All rights reserved.
          </Typography>
          <Box
            sx={{
              display: 'flex',
              mt: { xs: 2, sm: 0 },
            }}
          >
            <Link
              href="#"
              variant="body2"
              color="text.secondary"
              sx={{ mr: 3, textDecoration: 'none', '&:hover': { color: theme.palette.primary.main } }}
            >
              Privacy Policy
            </Link>
            <Link
              href="#"
              variant="body2"
              color="text.secondary"
              sx={{ mr: 3, textDecoration: 'none', '&:hover': { color: theme.palette.primary.main } }}
            >
              Terms of Service
            </Link>
            <Link
              href="#"
              variant="body2"
              color="text.secondary"
              sx={{ textDecoration: 'none', '&:hover': { color: theme.palette.primary.main } }}
            >
              Cookie Policy
            </Link>
          </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;