// src/components/Header.jsx
import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Box,
  Button,
  Container,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Toolbar,
  Typography,
  useScrollTrigger,
  useTheme,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import { Link as ScrollLink } from 'react-scroll';

// Create an ElevationScroll component to show shadow on scroll
function ElevationScroll(props) {
  const { children } = props;
  const trigger = useScrollTrigger({
    disableHysteresis: true,
    threshold: 0,
  });

  return React.cloneElement(children, {
    elevation: trigger ? 4 : 0,
    sx: {
      backgroundColor: trigger ? 'background.default' : 'transparent',
      transition: 'background-color 0.3s ease',
    },
  });
}

const Header = () => {
  const theme = useTheme();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const navItems = [
    { name: 'Features', id: 'features' },
    { name: 'How It Works', id: 'how-it-works' },
    { name: 'Testimonials', id: 'testimonials' },
    { name: 'Pricing', id: 'pricing' },
    { name: 'FAQ', id: 'faq' },
  ];

  const drawer = (
    <Box onClick={handleDrawerToggle} sx={{ textAlign: 'center' }}>
      <Typography variant="h6" sx={{ my: 2 }}>
        DiagnosisAI
      </Typography>
      <List>
        {navItems.map((item) => (
          <ListItem key={item.id} disablePadding>
            <ListItemButton
              sx={{ textAlign: 'center' }}
              component={ScrollLink}
              to={item.id}
              spy={true}
              smooth={true}
              offset={-70}
              duration={500}
            >
              <ListItemText primary={item.name} />
            </ListItemButton>
          </ListItem>
        ))}
        <ListItem disablePadding>
          <ListItemButton
            sx={{
              textAlign: 'center',
              backgroundColor: theme.palette.secondary.main,
              color: theme.palette.secondary.contrastText,
              my: 1,
              mx: 2,
              borderRadius: 1,
              '&:hover': {
                backgroundColor: theme.palette.secondary.dark,
              },
            }}
            component={ScrollLink}
            to="contact"
            spy={true}
            smooth={true}
            offset={-70}
            duration={500}
          >
            <ListItemText primary="Request Demo" />
          </ListItemButton>
        </ListItem>
      </List>
    </Box>
  );

  return (
    <Box sx={{ flexGrow: 1 }}>
      <ElevationScroll>
        <AppBar 
          position="fixed" 
          color="transparent"
          sx={{
            boxShadow: scrolled ? '0px 2px 10px rgba(0, 0, 0, 0.1)' : 'none',
          }}
        >
          <Container maxWidth="lg">
            <Toolbar>
              <Typography
                variant="h6"
                component="div"
                sx={{ 
                  flexGrow: 1, 
                  color: scrolled ? theme.palette.text.primary : theme.palette.primary.main,
                  fontWeight: 700 
                }}
              >
                DiagnosisAI
              </Typography>
              <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
                {navItems.map((item) => (
                  <Button
                    key={item.id}
                    sx={{ 
                      mx: 1, 
                      color: scrolled ? theme.palette.text.primary : theme.palette.text.primary,
                      '&:hover': {
                        color: theme.palette.primary.main,
                      }
                    }}
                    component={ScrollLink}
                    to={item.id}
                    spy={true}
                    smooth={true}
                    offset={-70}
                    duration={500}
                  >
                    {item.name}
                  </Button>
                ))}
                <Button
                  variant="contained"
                  color="secondary"
                  sx={{ ml: 2 }}
                  component={ScrollLink}
                  to="contact"
                  spy={true}
                  smooth={true}
                  offset={-70}
                  duration={500}
                >
                  Request Demo
                </Button>
              </Box>
              <IconButton
                color="inherit"
                aria-label="open drawer"
                edge="end"
                onClick={handleDrawerToggle}
                sx={{ 
                  ml: 1, 
                  display: { md: 'none' },
                  color: scrolled ? theme.palette.text.primary : theme.palette.text.primary 
                }}
              >
                <MenuIcon />
              </IconButton>
            </Toolbar>
          </Container>
        </AppBar>
      </ElevationScroll>
      <Box component="nav">
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better mobile performance
          }}
          anchor="right"
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: 240 },
          }}
        >
          {drawer}
        </Drawer>
      </Box>
      <Toolbar />
    </Box>
  );
};

export default Header;