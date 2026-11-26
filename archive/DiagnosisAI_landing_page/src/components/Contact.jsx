// src/components/Contact.jsx
import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  Container,
  Grid,
  TextField,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Snackbar,
  Alert,
  useTheme,
} from '@mui/material';
import { motion } from 'framer-motion';
import EmailIcon from '@mui/icons-material/Email';
import LocalPhoneIcon from '@mui/icons-material/LocalPhone';
import LocationOnIcon from '@mui/icons-material/LocationOn';

const Contact = () => {
  const theme = useTheme();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    organization: '',
    role: '',
    interestedIn: '',
    message: '',
  });
  const [errors, setErrors] = useState({});
  const [openSnackbar, setOpenSnackbar] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    // Clear error when user corrects field
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: '' }));
    }
  };

  const validate = () => {
    const newErrors = {};
    if (!formData.name.trim()) newErrors.name = 'Name is required';
    if (!formData.email.trim()) newErrors.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(formData.email))
      newErrors.email = 'Email is invalid';
    if (!formData.organization.trim())
      newErrors.organization = 'Organization is required';
    if (!formData.role) newErrors.role = 'Role is required';
    if (!formData.interestedIn)
      newErrors.interestedIn = 'Please select an option';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validate()) {
      // In a real implementation, this would send data to a server
      console.log('Form submitted:', formData);
      setOpenSnackbar(true);
      // Reset form
      setFormData({
        name: '',
        email: '',
        phone: '',
        organization: '',
        role: '',
        interestedIn: '',
        message: '',
      });
    }
  };

  const handleCloseSnackbar = () => {
    setOpenSnackbar(false);
  };

  const contactInfo = [
    {
      icon: <EmailIcon />,
      title: 'Email',
      content: 'info@assessmentai.com',
    },
    {
      icon: <LocalPhoneIcon />,
      title: 'Phone',
      content: '+1 (555) 123-4567',
    },
    {
      icon: <LocationOnIcon />,
      title: 'Office',
      content: '123 Medical Plaza, Suite 500, Boston, MA 02115',
    },
  ];

  return (
    <Box
      id="contact"
      sx={{
        py: { xs: 8, md: 12 },
        backgroundColor: theme.palette.background.paper,
        background: `linear-gradient(180deg, ${theme.palette.background.paper} 0%, ${theme.palette.background.default} 100%)`,
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
              Request a Demo
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
              See how our platform can transform your assessment process
            </Typography>
          </motion.div>
        </Box>

        <Grid container spacing={6}>
          <Grid item xs={12} md={5}>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <Box>
                <Typography variant="h4" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                  Get in Touch
                </Typography>
                <Typography variant="body1" paragraph color="text.secondary">
                  Schedule a personalized demo with our team to see how our platform can reduce your assessment waitlist and improve clinical efficiency.
                </Typography>
                <Typography variant="body1" paragraph color="text.secondary">
                  We'll customize the demonstration to address your specific clinical needs and operational challenges.
                </Typography>

                <Box sx={{ mt: 4 }}>
                  {contactInfo.map((info, index) => (
                    <Box
                      key={index}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        mb: 3,
                      }}
                    >
                      <Box
                        sx={{
                          width: 40,
                          height: 40,
                          borderRadius: 2,
                          backgroundColor: `${theme.palette.primary.main}15`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          mr: 2,
                          color: theme.palette.primary.main,
                        }}
                      >
                        {info.icon}
                      </Box>
                      <Box>
                        <Typography variant="subtitle2" color="text.secondary">
                          {info.title}
                        </Typography>
                        <Typography variant="body1" sx={{ fontWeight: 500 }}>
                          {info.content}
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </Box>
              </Box>
            </motion.div>
          </Grid>

          <Grid item xs={12} md={7}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <Card sx={{ p: 4, borderRadius: 3 }}>
                <form onSubmit={handleSubmit}>
                  <Grid container spacing={3}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Full Name"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        error={!!errors.name}
                        helperText={errors.name}
                        required
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Email Address"
                        name="email"
                        type="email"
                        value={formData.email}
                        onChange={handleChange}
                        error={!!errors.email}
                        helperText={errors.email}
                        required
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Phone Number"
                        name="phone"
                        value={formData.phone}
                        onChange={handleChange}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Organization"
                        name="organization"
                        value={formData.organization}
                        onChange={handleChange}
                        error={!!errors.organization}
                        helperText={errors.organization}
                        required
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth error={!!errors.role} required>
                        <InputLabel>Your Role</InputLabel>
                        <Select
                          name="role"
                          value={formData.role}
                          label="Your Role"
                          onChange={handleChange}
                        >
                          <MenuItem value="Physician">Physician</MenuItem>
                          <MenuItem value="Psychologist">Psychologist</MenuItem>
                          <MenuItem value="Psychiatrist">Psychiatrist</MenuItem>
                          <MenuItem value="Administrator">
                            Administrator
                          </MenuItem>
                          <MenuItem value="Other">Other</MenuItem>
                        </Select>
                        {errors.role && (
                          <Typography variant="caption" color="error">
                            {errors.role}
                          </Typography>
                        )}
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl
                        fullWidth
                        error={!!errors.interestedIn}
                        required
                      >
                        <InputLabel>Interested In</InputLabel>
                        <Select
                          name="interestedIn"
                          value={formData.interestedIn}
                          label="Interested In"
                          onChange={handleChange}
                        >
                          <MenuItem value="Starter Plan">
                            Starter Plan
                          </MenuItem>
                          <MenuItem value="Professional Plan">
                            Professional Plan
                          </MenuItem>
                          <MenuItem value="Enterprise Plan">
                            Enterprise Plan
                          </MenuItem>
                          <MenuItem value="Just Information">
                            Just Information
                          </MenuItem>
                        </Select>
                        {errors.interestedIn && (
                          <Typography variant="caption" color="error">
                            {errors.interestedIn}
                          </Typography>
                        )}
                      </FormControl>
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Message (Optional)"
                        name="message"
                        multiline
                        rows={4}
                        value={formData.message}
                        onChange={handleChange}
                        placeholder="Tell us about your specific needs or challenges..."
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Button
                        type="submit"
                        variant="contained"
                        color="primary"
                        size="large"
                        fullWidth
                        sx={{ py: 1.5 }}
                      >
                        Request Demo
                      </Button>
                    </Grid>
                  </Grid>
                </form>
              </Card>
            </motion.div>
          </Grid>
        </Grid>
      </Container>
      
      <Snackbar 
        open={openSnackbar} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity="success" sx={{ width: '100%' }}>
          Thank you! Your demo request has been submitted successfully. Our team will contact you soon.
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Contact;