// src/components/mockups/Dashboard.jsx
import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Divider, 
  Chip, 
  Avatar, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  LinearProgress,
  Grid,
  Paper,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  useMediaQuery,
  useTheme,
  IconButton
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import AssessmentIcon from '@mui/icons-material/Assessment';
import DescriptionIcon from '@mui/icons-material/Description';
import MedicalInformationIcon from '@mui/icons-material/MedicalInformation';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';

const Dashboard = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [reportOpen, setReportOpen] = useState(false);
  
  // Always show all sections expanded in mobile view to maintain full landscape view
  const [expandedSections, setExpandedSections] = useState({
    questionnaires: true,
    report: true,
    diagnosis: true
  });
  
  // Import the ReportContent component for use in the dialog
  const ReportContent = React.lazy(() => import('./ReportContent'));
  
  // Sample assessment sections with completion status
  const reportSections = [
    { name: 'Background and Concerns', completed: true },
    { name: 'Challenging Behavior', completed: true },
    { name: 'Key Mental Health Topics', completed: true },
    { name: 'Academic History/Scholarly Skills', completed: true },
    { name: 'Family History', completed: true },
    { name: 'Past Medical History', completed: true },
    { name: 'Developmental History', completed: true },
    { name: 'Gesture', completed: true },
    { name: 'Social Imaginative Play', completed: true },
    { name: 'Conversational Interchange', completed: true },
    { name: 'Repetitive or Unusual Speech', completed: true },
    { name: 'Nonverbal Behaviors to Regulate Emotion', completed: true },
    { name: 'Developing Peer Relationships', completed: true },
    { name: 'Shared Enjoyment', completed: true },
    { name: 'Socioemotional Reciprocity', completed: true },
    { name: 'Preoccupation', completed: true },
    { name: 'Routines', completed: true },
    { name: 'Repetitive Movements', completed: true },
    { name: 'Sensory Sensitivities', completed: true },
    { name: 'Development at and Before 36 Months', completed: true },
  ];
  
  const handleOpenReport = () => {
    setReportOpen(true);
  };
  
  const handleCloseReport = () => {
    setReportOpen(false);
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  return (
    <Box sx={{ p: 0 }}>
      {/* Report Dialog */}
      <Dialog 
        open={reportOpen} 
        onClose={handleCloseReport}
        maxWidth="md"
        fullWidth
        scroll="paper"
      >
        <DialogTitle sx={{ bgcolor: '#1A7B8C', color: 'white' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            Assessment Report
            <Chip 
              label="Patient: John" 
              size="small" 
              avatar={<Avatar sx={{ bgcolor: 'white', color: '#1A7B8C' }}>J</Avatar>}
              sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} 
            />
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          <React.Suspense fallback={<Box sx={{ p: 4, textAlign: 'center' }}><Typography>Loading report...</Typography></Box>}>
            <ReportContent />
          </React.Suspense>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseReport} color="primary">Close</Button>
          <Button variant="contained" color="primary">Download Report</Button>
        </DialogActions>
      </Dialog>
      
      {/* Dashboard Header */}
      <Box sx={{ p: { xs: 0.5, sm: 2 }, bgcolor: '#1A7B8C', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="body2" sx={{ 
          fontSize: { xs: '0.55rem', sm: '1rem' }, 
          fontWeight: 'medium' 
        }}>
          DiagnosisAI Dashboard
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          {/* Custom chip without Avatar component */}
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            bgcolor: 'rgba(255,255,255,0.2)',
            borderRadius: '12px',
            height: { xs: 12, sm: 32 },
            px: { xs: 0.5, sm: 1 }
          }}>
            {/* Custom mini avatar */}
            <Box sx={{
              width: { xs: 6, sm: 24 },
              height: { xs: 6, sm: 24 },
              borderRadius: '50%',
              bgcolor: 'white',
              color: '#1A7B8C',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mr: 0.5,
              fontSize: { xs: '0.15rem', sm: '0.8rem' }
            }}>
              DS
            </Box>
            <Typography sx={{ 
              fontSize: { xs: '0.45rem', sm: '0.8rem' },
              color: 'white'
            }}>
              Dr. Smith
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Dashboard Content */}
      <Box sx={{ p: { xs: 0.5, sm: 3 } }}>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5, fontSize: { xs: '0.4rem', sm: '0.875rem' } }}>
          Assessment Date: March 3, 2025 | Patient: John (8)
        </Typography>
        
        {/* On mobile, use a row layout instead of column layout to maintain landscape */}
        <Grid container spacing={{ xs: 0.5, sm: 3 }} sx={{
          flexDirection: { xs: 'row', md: 'row' }  // Always use row layout
        }}>
          {/* Section 1: Questionnaires Completed */}
          <Grid item xs={3} md={4}>
            <Card sx={{ mb: { xs: 1, sm: 3 }, overflow: 'hidden' }}>
              <CardContent sx={{ p: { xs: 0.5, sm: 3 } }}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  mb: expandedSections.questionnaires ? { xs: 0.5, sm: 2 } : 0
                }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <AssessmentIcon sx={{ color: '#1A7B8C', mr: 0.5, fontSize: { xs: 12, sm: 24 } }} />
                    <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.45rem', sm: '1.25rem' } }}>
                      Questionnaires
                    </Typography>
                  </Box>
                </Box>
                
                {expandedSections.questionnaires && (
                  <>
                    <Divider sx={{ mb: { xs: 0.5, sm: 2 } }} />
                    
                    <List sx={{ p: 0 }}>
                      <ListItem sx={{ px: { xs: 0.5, sm: 1 }, py: { xs: 0.3, sm: 1 } }}>
                        <ListItemIcon sx={{ minWidth: { xs: 18, sm: 40 } }}>
                          <CheckCircleIcon sx={{ color: '#64B464', fontSize: { xs: 10, sm: 20 } }} />
                        </ListItemIcon>
                                                  <ListItemText 
                          primary={<Typography variant="body1" sx={{ fontSize: { xs: '0.4rem', sm: '1rem' } }}>DAWBA</Typography>}
                          secondary={<Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.4rem', sm: '0.875rem' } }}>Score: 17/25</Typography>}
                        />
                      </ListItem>
                      
                      <ListItem sx={{ px: { xs: 0.5, sm: 1 }, py: { xs: 0.3, sm: 1 } }}>
                        <ListItemIcon sx={{ minWidth: { xs: 18, sm: 40 } }}>
                          <CheckCircleIcon sx={{ color: '#64B464', fontSize: { xs: 10, sm: 20 } }} />
                        </ListItemIcon>
                                                  <ListItemText 
                          primary={<Typography variant="body1" sx={{ fontSize: { xs: '0.4rem', sm: '1rem' } }}>Autism Quotient</Typography>}
                          secondary={<Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.4rem', sm: '0.875rem' } }}>Score: 32/50</Typography>}
                        />
                      </ListItem>
                    </List>

                    {!isMobile && (
                      <Box sx={{ mt: 2, p: 1.5, bgcolor: 'rgba(26, 123, 140, 0.1)', borderRadius: 1 }}>
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>
                          All required questionnaires have been completed and analyzed.
                        </Typography>
                      </Box>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Section 2: Assessment Report */}
          <Grid item xs={6} md={5}>
            <Card sx={{ mb: { xs: 1, sm: 3 }, overflow: 'hidden' }}>
              <CardContent sx={{ p: { xs: 1, sm: 3 } }}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  mb: expandedSections.report ? { xs: 0.5, sm: 2 } : 0
                }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <DescriptionIcon sx={{ color: '#1A7B8C', mr: 0.5, fontSize: { xs: 12, sm: 24 } }} />
                    <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.45rem', sm: '1.25rem' } }}>
                      Assessment Report
                    </Typography>
                  </Box>
                </Box>
                
                {expandedSections.report && (
                  <>
                    <Divider sx={{ mb: { xs: 0.5, sm: 2 } }} />
                    
                    <Box sx={{ mb: { xs: 0.5, sm: 2 } }}>
                      <Typography variant="body2" sx={{ mb: 0.5, fontSize: { xs: '0.4rem', sm: '0.875rem' } }}>
                        Report generated from interview data
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={100} 
                        sx={{ height: { xs: 4, sm: 8 }, borderRadius: 4, bgcolor: 'rgba(26, 123, 140, 0.1)', '& .MuiLinearProgress-bar': { bgcolor: '#64B464' } }} 
                      />
                    </Box>
                    
                                          <Box sx={{ maxHeight: { xs: 45, sm: 300 }, overflow: 'auto', pr: 0.5 }}>
                      <Grid container spacing={1}>
                        {reportSections.slice(0, isMobile ? 6 : reportSections.length).map((section, index) => (
                          <Grid item xs={6} sm={6} key={index}>
                            <Box sx={{ 
                              display: 'flex', 
                              alignItems: 'center', 
                              p: { xs: 0.2, sm: 1 }, 
                              borderRadius: 1,
                              bgcolor: 'rgba(26, 123, 140, 0.05)',
                              mb: 0.3
                            }}>
                              <CheckCircleIcon sx={{ fontSize: { xs: 8, sm: 16 }, mr: 0.3, color: '#64B464' }} />
                              <Typography variant="body2" noWrap sx={{ fontSize: { xs: '0.4rem', sm: '0.875rem' } }}>
                                {section.name}
                              </Typography>
                            </Box>
                          </Grid>
                        ))}
                      </Grid>
                    </Box>
                    
                    <Box sx={{ mt: { xs: 0.5, sm: 2 }, p: { xs: 0.5, sm: 1.5 }, bgcolor: 'rgba(100, 180, 100, 0.1)', borderRadius: 1 }}>
                      <Typography variant="body2" sx={{ color: '#64B464', fontWeight: 500, fontSize: { xs: '0.4rem', sm: '0.875rem' } }}>
                        Report generation complete
                      </Typography>
                    </Box>
                    
                    <Box sx={{ mt: 1, display: 'flex', justifyContent: 'center' }}>
                      <Button 
                        variant="outlined" 
                        color="primary"
                        endIcon={<DescriptionIcon sx={{ fontSize: { xs: 12, sm: 20 } }} />}
                        onClick={handleOpenReport}
                        size={isMobile ? "small" : "medium"}
                        sx={{ fontSize: { xs: '0.4rem', sm: '0.875rem' }, py: { xs: 0.2, sm: 1 }, px: { xs: 0.5, sm: 1.5 } }}
                      >
                        Preview
                      </Button>
                    </Box>
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Section 3: Diagnosis Likelihood */}
          <Grid item xs={3} md={3}>
            <Card sx={{ mb: { xs: 1, sm: 3 }, overflow: 'hidden' }}>
              <CardContent sx={{ p: { xs: 1, sm: 3 } }}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  mb: expandedSections.diagnosis ? { xs: 0.5, sm: 2 } : 0
                }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <MedicalInformationIcon sx={{ color: '#1A7B8C', mr: 0.5, fontSize: { xs: 12, sm: 24 } }} />
                    <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.45rem', sm: '1.25rem' } }}>
                      Diagnosis
                    </Typography>
                  </Box>
                </Box>
                
                {expandedSections.diagnosis && (
                  <>
                    <Divider sx={{ mb: { xs: 1, sm: 3 } }} />
                    
                    <Box sx={{ textAlign: 'center', mb: { xs: 1, sm: 3 } }}>
                      <Box sx={{ 
                        position: 'relative',
                        width: { xs: 32, sm: 160 },
                        height: { xs: 32, sm: 160 },
                        margin: '0 auto',
                        borderRadius: '50%',
                        background: `conic-gradient(#1A7B8C 0% 88%, #f0f0f0 88% 100%)`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mb: { xs: 0.5, sm: 2 }
                      }}>
                        <Box sx={{ 
                          width: { xs: 26, sm: 140 }, 
                          height: { xs: 26, sm: 140 }, 
                          borderRadius: '50%', 
                          bgcolor: 'white',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexDirection: 'column'
                        }}>
                          <Typography variant="h3" sx={{ fontWeight: 600, color: '#1A7B8C', fontSize: { xs: '0.45rem', sm: '2.5rem' }, lineHeight: 1 }}>88%</Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.2rem', sm: '0.75rem' } }}>Confidence</Typography>
                        </Box>
                      </Box>
                      
                      <Box sx={{ p: { xs: 0.5, sm: 2 }, bgcolor: 'rgba(26, 123, 140, 0.1)', borderRadius: 2, mb: { xs: 0.5, sm: 2 } }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600, color: '#1A7B8C', fontSize: { xs: '0.4rem', sm: '1rem' } }}>
                          ASD Criteria Met
                        </Typography>
                      </Box>
                      
                      {!isMobile && (
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>
                          Based on comprehensive assessment of questionnaires, interview data, and behavioral observations.
                        </Typography>
                      )}
                    </Box>
                    
                    <Box sx={{ mt: { xs: 0.5, sm: 2 } }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography variant="body2" sx={{ fontSize: { xs: '0.4rem', sm: '0.875rem' } }}>DSM-5:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600, fontSize: { xs: '0.4rem', sm: '0.875rem' } }}>High</Typography>
                      </Box>
                    </Box>
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};

export default Dashboard;