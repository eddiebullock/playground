// src/components/mockups/DashboardMockup.jsx
import React from 'react';
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
  ListItemAvatar, 
  ListItemText, 
  LinearProgress,
  Grid
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import EventNoteIcon from '@mui/icons-material/EventNote';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

const DashboardMockup = () => {
  return (
    <Box sx={{ p: 0 }}>
      {/* Dashboard Header */}
      <Box sx={{ p: 2, bgcolor: '#1A7B8C', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6">DiagnosisAI Dashboard</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip 
            label="Dr. Sarah" 
            size="small" 
            avatar={<Avatar sx={{ bgcolor: 'white', color: '#1A7B8C' }}>S</Avatar>}
            sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} 
          />
        </Box>
      </Box>

      {/* Dashboard Content */}
      <Box sx={{ p: 3 }}>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>Today: Monday, March 3, 2025</Typography>
        
        {/* Stats Cards */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6} sm={3}>
            <Card elevation={0} sx={{ bgcolor: 'rgba(26, 123, 140, 0.1)' }}>
              <CardContent sx={{ p: 2 }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: '#1A7B8C' }}>12</Typography>
                <Typography variant="body2" color="text.secondary">Assessments Today</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card elevation={0} sx={{ bgcolor: 'rgba(230, 181, 74, 0.1)' }}>
              <CardContent sx={{ p: 2 }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: '#E6B54A' }}>42</Typography>
                <Typography variant="body2" color="text.secondary">This Month</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card elevation={0} sx={{ bgcolor: 'rgba(255, 127, 110, 0.1)' }}>
              <CardContent sx={{ p: 2 }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: '#FF7F6E' }}>8</Typography>
                <Typography variant="body2" color="text.secondary">Pending Reports</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card elevation={0} sx={{ bgcolor: 'rgba(100, 180, 100, 0.1)' }}>
              <CardContent sx={{ p: 2 }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: '#64B464' }}>94%</Typography>
                <Typography variant="body2" color="text.secondary">Completion Rate</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Status Chart */}
        <Box sx={{ mb: 4 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Assessment Pipeline</Typography>
            <Chip 
              size="small" 
              label="View All" 
              variant="outlined" 
              sx={{ fontSize: '0.75rem', height: 24 }}
            />
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#1A7B8C' }}></Box>
              <Typography variant="caption">Complete (24)</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#E6B54A' }}></Box>
              <Typography variant="caption">In Progress (16)</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#FF7F6E' }}></Box>
              <Typography variant="caption">Scheduled (8)</Typography>
            </Box>
          </Box>
          
          <Box sx={{ width: '100%', height: 20, bgcolor: '#f0f0f0', borderRadius: 10, overflow: 'hidden', display: 'flex' }}>
            <Box sx={{ width: '50%', height: '100%', bgcolor: '#1A7B8C' }}></Box>
            <Box sx={{ width: '33%', height: '100%', bgcolor: '#E6B54A' }}></Box>
            <Box sx={{ width: '17%', height: '100%', bgcolor: '#FF7F6E' }}></Box>
          </Box>
        </Box>

        {/* Recent Patients */}
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Today's Patients</Typography>
            <Chip 
              size="small" 
              label="View Schedule" 
              variant="outlined" 
              sx={{ fontSize: '0.75rem', height: 24 }}
            />
          </Box>
          
          <List sx={{ p: 0 }}>
            <ListItem sx={{ px: 2, py: 1, borderRadius: 1, bgcolor: 'rgba(26, 123, 140, 0.05)', mb: 1 }}>
              <ListItemAvatar>
                <Avatar sx={{ bgcolor: '#1A7B8C', width: 36, height: 36 }}>AT</Avatar>
              </ListItemAvatar>
              <ListItemText 
                primary={<Typography variant="body2" sx={{ fontWeight: 500 }}>Alex Thompson (8)</Typography>}
                secondary={
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <Chip label="ASD" size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
                    <Box sx={{ display: 'flex', alignItems: 'center', fontSize: '0.75rem' }}>
                      <CheckCircleIcon sx={{ fontSize: 14, mr: 0.5, color: 'success.main' }} />
                      Complete
                    </Box>
                  </Box>
                }
              />
            </ListItem>
            
            <ListItem sx={{ px: 2, py: 1, borderRadius: 1, bgcolor: 'rgba(230, 181, 74, 0.05)', mb: 1 }}>
              <ListItemAvatar>
                <Avatar sx={{ bgcolor: '#E6B54A', width: 36, height: 36 }}>MW</Avatar>
              </ListItemAvatar>
              <ListItemText 
                primary={<Typography variant="body2" sx={{ fontWeight: 500 }}>Maya Williams (12)</Typography>}
                secondary={
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <Chip label="ADHD" size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
                    <Box sx={{ display: 'flex', alignItems: 'center', fontSize: '0.75rem' }}>
                      <AccessTimeIcon sx={{ fontSize: 14, mr: 0.5, color: '#E6B54A' }} />
                      In Progress
                    </Box>
                  </Box>
                }
              />
            </ListItem>
            
            <ListItem sx={{ px: 2, py: 1, borderRadius: 1, bgcolor: 'rgba(255, 127, 110, 0.05)' }}>
              <ListItemAvatar>
                <Avatar sx={{ bgcolor: '#FF7F6E', width: 36, height: 36 }}>EJ</Avatar>
              </ListItemAvatar>
              <ListItemText 
                primary={<Typography variant="body2" sx={{ fontWeight: 500 }}>Ethan Johnson (7)</Typography>}
                secondary={
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <Chip label="ASD/ADHD" size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
                    <Box sx={{ display: 'flex', alignItems: 'center', fontSize: '0.75rem' }}>
                      <EventNoteIcon sx={{ fontSize: 14, mr: 0.5, color: '#FF7F6E' }} />
                      15:30 - 17:00
                    </Box>
                  </Box>
                }
              />
            </ListItem>
          </List>
        </Box>
      </Box>
    </Box>
  );
};

export default DashboardMockup;