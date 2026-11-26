import React from "react";
import { Box, Typography, TextField, Button, CircularProgress } from "@mui/material";

const AnonymizedTextDisplay = ({ anonymizedText, handleGenerateReport, isReportLoading }) => {
  if (!anonymizedText) return null;

  return (
    <Box sx={{ marginTop: "20px", textAlign: "left" }}>
      <Typography variant="h6" gutterBottom>
        Anonymized Text
      </Typography>
      <TextField
        value={anonymizedText}
        multiline
        rows={6}
        variant="outlined"
        fullWidth
        InputProps={{
          readOnly: true,
        }}
      />
      <Button
        variant="contained"
        color="secondary"
        onClick={handleGenerateReport}
        disabled={isReportLoading}
        sx={{ marginTop: "20px" }}
      >
        {isReportLoading ? <CircularProgress size={24} color="inherit" /> : "Generate Assessment Report"}
      </Button>
    </Box>
  );
};

export default AnonymizedTextDisplay;
