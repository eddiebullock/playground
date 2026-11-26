import React from "react";
import { Box, Typography, Button } from "@mui/material";

const ReportDisplay = ({ report, handleDownloadDocx }) => {
  if (!report) return null;

  return (
    <Box sx={{ marginTop: "20px", textAlign: "left" }}>
      <Typography variant="h6" gutterBottom>
        Assessment Report
      </Typography>
      <div
        style={{
          border: "1px solid #ccc",
          borderRadius: "5px",
          padding: "10px",
          minHeight: "300px",
          backgroundColor: "#fafafa",
          overflow: "auto",
        }}
        dangerouslySetInnerHTML={{ __html: report }}
      ></div>
      <Button
        variant="contained"
        color="primary"
        onClick={handleDownloadDocx}
        sx={{ marginTop: "20px" }}
      >
        Download as Word Document
      </Button>
    </Box>
  );
};

export default ReportDisplay;
