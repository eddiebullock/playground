import React, { useState, useEffect } from "react";
import {
  TextField,
  Button,
  Checkbox,
  CircularProgress,
  Box,
  Typography,
  FormControlLabel,
  Card,
  Snackbar,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Collapse,
} from "@mui/material";
import { anonymizeText, generateReport } from "../services/api";
import { Document, Packer, Paragraph, HeadingLevel, AlignmentType } from "docx";
import { saveAs } from "file-saver";
import { motion } from "framer-motion";

const stripHtmlTags = (html) => {
  const tempDiv = document.createElement("div");
  tempDiv.innerHTML = html;
  return tempDiv.textContent || tempDiv.innerText || "";
};

const Anonymizer = () => {
  const [text, setText] = useState(localStorage.getItem("text") || "");
  const [transcript, setTranscript] = useState(localStorage.getItem("transcript") || "");
  const [includeTranscript, setIncludeTranscript] = useState(false);
  const [anonymizedText, setAnonymizedText] = useState("");
  const [report, setReport] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isReportLoading, setIsReportLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [snackbar, setSnackbar] = useState({ open: false, message: "", severity: "success" });

  const steps = ["Enter text", "Anonymize", "Generate report", "Download report"];

  useEffect(() => {
    localStorage.setItem("text", text);
    localStorage.setItem("transcript", transcript);
  }, [text, transcript]);

  const handleAnonymize = async () => {
    if (!text.trim() && (!includeTranscript || !transcript.trim())) {
      setError("Please enter clinician notes or upload a transcript.");
      return;
    }
    setError("");
    setIsLoading(true);
    setAnonymizedText("");
    setReport("");

    const combinedText = `${text}\n${includeTranscript && transcript ? transcript : ""}`.trim();

    try {
      const response = await anonymizeText(combinedText);
      setAnonymizedText(response.anonymized_text);
      setActiveStep(1);
      setSnackbar({ open: true, message: "Text anonymized successfully!", severity: "success" });
    } catch (err) {
      setError(err.message || "Failed to anonymize text.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!anonymizedText.trim()) {
      setError("Anonymized text is required to generate a report.");
      return;
    }
    setError("");
    setIsReportLoading(true);
    setReport("");

    try {
      const response = await generateReport(anonymizedText);
      setReport(response.report || "");
      setActiveStep(2);
      setSnackbar({ open: true, message: "Report generated successfully!", severity: "success" });
    } catch (err) {
      setError(err.message || "Failed to generate report.");
    } finally {
      setIsReportLoading(false);
    }
  };

  const handleDownloadDocx = () => {
    if (!report.trim()) {
      setError("No report available to download.");
      return;
    }

    const textContent = stripHtmlTags(report)
      .replace(/\n+/g, "\n")
      .replace(/###/g, "\n\n")
      .replace(/Key Mental Health Topic/g, "\nKey Mental Health Topic")
      .replace(/\d+\.\s/g, "")
      .replace(/Section \d+/, "")
      .replace(/(Background and Key Topics|Communication|Reciprocal Social Interaction|Restricted and Repetitive Behaviors)/g, "\n$1\n")
      .replace(/([A-Za-z\s]+):/g, "\n$1:\n")
      .replace(/[:\n]\s+/g, "\n")
      .trim();

    const lines = textContent.split("\n").map((line) => line.trim());

    const doc = new Document({
      sections: [
        {
          properties: {},
          children: [
            new Paragraph({
              text: "Autism Assessment Report",
              heading: HeadingLevel.TITLE,
              bold: true,
              alignment: AlignmentType.CENTER,
              spacing: { after: 200 },
            }),
            ...lines.flatMap((line) => {
              if (
                ["Background and Key Topics", "Communication", "Reciprocal Social Interaction", "Restricted and Repetitive Behaviors"].includes(
                  line
                )
              ) {
                return [
                  new Paragraph({ text: "", spacing: { after: 150 } }),
                  new Paragraph({
                    text: line,
                    heading: HeadingLevel.HEADING_1,
                    bold: true,
                    spacing: { after: 100 },
                  }),
                ];
              } else if (line.endsWith(":") || line.match(/^[A-Za-z\s]+$/)) {
                return [
                  new Paragraph({ text: "", spacing: { after: 50 } }),
                  new Paragraph({
                    text: line.replace(":", ""),
                    heading: HeadingLevel.HEADING_2,
                    bold: true,
                    spacing: { after: 30 },
                  }),
                ];
              } else if (line.length > 0) {
                return [
                  new Paragraph({
                    text: line,
                    spacing: { after: 20 },
                  }),
                ];
              }
              return [];
            }),
          ],
        },
      ],
    });

    Packer.toBlob(doc).then((blob) => {
      saveAs(blob, "assessment_report.docx");
    });
  };

  return (
    <Box sx={{ maxWidth: "800px", margin: "auto", textAlign: "center" }}>
      <Stepper
        activeStep={activeStep}
        alternativeLabel
        sx={{
          position: "sticky",
          top: 0,
          zIndex: 1000,
          backgroundColor: "#fff", // Matches the page background for a seamless look
          padding: 2,
          boxShadow: "0px 2px 4px rgba(0, 0, 0, 0.1)", // Optional subtle shadow for visual separation
          marginBottom: 3,
        }}
      >
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Card sx={{ padding: 3, marginBottom: 3 }}>
        <Typography variant="h6">Enter Clinician Notes</Typography>
        <TextField
          label="Enter clinician notes"
          multiline
          rows={6}
          variant="outlined"
          fullWidth
          value={text}
          onChange={(e) => setText(e.target.value)}
          error={!!error}
          helperText={error || `${text.length} characters`}
          sx={{ marginBottom: 2 }}
        />

        <FormControlLabel
          control={<Checkbox checked={includeTranscript} onChange={(e) => setIncludeTranscript(e.target.checked)} />}
          label="Include transcript"
        />

        <Collapse in={includeTranscript} timeout="auto" unmountOnExit>
          <TextField
            label="Paste assessment transcript here"
            multiline
            rows={6}
            variant="outlined"
            fullWidth
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            helperText={`${transcript.length} characters`}
            sx={{ marginTop: 2 }}
          />
        </Collapse>

        <Button
          variant="contained"
          color="primary"
          onClick={handleAnonymize}
          disabled={isLoading}
          sx={{ marginTop: 2 }}
        >
          {isLoading ? <CircularProgress size={24} color="inherit" /> : "Anonymize"}
        </Button>
      </Card>

      {anonymizedText && (
        <motion.div animate={{ opacity: 1 }} initial={{ opacity: 0 }}>
          <Card sx={{ padding: 3, marginBottom: 3 }}>
            <Typography variant="h6">Anonymized Text</Typography>
            <TextField
              value={anonymizedText}
              multiline
              rows={6}
              variant="outlined"
              fullWidth
              InputProps={{ readOnly: true }}
            />
            <Button
              variant="contained"
              color="secondary"
              onClick={handleGenerateReport}
              disabled={isReportLoading}
              sx={{ marginTop: 2 }}
            >
              {isReportLoading ? <CircularProgress size={24} color="inherit" /> : "Generate Report"}
            </Button>
          </Card>
        </motion.div>
      )}

      {report && (
        <motion.div animate={{ opacity: 1 }} initial={{ opacity: 0 }}>
          <Card sx={{ padding: 3, marginBottom: 3 }}>
            <Typography variant="h6">Assessment Report</Typography>
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
              sx={{ marginTop: 2 }}
            >
              Download as Word Document
            </Button>
          </Card>
        </motion.div>
      )}

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert onClose={() => setSnackbar({ ...snackbar, open: false })} severity={snackbar.severity}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Anonymizer;
