import React, { useState } from "react";
import {
  TextField,
  Button,
  Checkbox,
  CircularProgress,
  Box,
  Typography,
  FormControlLabel,
} from "@mui/material";
import { anonymizeText, generateReport } from "../services/api";
import { Document, Packer, Paragraph, HeadingLevel, AlignmentType } from "docx";
import { saveAs } from "file-saver";

const Anonymizer = () => {
  const [text, setText] = useState("");
  const [transcript, setTranscript] = useState(""); // For the assessment transcript
  const [includeTranscript, setIncludeTranscript] = useState(false); // Checkbox state
  const [anonymizedText, setAnonymizedText] = useState("");
  const [report, setReport] = useState(""); // Raw HTML returned by your API
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isReportLoading, setIsReportLoading] = useState(false);

  const handleAnonymize = async () => {
    if (!text.trim() && (!includeTranscript || !transcript.trim())) {
      setError("Please enter clinician notes or upload a transcript.");
      return;
    }
    setError("");

    const combinedText = `${text}\n${includeTranscript && transcript ? transcript : ""}`.trim();

    setIsLoading(true);
    setAnonymizedText("");
    setReport("");

    try {
      const response = await anonymizeText(combinedText);
      setAnonymizedText(response.anonymized_text);
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
      setReport(response.report || ""); // Raw HTML or plain text
    } catch (err) {
      setError(err.message || "Failed to generate report.");
    } finally {
      setIsReportLoading(false);
    }
  };

  const stripHtmlTags = (html) => {
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = html;
    return tempDiv.textContent || tempDiv.innerText || "";
  };

  const handleDownloadDocx = () => {
    if (!report.trim()) {
      setError("No report available to download.");
      return;
    }
  
    // Strip HTML and normalize text
    const textContent = stripHtmlTags(report)
      .replace(/\n+/g, "\n") // Remove excessive line breaks
      .replace(/###/g, "\n\n") // Ensure proper spacing for subheadings
      .replace(/Key Mental Health Topic/g, "\nKey Mental Health Topic") // Ensure line breaks before topics
      .replace(/\d+\.\s/g, "") // Remove numbered lists
      .replace(/Section \d+/, "") // Remove "Section 1, 2, 3..."
      .replace(/(Background and Key Topics|Communication|Reciprocal Social Interaction|Restricted and Repetitive Behaviors)/g, "\n$1\n") // Ensure section titles are spaced correctly
      .replace(/([A-Za-z\s]+):/g, "\n$1:\n") // Ensure subheadings have a line break
      .replace(/[:\n]\s+/g, "\n") // Ensure consistent spacing
      .trim();
  
    const lines = textContent.split("\n").map((line) => line.trim());
  
    const doc = new Document({
      sections: [
        {
          properties: {},
          children: [
            new Paragraph({
              text: "Autism Assessment Report",
              heading: HeadingLevel.TITLE, // Makes it the largest title
              bold: true,
              alignment: AlignmentType.CENTER, // Centers it
              spacing: { after: 200 }, // Adds extra space after the title
            }),
            ...lines.flatMap((line, index, arr) => {
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
              }
  
              // Detect subheadings (like "Academic History/Scholarly Skills") and bold them
              else if (line.endsWith(":") || line.match(/^[A-Za-z\s]+$/)) {
                return [
                  new Paragraph({ text: "", spacing: { after: 50 } }),
                  new Paragraph({
                    text: line.replace(":", ""), // Remove ":" to keep it clean
                    heading: HeadingLevel.HEADING_2,
                    bold: true,
                    spacing: { after: 30 },
                  }),
                ];
              }
  
              // Ensure paragraphs are properly spaced
              else if (line.length > 0) {
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
  
    // Generate and download the Word document
    Packer.toBlob(doc).then((blob) => {
      saveAs(blob, "assessment_report.docx");
    });
  };
  
  return (
    <Box sx={{ maxWidth: "600px", margin: "auto", textAlign: "center" }}>
      <TextField
        label="Enter clinician notes"
        multiline
        rows={6}
        variant="outlined"
        fullWidth
        value={text}
        onChange={(e) => setText(e.target.value)}
        error={!!error}
        helperText={error}
        sx={{ marginBottom: "20px" }}
      />

      <Typography variant="h6" gutterBottom>
        Add assessment transcript
      </Typography>
      <FormControlLabel
        control={
          <Checkbox
            checked={includeTranscript}
            onChange={(e) => setIncludeTranscript(e.target.checked)}
          />
        }
        label="Include transcript"
        sx={{ display: "block", marginBottom: "20px", textAlign: "left" }}
      />
      {includeTranscript && (
        <TextField
          label="Paste assessment transcript here"
          multiline
          rows={6}
          variant="outlined"
          fullWidth
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          sx={{ marginBottom: "20px" }}
        />
      )}

      <Button
        variant="contained"
        color="primary"
        onClick={handleAnonymize}
        disabled={isLoading}
        sx={{ marginBottom: "20px" }}
      >
        {isLoading ? <CircularProgress size={24} color="inherit" /> : "Anonymize"}
      </Button>

      {anonymizedText && (
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
      )}

      {report && (
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
      )}
    </Box>
  );
};

export default Anonymizer;
