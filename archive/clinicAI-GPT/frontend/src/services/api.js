// services/api.js
export const anonymizeText = async (text) => {
  const response = await fetch("http://localhost:5000/api/anonymize", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) throw new Error("Failed to anonymize text.");
  return response.json();
};

export const generateReport = async (anonymizedText) => {
  const response = await fetch("http://localhost:5000/api/generate_report", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ anonymized_text: anonymizedText }),
  });
  if (!response.ok) throw new Error("Failed to generate report.");
  return response.json();
};
