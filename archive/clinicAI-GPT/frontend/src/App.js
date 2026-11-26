import React from "react";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import Anonymizer from "./components/Anonymizer";

const theme = createTheme({
  palette: {
    primary: {
      main: "#1976d2", // Blue
    },
    secondary: {
      main: "#dc004e", // Red
    },
  },
  typography: {
    fontFamily: "Roboto, Arial, sans-serif",
  },
});

const App = () => {
  return (
    <ThemeProvider theme={theme}>
      <Anonymizer />
    </ThemeProvider>
  );
};

export default App;
