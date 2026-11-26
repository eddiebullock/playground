// src/firebase.js

// Import the functions you need from the Firebase SDKs
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth";
import { getFirestore, setLogLevel } from "firebase/firestore";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBNfDM3MnwCJ3LnIeLOGVF-G4Vi4VXZLDI",
  authDomain: "therapiesai.firebaseapp.com",
  projectId: "therapiesai",
  storageBucket: "therapiesai.appspot.com",
  messagingSenderId: "138200656971",
  appId: "1:138200656971:web:0099609221e4e083d071ce",
  measurementId: "G-FMQ309R2TN",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase services
const analytics = getAnalytics(app);
const auth = getAuth(app);
const db = getFirestore(app);

// Enable Firestore logging using setLogLevel
setLogLevel("debug");

// Export Firebase services
export { analytics, auth, db };
export default app;
