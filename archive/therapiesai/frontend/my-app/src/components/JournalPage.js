import { auth } from '../firebase';
import React, { useState } from 'react';

export default function JournalPage() {
  const [entry, setEntry] = useState('');
  const [happiness, setHappiness] = useState(5);
  const [recording, setRecording] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  let recognition;

  // Initialize Web Speech API
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = entry; // Use the existing entry as the base

      Array.from(event.results).forEach((result) => {
        if (result.isFinal) {
          finalTranscript += ' ' + result[0].transcript; // Append final result
        } else {
          interimTranscript += result[0].transcript; // Handle interim results
        }
      });

      setEntry(finalTranscript.trim()); // Update entry with only final results
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      alert('An error occurred during speech recognition. Please try again.');
      setRecording(false);
    };
  } else {
    alert('Voice input is not supported in your browser.');
  }

  const toggleRecording = () => {
    if (recording) {
      recognition.stop();
      setRecording(false);
    } else {
      recognition.start();
      setRecording(true);
    }
  };

  const handleSave = async () => {
    if (!entry.trim()) {
      alert('Journal entry cannot be empty.');
      return;
    }

    try {
      const user = auth.currentUser;
      if (!user) {
        alert('You must be logged in to save a journal entry.');
        return;
      }

      const response = await fetch('http://127.0.0.1:5000/add-entry', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: user.uid,
          entry,
          happiness,
        }),
      });

      const result = await response.json();
      if (response.ok) {
        alert(`Journal entry saved successfully! Current streak: ${result.streak} days`);
        setEntry('');
        setHappiness(5);
      } else {
        console.error('Error:', result);
        alert('Failed to save the entry.');
      }
    } catch (error) {
      console.error('Error saving journal entry:', error);
      alert('Failed to save the entry. Please try again.');
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-50 p-4">
      <h1 className="text-2xl font-bold mb-4">Journal Your Day</h1>
      <textarea
        placeholder="How are you feeling today?"
        value={entry}
        onChange={(e) => setEntry(e.target.value)}
        className="w-full max-w-md p-3 border rounded-lg mb-4 h-40 resize-none"
      />
      <div className="mb-4">
        <button
          onClick={toggleRecording}
          className={`px-4 py-2 rounded-lg ${
            recording ? 'bg-red-500' : 'bg-blue-500'
          } text-white`}
        >
          {recording ? 'Stop Recording' : 'Start Recording'}
        </button>
      </div>
      <label className="mb-2">Happiness Score: {happiness}</label>
      <input
        type="range"
        min="1"
        max="10"
        value={happiness}
        onChange={(e) => setHappiness(Number(e.target.value))}
        className="w-full max-w-md mb-4"
      />
      <button
        onClick={handleSave}
        className="bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600"
      >
        Save Entry
      </button>
      {message && <p className="text-green-500 mt-4">{message}</p>}
      {error && <p className="text-red-500 mt-4">{error}</p>}
    </div>
  );
}
